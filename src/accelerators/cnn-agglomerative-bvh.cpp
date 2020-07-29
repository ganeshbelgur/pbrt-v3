/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is an extension to pbrt written by Ganesh Belgur.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// accelerators/cnn-agglomerative-bvh.cpp*
#include "accelerators/bvh.h"
#include "accelerators/cnn-agglomerative-bvh.h"
#include "interaction.h"
#include "paramset.h"
#include "stats.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// BVHAccel Local Declarations
struct CNNAgglomerativeBVHPrimitiveInfo {
    CNNAgglomerativeBVHPrimitiveInfo() {}
    CNNAgglomerativeBVHPrimitiveInfo(
        size_t primitiveNumber,
        const Bounds3f &bounds)
        : primitiveNumber(primitiveNumber),
          bounds(bounds),
          centroid(.5f * bounds.pMin + .5f * bounds.pMax),
          normalized_volume(0.0f),
          pixel(Point2i(0, 0)),
          volume((bounds.pMax.x - bounds.pMin.x) 
            * (bounds.pMax.y - bounds.pMin.y) 
            * (bounds.pMax.z - bounds.pMin.z)) {}
    size_t primitiveNumber;
    Bounds3f bounds;
    Point3f centroid;
    Point2i pixel;
    double volume;
    double normalized_volume;
};

struct CNNBVHNode {
  CNNBVHNode(
    int id,
    Bounds3f bounds,
    CNNBVHNode* leftChild = nullptr,
    CNNBVHNode* rightChild = nullptr):
    m_id(id),
    m_bounds(bounds), m_leftChild(leftChild),
    m_rightChild(rightChild) {}
  int m_id;
  Bounds3f m_bounds;
  CNNBVHNode *m_leftChild;
  CNNBVHNode *m_rightChild;
};

void ShowImg(
    const std::vector<std::vector<double>>& img) {
  cv::Mat sp_img(img.size(), img.at(0).size(), CV_64FC1);
  for (int i = 0; i < sp_img.rows; ++i) {
    for (int j = 0; j < sp_img.cols; ++j) {
      sp_img.at<double>(i, j) = img.at(i).at(j); //Intensity value
    }
  }
  cv::imshow("Intensity Image", sp_img);
  cv::waitKey(0);
}

void getProjection(const Point3f& point,
                   const double& fov_rad,
                   const double& fov_down_rad,
                   int frameWidth,
                   int frameHeight,
                   int* pixel_v,
                   int* pixel_u) {
    double range = sqrt(
         point.x * point.x + 
         point.y * point.y + 
         point.z * point.z);
    //  Getting the angle of all the points
    auto yaw = atan2(point.y, point.x);
    auto pitch = asin(point.z / range);
    // Get projections in image coords and normalizing
    double v = 0.5 * (yaw / M_PI + 1.0);
    double u = 1.0 - (pitch + std::abs(fov_down_rad)) / fov_rad;
    // Scaling as per the lidar config given
    v *= frameWidth;
    u *= frameHeight;
    // round and clamp for use as index
    v = floor(v);
    v = std::min(double(frameWidth - 1), v);
    v = std::max(0.0, v);
    *pixel_v = int(v);

    u = floor(u);
    u = std::min(double(frameHeight - 1), u);
    u = std::max(0.0, u);
    *pixel_u = int(u);
}

void makeSphericalProjection( 
    const std::vector<CNNAgglomerativeBVHPrimitiveInfo>& primitiveInfo,
    std::vector<std::vector<double>>& frame,
    std::vector<Point2i>& pixelLocations,
    int frameWidth,
    int frameHeight,
    int fov_up = 180,
    int fov_down = -180) {
  // Converting to Radians
  double fov_up_rad = (fov_up / 180) * M_PI;
  double fov_down_rad = (fov_down / 180) * M_PI;

  // Getting total Field of View
  double fov_rad = std::abs(fov_up_rad) + std::abs(fov_down_rad);

  for (auto const &info : primitiveInfo) {
    // Getting Pixel from Point
    int pixel_v = 0;
    int pixel_u = 0;
    
    getProjection(
        info.centroid,
        fov_rad,
        fov_down_rad,
        frameWidth,
        frameHeight,
        &pixel_v,
        &pixel_u
    );
    
    // frame.at(pixel_u).at(pixel_v) = info.normalized_volume;
    frame.at(pixel_u).at(pixel_v) = 0.8f;
    Point2i pixel(pixel_u, pixel_v);
    pixelLocations.push_back(pixel);
  }
}

CNNBVHNode* CNNAgglomerativeBVHAccel::toTree(
  const Eigen::MatrixXd &dendrogram,
  const std::vector<CNNAgglomerativeBVHPrimitiveInfo> &info,
  int primitiveSize) {
  m_nodes = std::vector<CNNBVHNode*>(primitiveSize * 2 - 1);
  for(int i = 0; i < primitiveSize; ++i)
  {
    m_nodes[i] = new CNNBVHNode(i, info[i].bounds);
  }

  CNNBVHNode *currentRoot = nullptr;
  for(int i = 0; i < primitiveSize-1; ++i)
  {
    int fi = int(dendrogram(i, 0));
    int fj = int(dendrogram(i, 1));

    currentRoot = new CNNBVHNode(
      i + primitiveSize,
      Union(m_nodes[fi]->m_bounds, m_nodes[fj]->m_bounds),
      m_nodes[fi],
      m_nodes[fj]);

    m_nodes[i + primitiveSize] = currentRoot;
  }

  return currentRoot;
}

// CNNAgglomerativeBVHAccel Method Definitions
CNNAgglomerativeBVHAccel::CNNAgglomerativeBVHAccel(
    const std::vector<std::shared_ptr<Primitive>> &p,
    int frameWidth = 1024, int frameHeight = 512) 
    : m_primitives(p),
      m_frameWidth(frameWidth),
      m_frameHeight(frameHeight) {
    // Cannot generate a BVH if there are no primitives
    if (m_primitives.size() == 0) return;

    double max_volume = 0.0;
    std::vector<CNNAgglomerativeBVHPrimitiveInfo> 
        primitiveInfo(m_primitives.size());
    Eigen::MatrixXd m(m_primitives.size(), 3);
    for (size_t i = 0; i < m_primitives.size(); ++i) {
        primitiveInfo[i] = {i, m_primitives[i]->WorldBound()};
        if (max_volume < primitiveInfo[i].volume)
            max_volume = primitiveInfo[i].volume;
        m(i, 0) = primitiveInfo[i].centroid.x;
        m(i, 1) = primitiveInfo[i].centroid.y;
        m(i, 2) = primitiveInfo[i].centroid.z;
    }
    double inv_max_volume = 1.0 / max_volume;
    for(auto &item : primitiveInfo)
        item.normalized_volume = item.volume * inv_max_volume;

    // Agglomerative clustering will be
    // performed on the pixel values
    std::vector<Point2i> pixelLocations;
    std::vector<std::vector<double>> frame(
      m_frameHeight,
      std::vector<double>(m_frameWidth, 0.0));
    makeSphericalProjection(
        primitiveInfo,
        frame,
        pixelLocations,
        frameWidth,
        frameHeight);

    // Prediction from the model
    std::string distanceMode = "single";

    // Binding and call to Scipy's linkage method for clustering
    py::scoped_interpreter python;
    auto hierarchy = 
      py::module::import("scipy.cluster.hierarchy");
    auto ddData = hierarchy.attr("linkage")(
      m,
      py::str(std::string(distanceMode)));
    Eigen::MatrixXd dendrogram =
      ddData.cast<Eigen::MatrixXd>();

    // Converting dendrogram to binary tree
    toTree(dendrogram, primitiveInfo, m_primitives.size());
}

bool CNNAgglomerativeBVHAccel::Intersect(
  const Ray &ray,
  SurfaceInteraction *isect) const {
    if (!m_nodes.size()) return false;
    bool hit = false;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {invDir.x < 0, invDir.y < 0, invDir.z < 0};
    int nodesToVisit[64];
    int toVisitOffset = 0, currentNodeIndex = 2 * (m_primitives.size() - 1);

    while (true) {
      CNNBVHNode *currentNode = m_nodes[currentNodeIndex];
      if (currentNode->m_bounds.IntersectP(ray, invDir, dirIsNeg))
      {
        if (!currentNode->m_leftChild && !currentNode->m_rightChild) {
          if (m_primitives[currentNodeIndex]->Intersect(
                            ray, isect))
            hit = true;
          if (toVisitOffset == 0) break;
          currentNodeIndex = nodesToVisit[--toVisitOffset];
        } else {
          currentNodeIndex = currentNode->m_leftChild->m_id;
          nodesToVisit[toVisitOffset++] = currentNode->m_rightChild->m_id;
        }
      } else {
        if (toVisitOffset == 0) break;
        currentNodeIndex = nodesToVisit[--toVisitOffset];
      }
    }
    return hit;
}

bool CNNAgglomerativeBVHAccel::IntersectP(const Ray &ray) const {
    if (!m_nodes.size()) return false;
    bool hit = false;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {invDir.x < 0, invDir.y < 0, invDir.z < 0};
    int nodesToVisit[64];
    int toVisitOffset = 0, currentNodeIndex = 2 * (m_primitives.size() - 1);

    while (true) {
      CNNBVHNode *currentNode = m_nodes[currentNodeIndex];
      if (currentNode->m_bounds.IntersectP(ray, invDir, dirIsNeg))
      {
        if (!currentNode->m_leftChild && !currentNode->m_rightChild) {
          if (m_primitives[currentNodeIndex]->IntersectP(
                            ray))
            return true;
          if (toVisitOffset == 0) break;
          currentNodeIndex = nodesToVisit[--toVisitOffset];
        } else {
          currentNodeIndex = currentNode->m_leftChild->m_id;
          nodesToVisit[toVisitOffset++] = currentNode->m_rightChild->m_id;
        }
      } else {
        if (toVisitOffset == 0) break;
        currentNodeIndex = nodesToVisit[--toVisitOffset];
      }
    }
    return false;
}

Bounds3f CNNAgglomerativeBVHAccel::WorldBound() const {
  return !m_nodes.empty() ? 
    m_nodes[2 * (m_primitives.size() - 1)]->m_bounds : Bounds3f();
}

CNNAgglomerativeBVHAccel::~CNNAgglomerativeBVHAccel()
{
}

std::shared_ptr<CNNAgglomerativeBVHAccel> CreateCNNAgglomerativeBVHAccelerator(
    const std::vector<std::shared_ptr<Primitive>> &prims, const ParamSet &ps) {
  return std::make_shared<CNNAgglomerativeBVHAccel>(prims);
}