
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_ACCELERATORS_CNN_AGGLOMERATIVE_BVH_H
#define PBRT_ACCELERATORS_CNN_AGGLOMERATIVE_BVH_H

// accelerators/cnn-agglomerative-bvh.h*
#include "pbrt.h"
#include "primitive.h"
#include <atomic>

#include <Eigen/Dense>

struct CNNBVHNode;
struct CNNAgglomerativeBVHPrimitiveInfo;

class CNNAgglomerativeBVHAccel : public Aggregate {
    public:
    CNNAgglomerativeBVHAccel(const std::vector<std::shared_ptr<Primitive>> &p,
                             int frameWidth, int frameHeight);
    Bounds3f WorldBound() const;
    ~CNNAgglomerativeBVHAccel();

    bool Intersect(const Ray &ray, SurfaceInteraction *isect) const;
    bool IntersectP(const Ray &ray) const;

    private:
    CNNBVHNode* toTree(
      const Eigen::MatrixXd &dendrogram,
      const std::vector<CNNAgglomerativeBVHPrimitiveInfo> &info,
      int primitiveSize);

    int m_frameWidth;
    int m_frameHeight;

    std::vector<std::shared_ptr<Primitive>> m_primitives;
    std::vector<CNNBVHNode*> m_nodes;
};

std::shared_ptr<CNNAgglomerativeBVHAccel> CreateCNNAgglomerativeBVHAccelerator(
    const std::vector<std::shared_ptr<Primitive>> &prims, const ParamSet &ps);

#endif  // PBRT_ACCELERATORS_CNN_AGGLOMERATIVE_BVH_H