#ifndef VERTEXCLOUD_H
#define VERTEXCLOUD_H

#include <nanoflann.hpp>
#include <vector>

#include "af/eigen_extension.h"

template<typename Vertex>
struct VertexCloud {
    std::vector<Vertex> vec_;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return vec_.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline typename Vertex::value_type kdtree_get_pt(const size_t idx, const size_t dim) const {
        return vec_[idx][dim];
        // if (dim == 0) return pts[idx].x;
        // else if (dim == 1) return pts[idx].y;
        // else return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template<class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
        return false;
    }

    inline void clear() noexcept { vec_.clear(); }
    inline bool empty() const noexcept { return vec_.empty(); }
    inline std::size_t size() const noexcept { return vec_.size(); }
    inline void push_back(const Vertex& vertex) { vec_.push_back(vertex); }
    inline void push_back(Vertex&& vertex) { vec_.push_back(std::move(vertex)); }
};

using VertexCloundL2Tree =
    nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, VertexCloud<Vec3f> >, VertexCloud<Vec3f>, 3, unsigned int>;

#endif