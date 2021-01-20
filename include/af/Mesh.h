#ifndef MESH_H
#define MESH_H

#include <af/VertexCloud.h>

class Mesh
{
public:
    Mesh() = default;
    virtual ~Mesh() = default;

    VertexCloud<Vec3f>& vertexCloud() { return vertexCloud_; }
    std::vector<Vec3b>& colors() { return colors_; }
    std::vector<Vec3i>& faces() { return faces_; }
    const VertexCloud<Vec3f>& vertexCloud() const { return vertexCloud_; }
    const std::vector<Vec3b>& colors() const { return colors_; }
    const std::vector<Vec3i>& faces() const { return faces_; }

    unsigned int addVertex(const Vec3f &v, const Vec3b &c);
    void clear();

    bool savePly(const std::string &filename) const;
    bool loadPly(const std::string &filename);

protected:

    VertexCloud<Vec3f> vertexCloud_;
    std::vector<Vec3b> colors_;
    std::vector<Vec3i> faces_;
};

#endif
