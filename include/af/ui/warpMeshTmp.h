#ifndef WARPMESHTMP_H
#define WARPMESHTMP_H

#include "af/MotionGraph.h"
#include "af/ui/BufferVtkPoints.h"

namespace af {
void warpMesh(BufferVtkPoints& pointsWarped, BufferVtkPoints& points, const MotionGraph& graph);
bool savePlyTmp(const std::string &filename, BufferVtkPoints& points, const std::vector<Vec3i>& faces);
}  // namespace af

#endif