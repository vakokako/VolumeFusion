#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <fstream>

#include "af/DataTypes.cuh"
#include "af/GraphBuilder.h"
#include "af/Helper.cuh"
#include "af/MotionGraph.h"
#include "af/VertexManipulation.cuh"
#include "af/ComputeCorrespondence.cuh"
#include "af/ui/BufferVtkPoints.h"
#include "af/ui/warpMeshTmp.h"

namespace af {
void warpMesh(BufferVtkPoints& pointsWarped, BufferVtkPoints& points, const MotionGraph& graph) {
    std::cout << "starting mesh warping\n";

    std::size_t pointsCount = points.points()->GetNumberOfPoints();

    std::vector<Vec3f> pointsHost(pointsCount);
    thrust::device_vector<Vec3f> warpedMesh_d(pointsCount);
    af::DeviceKnnWeighted<Constants::motionGraphKNN> meshKnn_d(pointsCount);  // 24000000 * 3

    thrust::device_vector<Mat4f> motionGraphTransforms_d(graph.graph().size());  // 640000
    thrust::device_vector<float> motionGraphRadiuses_d(graph.graph().size());    // 40000

    std::copy_n(static_cast<Vec3f*>(points.points()->GetVoidPointer(0)), pointsCount, pointsHost.begin());
    thrust::copy_n(pointsHost.begin(), pointsCount, warpedMesh_d.begin());
    CUDA_CHECK;

    auto countNotIdentity = std::count_if(graph.transforms().begin(), graph.transforms().end(), [](const auto& mat){return !(mat.isIdentity(0.01f));});
    copyToDevice(motionGraphTransforms_d.data().get(), graph.transforms().data(), graph.graph().size());
    copyToDevice(motionGraphRadiuses_d.data().get(), graph.radiuses().data(), graph.graph().size());

    std::vector<Vecui<Constants::motionGraphKNN>> meshKnnIdxs;
    std::vector<Vecf<Constants::motionGraphKNN>> meshKnnDists;

    af::getKnnData(meshKnnIdxs, meshKnnDists, graph, pointsHost);
    copyToDevice(meshKnn_d.idxsPtr(), meshKnnIdxs.data(), pointsCount);
    copyToDevice(meshKnn_d.distsPtr(), meshKnnDists.data(), pointsCount);
    meshKnn_d.setSize(pointsCount);

    af::runComputeWarpWeightsKernel(meshKnn_d.weightsPtr(), motionGraphRadiuses_d.data().get(), meshKnn_d.idxsPtr(),
                                    meshKnn_d.distsPtr(), pointsCount);
    af::runWarpKernel(warpedMesh_d.data().get(), warpedMesh_d.data().get(), motionGraphTransforms_d.data().get(), meshKnn_d.idxsPtr(),
                      meshKnn_d.weightsPtr(), pointsCount);
    CUDA_CHECK;

    pointsWarped.loadFromDevice(warpedMesh_d.data().get(), pointsCount);

    std::cout << "pointsCount : " << pointsCount << "\n";
    std::cout << "countNotIdentity : " << countNotIdentity << "\n";
    std::cout << "mesh warped\n";
}

bool savePlyTmp(const std::string &filename, BufferVtkPoints& points, const std::vector<Vec3i>& faces) {
    std::size_t pointsCount = points.points()->GetNumberOfPoints();
    if (!pointsCount) {
        std::cout << "af::savePlyTmp(): Vertex cloud is empty.\n";
        return false;
    }

    std::vector<Vec3f> pointsHost(pointsCount);
    std::copy_n(static_cast<Vec3f*>(points.points()->GetVoidPointer(0)), pointsCount, pointsHost.begin());

    std::ofstream plyFile;
    plyFile.open(filename.c_str());
    if (!plyFile.is_open()) {
        std::cout << "af::savePlyTmp():" << filename << " couldn't be open.\n";
        return false;
    }

    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << (int)pointsHost.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    // plyFile << "property uchar red" << std::endl;
    // plyFile << "property uchar green" << std::endl;
    // plyFile << "property uchar blue" << std::endl;
    plyFile << "element face " << (int)faces.size() << std::endl;
    plyFile << "property list uchar int vertex_indices" << std::endl;
    plyFile << "end_header" << std::endl;

    // write vertices
    for (size_t i = 0; i < pointsHost.size(); i++) {
        plyFile << pointsHost[i][0] << " " << pointsHost[i][1] << " " << pointsHost[i][2];
        // plyFile << " " << (int)0 << " " << (int)0 << " " << (int)0;
        plyFile << std::endl;
    }

    // write faces
    for (size_t i = 0; i < faces.size(); i++) {
        plyFile << "3 " << (int)faces[i][0] << " " << (int)faces[i][1] << " " << (int)faces[i][2] << std::endl;
    }

    plyFile.close();
    std::cout << "af::savePlyTmp():" << filename << " saved.\n";

    return true;
}
}  // namespace af