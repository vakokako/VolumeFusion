#ifndef DEBUGHELPER_H
#define DEBUGHELPER_H

#include <af/Constants.h>
#include <af/eigen_extension.h>
#include <thrust/device_vector.h>

#include <fstream>
#include <string>

namespace debug {

inline void saveMesh(const std::vector<Vec3f>& mesh, const std::string& filename) {
    std::ofstream meshFile;
    meshFile.open(Constants::dataFolder + "/testDump/" + filename + ".txt");
    if (!meshFile.is_open())
        throw(std::runtime_error("saveMesh(): File couldn't be open."));

    // write vertices
    for (size_t i = 0; i < mesh.size(); i++) {
        meshFile << mesh[i][0] << " " << mesh[i][1] << " " << mesh[i][2] << std::endl;
    }

    meshFile.close();
}

inline void saveMesh(const thrust::device_vector<Vec3f>& mesh, const std::string& filename, unsigned int size = 0) {
    unsigned int meshSize = (size != 0) ? size : mesh.size();
    std::vector<Vec3f> meshHost(meshSize);
    thrust::copy_n(mesh.begin(), meshSize, meshHost.begin());
    saveMesh(meshHost, filename);
}

inline void saveMesh(Vec3f* mesh, const std::string& filename, unsigned int meshSize) {
    thrust::device_ptr<Vec3f> meshPtr_d(mesh);
    thrust::device_vector<Vec3f> wrapper_d(meshPtr_d, meshPtr_d + meshSize);
    saveMesh(wrapper_d, filename);
}

template<class T, int W, int H>
void print(const thrust::device_vector<Eigen::Matrix<T, W, H>>& matrix,
           unsigned int h,
           unsigned int w,
           const std::string& filename = "") {
    std::ofstream outFile;
    if (!filename.empty()) {
        outFile.open(Constants::dataFolder + "/testDump/" + filename + ".txt");
        if (!outFile.is_open())
            throw(std::runtime_error("saveMesh(): File couldn't be open."));
    }
    auto& output = !filename.empty() ? outFile : std::cout;

    std::vector<Eigen::Matrix<T, W, H>> hostCopy(matrix.size());
    thrust::copy_n(matrix.begin(), matrix.size(), hostCopy.begin());

    for (std::size_t i = 0; i < h; ++i) {
        for (std::size_t j = 0; j < w; ++j) {
            output << hostCopy[i * w + j].transpose() << " ";
        }
        output << "\n";
    }
    output << "\n";
}

template<class Type>
void print(const thrust::device_vector<Type>& matrix, unsigned int h, unsigned int w, const std::string& filename = "") {
    std::ofstream outFile;
    if (!filename.empty()) {
        outFile.open(Constants::dataFolder + "/testDump/" + filename + ".txt");
        if (!outFile.is_open())
            throw(std::runtime_error("saveMesh(): File couldn't be open."));
    }
    auto& output = !filename.empty() ? outFile : std::cout;

    std::vector<Type> hostCopy(matrix.size());
    thrust::copy_n(matrix.begin(), matrix.size(), hostCopy.begin());

    for (std::size_t i = 0; i < h; ++i) {
        for (std::size_t j = 0; j < w; ++j) {
            output << hostCopy[i * w + j] << " ";
        }
        output << "\n";
    }
    output << "\n";
}

}  // namespace debug

#endif