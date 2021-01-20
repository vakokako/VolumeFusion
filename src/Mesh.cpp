#include "af/Mesh.h"

#include <af/CameraModel.h>

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <fstream>
#include <iostream>

unsigned int Mesh::addVertex(const Vec3f &v, const Vec3b &c) {
    unsigned int vIdx = vertexCloud_.size();
    vertexCloud_.push_back(v);
    colors_.push_back(c);
    return vIdx;
}

void Mesh::clear() {
    vertexCloud_.clear();
    colors_.clear();
    faces_.clear();
}

bool Mesh::savePly(const std::string &filename) const {
    if (vertexCloud_.empty()) {
        std::cout << "Mesh::savePly(): Vertex cloud is empty.\n";
        return false;
    }

    std::ofstream plyFile;
    plyFile.open(filename.c_str());
    if (!plyFile.is_open()) {
        std::cout << "Mesh::savePly(): File couldn't be open.\n";
        return false;
    }

    plyFile << "ply" << std::endl;
    plyFile << "format ascii 1.0" << std::endl;
    plyFile << "element vertex " << vertexCloud_.size() << std::endl;
    plyFile << "property float x" << std::endl;
    plyFile << "property float y" << std::endl;
    plyFile << "property float z" << std::endl;
    plyFile << "property uchar red" << std::endl;
    plyFile << "property uchar green" << std::endl;
    plyFile << "property uchar blue" << std::endl;
    plyFile << "element face " << (int)faces_.size() << std::endl;
    plyFile << "property list uchar int vertex_indices" << std::endl;
    plyFile << "end_header" << std::endl;

    // write vertices
    for (size_t i = 0; i < vertexCloud_.size(); i++) {
        plyFile << vertexCloud_.vec_[i][0] << " " << vertexCloud_.vec_[i][1] << " " << vertexCloud_.vec_[i][2];
        plyFile << " " << (int)colors_[i][0] << " " << (int)colors_[i][1] << " " << (int)colors_[i][2];
        plyFile << std::endl;
    }

    // write faces
    for (size_t i = 0; i < faces_.size(); i++) {
        plyFile << "3 " << (int)faces_[i][0] << " " << (int)faces_[i][1] << " " << (int)faces_[i][2] << std::endl;
    }

    plyFile.close();

    return true;
}

bool Mesh::loadPly(const std::string &filename) {
    std::ifstream plyFile;
    plyFile.open(filename.c_str());
    if (!plyFile.is_open())
        throw(std::runtime_error("Mesh::loadPly(): File couldn't be open."));

    if (!vertexCloud_.empty()) {
        vertexCloud_.vec_.clear();
        colors_.clear();
        faces_.clear();
    }

    std::string currLine;
    std::getline(plyFile, currLine);
    std::getline(plyFile, currLine);
    plyFile >> currLine >> currLine >> currLine;
    std::size_t vectexCloudSize = std::stoi(currLine);
    std::getline(plyFile, currLine);
    std::getline(plyFile, currLine);
    std::getline(plyFile, currLine);
    std::getline(plyFile, currLine);
    std::getline(plyFile, currLine);
    std::getline(plyFile, currLine);
    std::getline(plyFile, currLine);
    plyFile >> currLine >> currLine >> currLine;
    std::size_t facesSize = std::stoi(currLine);
    std::getline(plyFile, currLine);
    std::getline(plyFile, currLine);
    std::getline(plyFile, currLine);

    // write vertices
    Vec3f pos;
    Vec3b color;
    for (size_t i = 0; i < vectexCloudSize; i++) {
        plyFile >> currLine;
        pos[0] = std::stof(currLine);
        plyFile >> currLine;
        pos[1] = std::stof(currLine);
        plyFile >> currLine;
        pos[2] = std::stof(currLine);
        plyFile >> currLine;
        color[0] = std::stoi(currLine);
        plyFile >> currLine;
        color[1] = std::stoi(currLine);
        plyFile >> currLine;
        color[2] = std::stoi(currLine);
        std::getline(plyFile, currLine);

        vertexCloud_.push_back(pos);
        colors_.push_back(color);
    }

    // write faces
    Vec3i face;
    for (size_t i = 0; i < facesSize; i++) {
        plyFile >> currLine;
        plyFile >> currLine;
        face[0] = std::stoi(currLine);
        plyFile >> currLine;
        face[1] = std::stoi(currLine);
        plyFile >> currLine;
        face[2] = std::stoi(currLine);
        std::getline(plyFile, currLine);

        faces_.push_back(face);
    }

    plyFile.close();

    return true;
}
