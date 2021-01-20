#ifndef VERTEXMANIPULATION_H
#define VERTEXMANIPULATION_H

#include <af/eigen_extension.h>
#include <opencv2/core.hpp>

namespace af {

void filterDepth(cv::Mat& depth, float threshold);

/**
 * @param K camera intrinsics matrix
 * @throw if depth empty or type is not CV_32FC1
 */
void depthToVertexMap(const Mat3f& K, const cv::Mat& depth, cv::Mat& vertexMap);

void normals(const cv::Mat& vertexMap, cv::Mat& normals);

/** @brief Calculates the center point of vertexMap */
Vec3f centroid(const cv::Mat& vertexMap);
Vec3f centroid(const std::vector<Vec3f>& vertexMap);
Vec3f centroidDBG(const cv::Mat& vertexMap, size_t& cntDebug, Vec3f& centrDebug);

}

#endif