#include <af/VertexManipulation.h>

#include <stdexcept>

namespace af {

void filterDepth(cv::Mat& depth, float threshold) {
    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            float& val = depth.at<float>(y, x);
            if (val > threshold || std::isnan(val)) {
                val = 0.0f;
            }
            // else if (val > 0.f) {
            //     val = 65.f;
            // }
        }
    }
}

void depthToVertexMap(const Mat3f& K, const cv::Mat& depth, cv::Mat& vertexMap) {
    if (depth.type() != CV_32FC1 || depth.empty())
        throw std::runtime_error("af::depthToVertexMap(): depth.type() != CV_32FC1 || depth.empty().");

    int w          = depth.cols;
    int h          = depth.rows;
    vertexMap      = cv::Mat::zeros(h, w, CV_32FC3);
    float fx       = K(0, 0);
    float fy       = K(1, 1);
    float cx       = K(0, 2);
    float cy       = K(1, 2);
    float fxInv    = 1.0f / fx;
    float fyInv    = 1.0f / fy;
    float* ptrVert = (float*)vertexMap.data;

    const float* ptrDepth = (const float*)depth.data;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float depthMeter = ptrDepth[y * w + x];
            if (depthMeter == 0.0f) {
                continue;
            }
            float x0 = (float(x) - cx) * fxInv;
            float y0 = (float(y) - cy) * fyInv;

            size_t off       = (y * w + x) * 3;
            ptrVert[off]     = x0 * depthMeter;
            ptrVert[off + 1] = y0 * depthMeter;
            ptrVert[off + 2] = depthMeter;
        }
    }
}

void normals(const cv::Mat& vertexMap, cv::Mat& normals) {
    if (vertexMap.type() != CV_32FC3 || vertexMap.empty())
        throw std::runtime_error("af::normals(): vertexMap.type() != CV_32FC3 || vertexMap.empty()");

    int h             = vertexMap.rows;
    int w             = vertexMap.cols;
    normals           = cv::Mat::zeros(h, w, CV_32FC3);
    Vec3f* ptrNormals = (Vec3f*)normals.data;

    const Vec3f* ptrVertex = (const Vec3f*)vertexMap.data;
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            int i = y * w + x;
            if (ptrVertex[i][2] <= 0.0f) {
                continue;
            }
            // clang-format off
            int iXl =  y      * w + (x - 1);
            int iXr =  y      * w + (x + 1);
            int iYt = (y - 1) * w +  x;
            int iYb = (y + 1) * w +  x;
            // clang-format on
            Vec3f ptrVertexiXr = ptrVertex[iXr];
            Vec3f ptrVertexiXl = ptrVertex[iXl];
            Vec3f ptrVertexiYb = ptrVertex[iYb];
            Vec3f ptrVertexiYt = ptrVertex[iYt];
            if (ptrVertexiXr[2] <= 0.0f || ptrVertexiXl[2] <= 0.0f || ptrVertexiYb[2] <= 0.0f || ptrVertexiYt[2] <= 0.0f)
                continue;
            Vec3f diffX = ptrVertexiXr - ptrVertexiXl;
            Vec3f diffY = ptrVertexiYb - ptrVertexiYt;
            Vec3f gX    = (diffX) / ((diffX).head(2).norm());
            Vec3f gY    = (diffY) / ((diffY).head(2).norm());
            Vec3f norm  = gY.cross(gX);
            norm.normalize();
            ptrNormals[i] = norm;
        }
    }
}

Vec3f centroid(const cv::Mat& vertexMap) {
    Vec3f centroid(0.0, 0.0, 0.0);

    size_t cnt = 0;
    for (int y = 0; y < vertexMap.rows; ++y) {
        for (int x = 0; x < vertexMap.cols; ++x) {
            cv::Vec3f pt = vertexMap.at<cv::Vec3f>(y, x);
            if (pt.val[2] > 0.0) {
                Vec3f pt3(pt.val[0], pt.val[1], pt.val[2]);
                centroid += pt3;
                ++cnt;
            }
        }
    }
    centroid /= float(cnt);

    return centroid;
}

Vec3f centroid(const std::vector<Vec3f>& vertexMap) {
    Vec3f centroid(0.0f, 0.0f, 0.0f);

    size_t cnt = 0;
    for (std::size_t i = 0; i < vertexMap.size(); ++i) {
        if (vertexMap[i][2] > 0.0) {
            centroid += vertexMap[i];
            ++cnt;
        }
    }
    centroid /= float(cnt);

    return centroid;
}

Vec3f centroidDBG(const cv::Mat& vertexMap, size_t& cntDebug, Vec3f& centrDebug) {
    Vec3f centroid(0.0, 0.0, 0.0);

    size_t cnt = 0;
    for (int y = 0; y < vertexMap.rows; ++y) {
        for (int x = 0; x < vertexMap.cols; ++x) {
            cv::Vec3f pt = vertexMap.at<cv::Vec3f>(y, x);
            if (pt.val[2] > 0.0) {
                Vec3f pt3(pt.val[0], pt.val[1], pt.val[2]);
                centroid += pt3;
                ++cnt;
            }
        }
    }
    centrDebug = centroid;
    cntDebug   = cnt;

    centroid /= float(cnt);

    return centroid;
}

}  // namespace af