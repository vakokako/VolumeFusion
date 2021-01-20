// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_HELPER_CUH
#define TUM_HELPER_CUH

#include "af/eigen_extension.h"
#include <cuda_runtime_api.h>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "af/TSDFVolume.h"
#include <iostream>

bool assure(bool expression, int line, std::string file);

// CUDA utility functions

// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);

#define SAY(x) std::cout<<x<<std::endl;

// compute grid size from block size
inline dim3 computeGrid1D(const dim3 &block, const int w)
{
    return dim3((w - 1) / block.x + 1, 1, 1);
}

inline dim3 computeGrid2D(const dim3 &block, const int w, const int h)
{
    return dim3((w - 1) / block.x + 1, (h - 1) / block.y + 1, 1);
}

inline dim3 computeGrid3D(const dim3 &block, const int w, const int h, const int s)
{
    return dim3((w - 1) / block.x + 1, (h - 1) / block.y + 1, (s - 1) / block.z + 1);
}

Mat6f computeA(const Vec6f &gradient);   //phi is phi_current from the paper
Vec6f computeb(const Vec6f &gradient, const Vec6f &ksi, float phi_cur, float phi_ref);

//tsdf is the general, unmodified, global tsdf; ksi is the pose of the current frame
Mat4f findPose(Mat4f &resultPose, const TSDFVolume &tsdf_reference, TSDFVolume &tsdf_new_frame, const float threshold, const float step_size);
void construct_sdf(TSDFVolume &tsdf_cur, TSDFVolume &tsdf_ref, Mat4f gt);


Mat4f normalizeProjection(Mat4f proj);

void computeGradient(const TSDFVolume &tsdf, const Mat4f &ksiProjection, Vec3f *gradientByX, Vec6f *gradient);

bool _isRotation(Mat3f& rotation);

// OpenCV image conversion
// interleaved to layered
void convertMatToLayered(float *aOut, const cv::Mat &mIn);

// layered to interleaved
void convertLayeredToMat(cv::Mat &mOut, const float *aIn);


// OpenCV GUI functions
// open camera
bool openCamera(cv::VideoCapture &camera, int device, int w = 640, int h = 480);

// show image
void showImage(std::string title, const cv::Mat &mat, int x, int y);

// show histogram
void showHistogram256(const char *windowTitle, int *histogram, int windowX, int windowY);


// adding Gaussian noise
void addNoise(cv::Mat &m, float sigma);

// convert ksi vector representation of a transformation matrix back
void ksiToTransformationMatrix(const Vec6f &ksi, Mat4f &transform);

// convert a projection matrix to ksi vector representation
void projectionMatrixToKsi(const Mat4f &proj, Vec6f &ksi);

// measuring time
class Timer
{
public:
	Timer() : tStart(0), running(false), sec(0.f)
	{
	}
	void start()
	{
        cudaDeviceSynchronize();
		tStart = clock();
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
        cudaDeviceSynchronize();
		clock_t tEnd = clock();
		sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
private:
	clock_t tStart;
	bool running;
	float sec;
};

#endif
