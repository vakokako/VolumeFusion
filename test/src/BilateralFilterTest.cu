#include <gtest/gtest.h>

#include <af/BilateralFilter.cuh>
#include <BilateralFilterImplementations.cu>

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void bilateralFilter(const Mat & input, Mat & output, int r,double sI, double sS);
void bilateralFilterOpenCV(const Mat & input, Mat & output, int r,double sI, double sS);
void bilateralFilter8(const Mat & input, Mat & output, int r,double sI, double sS);
void bilateralFilter16(const Mat & input, Mat & output, int r,double sI, double sS);
void bilateralFilter32(const Mat & input, Mat & output, int r,double sI, double sS);
void bilateralFilterTexture(const Mat & input, Mat & dstMat, int r, double sI, double sS);
void bilateralFilterTextureOpm(const Mat & input, Mat & dstMat, int r, double sI, double sS);
void bilateralFilterTextureOpmShared(const Mat & input, Mat & dstMat, int r, double sI, double sS, float thr);

TEST(BilateralFilterTest, SpeedCompare) {

	int width = 768, height = 768;
	unsigned short* pData = (unsigned short*)malloc(width * height * sizeof(unsigned short));
	//Mat src(cvSize(width, height), CV_16UC1, Scalar::all(0));
	Mat src(cvSize(width, height), CV_16UC1, pData);
    randu(src, Scalar(0), Scalar(100));

    printf("src[100][100] = %d\n", *((unsigned short*)(src.data)+100*src.cols+100));
    Mat dst(cvSize(width, height), CV_16UC1, Scalar(0));

    Mat fsrc(cvSize(width, height), CV_32FC1);
    Mat fdst(cvSize(width, height), CV_32FC1);
    src.convertTo(dst, CV_16UC1);
    src.convertTo(fsrc, CV_32FC1);
    src.convertTo(fdst, CV_32FC1);

    float *input_d = NULL;
    float *output_d = NULL;
    cudaMalloc(&input_d, width * height * sizeof(float));
    cudaMalloc(&output_d, width * height * sizeof(float));
    cudaMemcpy(input_d, fsrc.ptr(), fsrc.total() * sizeof(float), cudaMemcpyHostToDevice);

    int r = 4;
    float sI = 66.0, sS = 1;
    char szName[256];
    printf("================================\n");
    bilateralFilterOpenCV(fsrc, fdst, r, sI, sS);
    sprintf(szName, "sI_%.1f_sS_%.1f_filterCV.raw", sI, sS);
    fdst.convertTo(dst, CV_16UC1);
    printf("================================\n");
    // bilateralFilter8(src, dst, r, sI, sS);
    sprintf(szName, "sI_%.1f_sS_%.1f_filter8.raw", sI, sS);
    printf("================================\n");
    // bilateralFilter16(src, dst, r, sI, sS);
    sprintf(szName, "sI_%.1f_sS_%.1f_filter16.raw", sI, sS);
    printf("================================\n");
    // bilateralFilter32(fsrc, fdst, r, sI, sS);
    fdst.convertTo(dst, CV_16UC1);
    sprintf(szName, "sI_%.1f_sS_%.1f_filter32.raw", sI, sS);

    printf("================================\n");
    // bilateralFilterTexture(fsrc, fdst, r, sI, sS);
    fdst.convertTo(dst, CV_16UC1);
    sprintf(szName, "sI_%.1f_sS_%.1f_filterTex.raw", sI, sS);

    printf("================================\n");
    // bilateralFilterTextureOpm(fsrc, fdst, r, sI, sS);
    fdst.convertTo(dst, CV_16UC1);
    sprintf(szName, "sI_%.1f_sS_%.1f_filterOpm.raw", sI, sS);

    printf("================================\n");
    // bilateralFilterOpm(fsrc, fdst, r, sI, sS);
    fdst.convertTo(dst, CV_16UC1);
    sprintf(szName, "sI_%.1f_sS_%.1f_filterOpmGlobal.raw", sI, sS);

    printf("================================\n");
    clock_t startTime = clock();
    af::bilateralFilterOpmSafe(output_d, input_d, height, width, r, sI, sS);
    clock_t endTime = clock();
    float time = endTime - startTime;
    cudaMemcpy(fdst.ptr(), output_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Time for the bilateralFilterOpmSafe: %f ms\n", time * 1000 / CLOCKS_PER_SEC);
    printf("output[100][100] = %.1f\n", *(fdst.ptr<float>(100) + 100));
    printf("output[50][50] = %.1f\n", *(fdst.ptr<float>(50) + 50));
    fdst.convertTo(dst, CV_16UC1);
    sprintf(szName, "sI_%.1f_sS_%.1f_filterOpmGlobalSafe.raw", sI, sS);

    printf("================================\n");
    // bilateralFilterTextureOpmSharedBrocken(fsrc, fdst, r, sI, sS);
    fdst.convertTo(dst, CV_16UC1);
    sprintf(szName, "sI_%.1f_sS_%.1f_filterOpmSrdBrocken.raw", sI, sS);

    printf("================================\n");
    startTime = clock();
    af::bilateralFilterTextureOpmShared(output_d, input_d, height, width, r, sI, sS, 100);
    endTime = clock();
    time = endTime - startTime;
    cudaMemcpy(fdst.ptr(), output_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Time for the bilateralFilterTextureOpmShared: %f ms\n", time * 1000 / CLOCKS_PER_SEC);
    printf("output[100][100] = %.1f\n", *(fdst.ptr<float>(100) + 100));
    printf("output[50][50] = %.1f\n", *(fdst.ptr<float>(50) + 50));
    fdst.convertTo(dst, CV_16UC1);
    sprintf(szName, "sI_%.1f_sS_%.1f_filterOpmSrd.raw", sI, sS);

    printf("================================\n");
    // bilateralFilterGlobalOpm(fsrc, fdst, r, sI, sS);
    fdst.convertTo(dst, CV_16UC1);
    sprintf(szName, "sI_%.1f_sS_%.1f_filterGlobalOpm.raw", sI, sS);
	free(pData);
	pData = NULL;

    cudaFree(input_d);
    cudaFree(output_d);
}