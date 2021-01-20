// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "af/helper.cuh"
#include "af/gradient.cuh"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <limits>
#include <cassert>
#include <sophus/se3.hpp>
#include "af/cublas_v2.h"
#define _VNAME(x) #x
#define Print(x) std::cout<<_VNAME(x)<<":\n"<<x<<std::endl;



#define ensure(x) assure2(x, __LINE__, __FILE__)

bool assure2(bool expression, int line, std::string file)
{
    if(!expression)
    {
        std::cout<<" assert false: in line: "<<line<<" file: "<<file<<std::endl;
        exit(1);
    }

    return 1;
}


// cuda error checking
std::string prev_file = "";
int prev_line = 0;
void cuda_check(std::string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        std::cout << std::endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << std::endl;
        if (prev_line > 0)
            std::cout << "Previous CUDA call:" << std::endl << prev_file << ", line " << prev_line << std::endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}


// OpenCV image conversion: layered to interleaved
void convertLayeredToInterleaved(float *aOut, const float *aIn, int w, int h, int nc)
{
    if (!aOut || !aIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    if (nc == 1)
    {
        memcpy(aOut, aIn, w*h*sizeof(float));
        return;
    }

    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[(nc-1-c) + nc*(x + (size_t)w*y)] = aIn[x + (size_t)w*y + nOmega*c];
            }
        }
    }
}
void convertLayeredToMat(cv::Mat &mOut, const float *aIn)
{
    convertLayeredToInterleaved((float*)mOut.data, aIn, mOut.cols, mOut.rows, mOut.channels());
}


// OpenCV image conversion: interleaved to layered
void convertInterleavedToLayered(float *aOut, const float *aIn, int w, int h, int nc)
{
    if (!aOut || !aIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    if (nc == 1)
    {
        memcpy(aOut, aIn, w*h*sizeof(float));
        return;
    }

    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[x + (size_t)w*y + nOmega*c] = aIn[(nc-1-c) + nc*(x + (size_t)w*y)];
            }
        }
    }
}

void convertMatToLayered(float *aOut, const cv::Mat &mIn)
{
    convertInterleavedToLayered(aOut, (float*)mIn.data, mIn.cols, mIn.rows, mIn.channels());
}

// show cv:Mat in OpenCV GUI
// open camera using OpenCV
bool openCamera(cv::VideoCapture &camera, int device, int w, int h)
{
    if(!camera.open(device))
    {
        return false;
    }
    camera.set(CV_CAP_PROP_FRAME_WIDTH, w);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, h);
    return true;
}

// show cv:Mat in OpenCV GUI
void showImage(std::string title, const cv::Mat &mat, int x, int y)
{
    const char *wTitle = title.c_str();
    cv::namedWindow(wTitle, CV_WINDOW_AUTOSIZE);
    cv::moveWindow(wTitle, x, y);
    cv::imshow(wTitle, mat);
}

// show histogram in OpenCV GUI
void showHistogram256(const char *windowTitle, int *histogram, int windowX, int windowY)
{
    if (!histogram)
    {
        std::cerr << "histogram not allocated!" << std::endl;
        return;
    }

    const int nbins = 256;
    cv::Mat canvas = cv::Mat::ones(125, 512, CV_8UC3);

    float hmax = 0;
    for(int i = 0; i < nbins; ++i)
        hmax = max((int)hmax, histogram[i]);

    for (int j = 0, rows = canvas.rows; j < nbins-1; j++)
    {
        for(int i = 0; i < 2; ++i)
            cv::line(
                        canvas,
                        cv::Point(j*2+i, rows),
                        cv::Point(j*2+i, rows - (histogram[j] * 125.0f) / hmax),
                        cv::Scalar(255,128,0),
                        1, 8, 0
                        );
    }

    showImage(windowTitle, canvas, windowX, windowY);
}


// add Gaussian noise
float noise(float sigma)
{
    float x1 = (float)rand()/RAND_MAX;
    float x2 = (float)rand()/RAND_MAX;
    return sigma * sqrtf(-2*log(std::max(x1,0.000001f)))*cosf(2*M_PI*x2);
}

void addNoise(cv::Mat &m, float sigma)
{
    float *data = (float*)m.data;
    int w = m.cols;
    int h = m.rows;
    int nc = m.channels();
    size_t n = (size_t)w*h*nc;
    for(size_t i=0; i<n; i++)
    {
        data[i] += noise(sigma);
    }
}

void skewSymmetric_temp(const Vec3f &vec, Mat3f &matX)
{
    matX << 0, -vec(2), vec(1),
          vec(2), 0, -vec(0),
          -vec(1), vec(0), 0;
}

Mat4f normalizeProjection(Mat4f proj)
{
    return (proj / proj(3,3));
}

bool _isRotation(Mat3f& rotation)
{
    Vec3f v1 = rotation.block(0,0,3,1);
    Vec3f v2 = rotation.block(0,1,3,2);
    Vec3f v3 = rotation.block(0,2,3,3);

    if(abs(v1.norm() - 1) > 0.01 || abs(v2.norm() - 1) > 0.01 || abs(v2.norm() - 1) > 0.01 )
        return false;

    float value1 = v1.transpose()*v2;
    float value2 = v1.transpose()*v3;
    float value3 = v2.transpose()*v3;

    if(abs(value1) > 0.01 || abs(value2) > 0.01 || abs(value3) > 0.01)
        return false;

    return true;
}

void ksiToTransformationMatrix(const Vec6f &ksi, Mat4f &transform)
{
    transform = Sophus::SE3<float>::exp(ksi).matrix();
}

void projectionMatrixToKsi(const Mat4f &proj, Vec6f &ksi)
{
    Sophus::SE3<float> pose(proj);
    ksi = Sophus::SE3<float>::log(pose);
}

void normalize_temp(Vec3f& point){
    float norm = sqrt( (point[0])*(point[0]) + (point[1])*(point[1]) + (point[2])*(point[2]) );

    if(norm != 0){
        point[0] /= norm;
        point[1] /= norm;
        point[2] /= norm;
    }
}

void computeGradientByX(const TSDFVolume &tsdf, Vec3f *gradientByX)
{
    Vec3i dim = tsdf.dimensions();
    float *sdfs = tsdf.ptrTsdf();
    Vec3f dX = tsdf.voxelSize() * 2;
    for (size_t z = 0; z < dim[2]; ++z)
    {
        for (size_t y = 0; y < dim[1]; ++y)
        {
            for (size_t x = 0; x < dim[0]; ++x)
            {
                size_t off = z*dim[0]*dim[1] + y*dim[0] + x;
                size_t offX2 = z*dim[0]*dim[1] + y*dim[0] + x + ((x == dim[0] - 1) ? 0 : 1);
                size_t offX1 = z*dim[0]*dim[1] + y*dim[0] + x - ((x == 0) ? 0 : 1);
                size_t offY2 = z*dim[0]*dim[1] + (y + ((y == dim[1] - 1) ? 0 : 1))*dim[0] + x;
                size_t offY1 = z*dim[0]*dim[1] + (y - ((y == 0) ? 0 : 1))*dim[0] + x;
                size_t offZ2 = (z + ((z == dim[2] - 1) ? 0 : 1))*dim[0]*dim[1] + y*dim[0] + x;
                size_t offZ1 = (z - ((z == 0) ? 0 : 1))*dim[0]*dim[1] + y*dim[0] + x;

                gradientByX[off][0] = (sdfs[offX2] - sdfs[offX1]) / dX[0];
                gradientByX[off][1] = (sdfs[offY2] - sdfs[offY1]) / dX[1];
                gradientByX[off][2] = (sdfs[offZ2] - sdfs[offZ1]) / dX[2];

                normalize_temp(gradientByX[off]);

            }
        }
    }
}

Eigen::Matrix<float, 6, 6> computeA(const Vec6f& gradient)   //phi is phi_current from the paper
{
    return gradient * gradient.transpose();
}

Vec6f computeb(const Vec6f &gradient, const Vec6f &ksi, float phi_cur, float phi_ref)
{
    float scalar = (gradient.transpose() * ksi)(0,0) + phi_ref - phi_cur;
    return gradient * scalar;
}

Vec3f transform_temp(Vec3f point, Mat4f projection)
{
    return (projection * point.homogeneous()).head(3);
}


// TODO
Vec6f computeStepParameters(const TSDFVolume &tsdf_cur, const TSDFVolume &tsdf_ref, Vec6f &ksi_cur, Vec6f* gradient, Mat4f& poseVolume)
{
    //compute result (update step)
    Eigen::MatrixXf result(1,6);

    //volume dimensions
    Vec3i dim = tsdf_cur.dimensions();

    //pointers to sdf current
    float* wsdfs_cur = tsdf_cur.ptrTsdfWeights();
    float* sdfs_cur = tsdf_cur.ptrTsdf();

    //pointers to sdf reference
    float* wsdfs_ref = tsdf_ref.ptrTsdfWeights();
    float* sdfs_ref = tsdf_ref.ptrTsdf();


    //declare intermediate results
    Vec6f b_m = Vec6f::Zero();
    Mat6f A_m = Mat6f::Zero();

    //pass pose for gradient computation and compute gradient
    Mat4f projection_current;
    ksiToTransformationMatrix(ksi_cur, projection_current);
    Mat3f R = projection_current.block(0,0,3,3);

    Mat4f gt;
    gt << 0.998630, 0.015166, -0.050090, 0.025045,
          -0.015701, 0.999824, -0.010304, 0.005152,
          0.049925, 0.011076, 0.998692, 0.000655,
          0.000000, 0.000000, 0.000000, 1.000000;

    gt = gt.inverse();

    // Print(gt);

    //gt = Mat4f::Identity();
    //projection_current = gt;

    float error_function = 0.0;
    float total_weights = 0.0;
    float avg_error = 0.0;

    for (size_t z = 0; z < dim[2]; ++z)
    {
        for (size_t y = 0; y < dim[1]; ++y)
        {
            for (size_t x = 0; x < dim[0]; ++x)
            {
                //compute global offset
                size_t off = z*dim[0]*dim[1] + y*dim[0] + x;

                //reference frame
                Vec3f world_point_reference = tsdf_ref.voxelToCamera(Vec3i(x,y,z));
                float phi_ref = sdfs_ref[off];
                float weight_ref = wsdfs_ref[off];

                //current frame : rotate and translate
                Vec3f world_point_current = transform_temp(world_point_reference, projection_current);
                Vec3i voxel_point_current = tsdf_cur.cameraToVoxel(world_point_current);
                Vec3f object_point_current = tsdf_cur.cameraToObject(world_point_current);


                if (voxel_point_current[0] < 0 || voxel_point_current[0] > dim[0] - 1 ||
                        voxel_point_current[1] < 0 || voxel_point_current[1] > dim[1] - 1 ||
                        voxel_point_current[2] < 0 || voxel_point_current[2] > dim[2] - 1)
                    continue;


                //current frame : sdf and weight
                size_t off_current = voxel_point_current[2]*dim[0]*dim[1] + voxel_point_current[1]*dim[0] + voxel_point_current[0];
                float phi_current = tsdf_cur.interpolate3voxel(voxel_point_current[0], voxel_point_current[1], voxel_point_current[2]);
                float weight_current = wsdfs_cur[off_current];

                //computation
                A_m += weight_ref * weight_current * computeA(gradient[off_current]);
                b_m += weight_ref * weight_current * computeb(gradient[off_current], ksi_cur, phi_current, phi_ref);

                //error function
                error_function += 0.5 * weight_ref*weight_current * (phi_current - phi_ref) * (phi_current - phi_ref);
                total_weights += weight_ref * weight_current;
            }
        }
    }

    avg_error = std::sqrt(error_function / total_weights);

    //Print(A_m);
    //Print(b_m);
    //Print(error_function);
    ensure(A_m.determinant() != 0);
    return (A_m.inverse() * b_m);
}

void construct_sdf(TSDFVolume &tsdf_cur, TSDFVolume &tsdf_ref, Mat4f gt)
{
    //volume dimensions
    Vec3i dim = tsdf_cur.dimensions();

    float* wsdfs_cur = tsdf_cur.ptrTsdfWeights();
    float* sdfs_cur = tsdf_cur.ptrTsdf();

    float* wsdfs_ref = tsdf_ref.ptrTsdfWeights();
    float* sdfs_ref = tsdf_ref.ptrTsdf();

    Vec3f centroid = tsdf_cur.centroid();
    Vec3f voxelSize = tsdf_cur.voxelSize();
    Vec3f size = tsdf_cur.size();

    float* wsdfs_cur_d = NULL;
    float* sdfs_cur_d = NULL;
    float* wsdfs_ref_d = NULL;
    float* sdfs_ref_d = NULL;

    size_t sizeAll = tsdf_cur.gridSize();
    cudaDeviceSynchronize();
    cudaMalloc(&wsdfs_cur_d, sizeof(float) * sizeAll); CUDA_CHECK;
    cudaMalloc(&sdfs_cur_d, sizeof(float) * sizeAll); CUDA_CHECK;
    cudaMalloc(&wsdfs_ref_d, sizeof(float) * sizeAll); CUDA_CHECK;
    cudaMalloc(&sdfs_ref_d, sizeof(float) * sizeAll); CUDA_CHECK;

    cudaMemcpy(wsdfs_cur_d, wsdfs_cur, sizeof(float) * sizeAll, cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(sdfs_cur_d, sdfs_cur, sizeof(float) * sizeAll, cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(wsdfs_ref_d, wsdfs_ref, sizeof(float) * sizeAll, cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(sdfs_ref_d, sdfs_ref, sizeof(float) * sizeAll, cudaMemcpyHostToDevice); CUDA_CHECK;

    constructSDFCuda(sdfs_ref_d, wsdfs_ref_d, sdfs_cur_d, wsdfs_cur_d, gt, centroid, voxelSize, size, dim);

    cudaMemcpy(sdfs_ref, sdfs_ref_d, sizeof(float) * sizeAll, cudaMemcpyDeviceToHost);
    cudaMemcpy(wsdfs_ref, wsdfs_ref_d, sizeof(float) * sizeAll, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    cudaFree(wsdfs_cur_d);
    cudaFree(sdfs_cur_d);
    cudaFree(wsdfs_ref_d);
    cudaFree(sdfs_ref_d);

}

Mat4f findPose(Mat4f &poseVolume, const TSDFVolume &tsdf_reference, TSDFVolume &tsdf_new_frame, const float threshold, const float step_size)    //tsdf is the general, unmodified, global tsdf; ksi is the pose of the current frame
{
    bool cpu = false;

    Mat4f reference_pose = Mat4f::Identity();
    Mat4f current_pose = reference_pose;

    //define twist coordinates poses
    Vec6f ksi_reference, ksi_update, ksi_previous_step;

    //initialize twist coordinates poses : our initial guess is the identity
    projectionMatrixToKsi(reference_pose, ksi_reference);
    ksi_update = ksi_reference;
    ksi_previous_step = ksi_update;

    // init cuBLAS
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initialization failed\n";
        cpu = true;
    }

    // Extra variables
    Timer time;
    float error = std::numeric_limits<float>::max(); // initialize error to maximum float
    size_t sizeAll = tsdf_new_frame.gridSize();

    // Allocate arrays on CPU
    Vec3f *gradientByX = new Vec3f[tsdf_new_frame.gridSize()];
    Vec3f *gradientByXCuda = new Vec3f[sizeAll];
    Vec6f *gradientCuda = new Vec6f[sizeAll];
    Vec6f* gradient = new Vec6f[tsdf_reference.gridSize()];
    Eigen::Matrix<float, 6, 6> A = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
    float error_function = 0.0;
    Vec6f update;

    // Split parameters calculation
    int splitCount = 2;
    Vec3i completeDim = tsdf_new_frame.dimensions();
    Vec3i splitDim = Vec3i(completeDim[0], completeDim[1], completeDim[2] / splitCount);
    size_t sizeSplit = splitDim[0] * splitDim[1] * splitDim[2];

    // Allocate arrays on GPU
    Vec3f *gradientByX_d = NULL;
    float *tsdf_d_new_frame = NULL;
    float *tsdf_d_reference = NULL;
    float *wsdf_d_new_frame = NULL;
    float *wsdf_d_reference = NULL;
    Vec6f *gradient_d = NULL;
    float *A_d = NULL;
    float *b_d = NULL;
    float *error_d_function = NULL;
    //std::cout << "SIZEOF Vec3f = " << sizeof(Vec3f) << std::endl;
    //std::cout << "SIZEOF Vec6f = " << sizeof(Vec6f) << std::endl;
    //std::cout << "SIZEOF float = " << sizeof(float) << std::endl;
    //std::cout << "sizeAll = " << sizeAll << std::endl;
    //double total_mem = sizeof(Vec3f) * sizeAll + sizeof(Vec6f) * sizeAll + sizeof(float) * sizeAll * 6 + sizeof(float) * sizeAll * 6 * 6 + sizeof(float) * sizeAll * 6;
    //std::cout << "**** allocating " << total_mem / 1024 / 1024  << " MB ****" << std::endl;

    cudaMalloc(&gradientByX_d, sizeof(Vec3f) * sizeAll); CUDA_CHECK;
    cudaMalloc(&tsdf_d_new_frame, sizeof(float) * sizeAll); CUDA_CHECK;
    cudaMalloc(&tsdf_d_reference, sizeof(float) * sizeAll); CUDA_CHECK;
    cudaMalloc(&wsdf_d_new_frame, sizeof(float) * sizeAll); CUDA_CHECK;
    cudaMalloc(&wsdf_d_reference, sizeof(float) * sizeAll); CUDA_CHECK;
    cudaMalloc(&gradient_d, sizeof(Vec6f) * sizeAll); CUDA_CHECK;
//    cudaMalloc(&A_d, sizeof(float) * sizeAll * 6 * 6); CUDA_CHECK;
//    cudaMalloc(&b_d, sizeof(float) * sizeAll * 6); CUDA_CHECK;
//    cudaMalloc(&error_d_function, sizeof(float) * sizeAll); CUDA_CHECK;
    cudaMalloc(&A_d, sizeof(float) * sizeSplit * 6 * 6); CUDA_CHECK;
    cudaMalloc(&b_d, sizeof(float) * sizeSplit * 6); CUDA_CHECK;
    cudaMalloc(&error_d_function, sizeof(float) * sizeSplit); CUDA_CHECK;

    float *one_d_vec = NULL;
    size_t nbytes_result = (size_t) sizeSplit * sizeof(float);
    cudaMalloc(&one_d_vec, nbytes_result);
    initCuda(one_d_vec, sizeSplit);

    // CPU => GPU: TSDFs
    cudaMemcpy(tsdf_d_new_frame, tsdf_new_frame.ptrTsdf(), sizeof(float) * sizeAll, cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(tsdf_d_reference, tsdf_reference.ptrTsdf(), sizeof(float) * sizeAll, cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(wsdf_d_new_frame, tsdf_new_frame.ptrTsdfWeights(), sizeof(float) * sizeAll, cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(wsdf_d_reference, tsdf_reference.ptrTsdfWeights(), sizeof(float) * sizeAll, cudaMemcpyHostToDevice); CUDA_CHECK;

    /**
     * 0. Compute gradientByX
     **/
    std::cout << "> compute gradientByX" << std::endl;
    if (cpu) {
        time.start();
        computeGradientByX(tsdf_new_frame, gradientByX);
        std::cout << "cpu time:" << time.get() << std::endl;
    }
    time.start();
    computeGradientByXCuda(gradientByX_d, tsdf_d_new_frame, tsdf_new_frame.voxelSize(), tsdf_new_frame.dimensions());
    std::cout << "gpu time:" << time.get() << std::endl;
    // GPU => CPU: gradientByX
    cudaMemcpy(gradientByXCuda, gradientByX_d, sizeof(Vec3f) * sizeAll, cudaMemcpyDeviceToHost);

    /**
     * 1. Optimize
     **/
    while(error > threshold)
    {
        // Matrix pose
        ksiToTransformationMatrix(ksi_update, current_pose);

        /**
         * 1.1. Compute the gradient
         **/
        std::cout << "> compute the gradient" << std::endl;
        if (cpu) {
            time.start();
            computeGradient(tsdf_new_frame, current_pose, gradientByX, gradient);
            std::cout << "cpu time:" << time.get() << std::endl;
        }
        time.start();
        computeGradientCuda(gradient_d, gradientByX_d, current_pose, tsdf_new_frame.centroid(), tsdf_new_frame.voxelSize(), tsdf_new_frame.size(), tsdf_new_frame.dimensions());
        std::cout << "gpu time:" << time.get() << std::endl;
        // GPU => CPU: gradient
        cudaMemcpy(gradientCuda, gradient_d, sizeof(Vec6f) * sizeAll, cudaMemcpyDeviceToHost);

        /**
         * 1.2. Compute update step
         **/
        std::cout << "> compute update state" << std::endl;
        ksi_previous_step = ksi_update;
        if (cpu) {
            time.start();
            update = computeStepParameters(tsdf_new_frame, tsdf_reference, ksi_update, gradient, poseVolume);
            std::cout << "cpu time:" << time.get() << std::endl;
        } else {
            time.start();
            Mat4f projection_current;
            ksiToTransformationMatrix(ksi_update, projection_current);

            A = Mat6f::Zero();
            b = Vec6f::Zero();
            error_function = 0;
            for (int iSplit = 0; iSplit < splitCount; ++iSplit) {
                int offsetZ = iSplit * splitDim[2];

                computeStepParametersCuda(
                    tsdf_d_new_frame,
                    wsdf_d_new_frame,
                    tsdf_d_reference,
                    wsdf_d_reference,
                    projection_current,
                    gradient_d,
                    tsdf_new_frame.centroid(),
                    tsdf_new_frame.voxelSize(),
                    tsdf_new_frame.size(),
                    splitDim,
                    tsdf_new_frame.dimensions(),
                    offsetZ,
                    ksi_update,
                    // results:
                    A_d,
                    b_d,
                    error_d_function
                );
                float splitSumA = 0;
                float splitSumb = 0;
                float splitSumError = 0;

                cublasSasum(handle, sizeSplit, error_d_function, 1, &splitSumError);
                std::cout << "split i : " << iSplit << " splitSumError: " << splitSumError << std::endl;
                error_function += splitSumError;
                for (int i = 0; i < 6; i++) {
                    cublasSdot(handle, sizeSplit, &b_d[sizeSplit * i], 1, one_d_vec, 1, &splitSumb);
                    b[i] += splitSumb;
                }
                for (int i = 0; i < 6 * 6; i++) {
                    cublasSdot(handle, sizeSplit, &A_d[sizeSplit * i], 1, one_d_vec, 1, &splitSumA);
                    A(i / 6, i % 6) += splitSumA;
                }


            }

//            Print(A);
//            Print(b);
            Print(error_function);
//            Print(total_weights);
            std::cout << "gpu time:" << time.get() << std::endl;

            ensure(A.determinant() != 0);
            update = (A.inverse() * b);
        }

        /**
         * 1.3. update current guess
         **/
        Vec6f diff = update - ksi_update;
        ksi_update = ksi_update + step_size * diff;

        Mat4f translational_update_matrix_current, translational_update_matrix_previous;
        ksiToTransformationMatrix(ksi_update, translational_update_matrix_current);
        ksiToTransformationMatrix(ksi_previous_step, translational_update_matrix_previous);

        Vec3f translational_update_current = translational_update_matrix_current.block(0,3,3,3);
        Vec3f translational_update_previous = translational_update_matrix_previous.block(0,3,3,3);

        error = (translational_update_current - translational_update_previous).norm();

        ksiToTransformationMatrix(ksi_update, current_pose);
    }

    // free memory
    cudaFree(tsdf_d_new_frame); CUDA_CHECK;
    cudaFree(wsdf_d_new_frame); CUDA_CHECK;
    cudaFree(tsdf_d_reference); CUDA_CHECK;
    cudaFree(wsdf_d_reference); CUDA_CHECK;
    cudaFree(gradientByX_d); CUDA_CHECK;
    cudaFree(gradient_d); CUDA_CHECK;
    cudaFree(one_d_vec); CUDA_CHECK;
    cudaFree(A_d); CUDA_CHECK;
    cudaFree(b_d); CUDA_CHECK;
    cudaFree(error_d_function); CUDA_CHECK;
    delete[] gradientByXCuda;
    delete[] gradientCuda;

    //cudaMalloc(&gradientByX_d, sizeof(Vec3f) * sizeAll); CUDA_CHECK;
    //cudaMalloc(&tsdf_d_new_frame, sizeof(float) * sizeAll); CUDA_CHECK;
    //cudaMalloc(&tsdf_d_reference, sizeof(float) * sizeAll); CUDA_CHECK;
    //cudaMalloc(&wsdf_d_new_frame, sizeof(float) * sizeAll); CUDA_CHECK;
    //cudaMalloc(&wsdf_d_reference, sizeof(float) * sizeAll); CUDA_CHECK;
    //cudaMalloc(&gradient_d, sizeof(Vec6f) * sizeAll); CUDA_CHECK;
    //cudaMalloc(&A_d, sizeof(float) * sizeAll * 6 * 6); CUDA_CHECK;
    //cudaMalloc(&b_d, sizeof(float) * sizeAll * 6); CUDA_CHECK;
    //cudaMalloc(&error_d_function, sizeof(float) * sizeAll); CUDA_CHECK

    // transorm the pose back to 4x4 matrix and return it
    Mat4f found_pose;
    ksiToTransformationMatrix(ksi_update, found_pose);
    return found_pose;
}

void computeGradient(const TSDFVolume &tsdf, const Mat4f &projection, Vec3f *gradientByX, Vec6f *gradient)
{
    Vec3i dim = tsdf.dimensions();

    Mat4f projectionInverse = projection.inverse();

    // calculate gradient for each voxel
    for (size_t z = 0; z < dim[2]; ++z)
    {
        for (size_t y = 0; y < dim[1]; ++y)
        {
            for (size_t x = 0; x < dim[0]; ++x)
            {
                //reference frame
                Vec3f world_point_reference = tsdf.voxelToCamera(Vec3i(x,y,z));

                //current frame : rotate and translate
                Vec3f world_point_current = transform_temp(world_point_reference, projection);

                ensure(projectionInverse(3,3) == 1);
                Vec3f ksiV = (projectionInverse * world_point_current.homogeneous()).head(3);
                Mat3f ksiVX;
                skewSymmetric_temp(ksiV, ksiVX);

                Eigen::Matrix<float, 3, 6> concatenation;
                concatenation << Mat3f::Identity(), -1 * ksiVX;

                size_t off_current = z*dim[0]*dim[1] + y*dim[0] + x;
                gradient[off_current] = gradientByX[off_current].transpose() * concatenation;
            }
        }
    }
}

