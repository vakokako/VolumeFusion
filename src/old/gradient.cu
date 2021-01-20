// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "af/gradient.cuh"
#include "af/helper.cuh"
#include "af/cuda_runtime.h"
#include "af/stdio.h"
#define _VNAME(x) #x
#define Print(x) std::cout<<_VNAME(x)<<":\n"<<x<<std::endl;


#define ensure(x) assure2(x, __LINE__, __FILE__)

bool assure2(bool expression, int line, std::string file);

__device__
float interpolate3voxel(float x, float y, float z, const float* m_tsdf, const Vec3i& m_dim)
{
    Vec3f voxel(x, y, z);
    if (voxel[0] < 0.0f || voxel[0] > m_dim[0] - 1 ||
            voxel[1] < 0.0f || voxel[1] > m_dim[1] - 1 ||
            voxel[2] < 0.0f || voxel[2] > m_dim[2] - 1)
        return -1.0f;

    // tri-linear interpolation
    const int x0 = static_cast<int>(voxel[0]);
    const int y0 = static_cast<int>(voxel[1]);
    const int z0 = static_cast<int>(voxel[2]);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;

    if (x1 >= m_dim[0] || y1 >= m_dim[1] || z1 >= m_dim[2])
        return m_tsdf[z0 * m_dim[0] * m_dim[1] + y0 * m_dim[0] + x0];

    const float xd = voxel[0] - x0;
    const float yd = voxel[1] - y0;
    const float zd = voxel[2] - z0;

    const float sdf000 = m_tsdf[z0 * m_dim[0] * m_dim[1] + y0 * m_dim[0] + x0];
    const float sdf010 = m_tsdf[z0 * m_dim[0] * m_dim[1] + y1 * m_dim[0] + x0];
    const float sdf001 = m_tsdf[z1 * m_dim[0] * m_dim[1] + y0 * m_dim[0] + x0];
    const float sdf011 = m_tsdf[z1 * m_dim[0] * m_dim[1] + y1 * m_dim[0] + x0];
    const float sdf100 = m_tsdf[z0 * m_dim[0] * m_dim[1] + y0 * m_dim[0] + x1];
    const float sdf110 = m_tsdf[z0 * m_dim[0] * m_dim[1] + y1 * m_dim[0] + x1];
    const float sdf101 = m_tsdf[z1 * m_dim[0] * m_dim[1] + y0 * m_dim[0] + x1];
    const float sdf111 = m_tsdf[z1 * m_dim[0] * m_dim[1] + y1 * m_dim[0] + x1];

    const float c00 = sdf000 * (1.0f - xd) + sdf100 * xd;
    const float c10 = sdf010 * (1.0f - xd) + sdf110 * xd;
    const float c01 = sdf001 * (1.0f - xd) + sdf101 * xd;
    const float c11 = sdf011 * (1.0f - xd) + sdf111 * xd;

    const float c0 = c00 * (1.0f - yd) + c10 * yd;
    const float c1 = c01 * (1.0f - yd) + c11 * yd;

    return c0 * (1.0f - zd) + c1 * zd;
}

__device__
Vec3f voxelToWorld(const Vec3i &voxel, const Vec3f &voxelSize, const Vec3f &size)
{
    Vec3f pt = voxel.cast<float>().cwiseProduct(voxelSize) - size*0.5f;
    return pt;
}

__device__
Vec3i worldToVoxel(const Vec3f &pt, const Vec3f &voxelSize, const Vec3f &size)
{
    Vec3f voxelSizeInv(1.0 / voxelSize[0], 1.0 / voxelSize[1], 1.0 / voxelSize[2]);
    Vec3f voxelF = (pt + 0.5f * size).cwiseProduct(voxelSizeInv);
    Vec3i voxelIdx = voxelF.cast<int>();
    return voxelIdx;
}

__device__
Vec3f voxelToCamera(const Vec3i &pt, const Vec3f &centroid, const Vec3f &voxelSize, const Vec3f &size)
{
    Vec3f world = voxelToWorld(pt, voxelSize, size);
    return world + centroid;
}

__device__
Vec3i cameraToVoxel(const Vec3f &pt, const Vec3f &centroid, const Vec3f &voxelSize, const Vec3f &size)
{
    Vec3f world = pt - centroid;
    Vec3i _vox(worldToVoxel(world, voxelSize, size));
    return _vox;
}

__device__
Vec3f transform(Vec3f point, Mat4f projection)
{
    return (projection * point.homogeneous()).head(3);
}

__device__
void normalize(Vec3f& point){
    float norm = sqrt( (point[0])*(point[0]) + (point[1])*(point[1]) + (point[2])*(point[2]) );

    if(norm != 0){
        point[0] /= norm;
        point[1] /= norm;
        point[2] /= norm;
    }
}

__device__
Mat6f computeA_cuda(const Vec6f& gradient)   //phi is phi_current from the paper
{
    return gradient * gradient.transpose();
}

__device__
Vec6f computeb_cuda(const Vec6f &gradient, const Vec6f &ksi, float phi_cur, float phi_ref)
{
    float scalar = (gradient.transpose() * ksi)(0,0) + phi_ref - phi_cur;
    return gradient * scalar;
}

__device__
void skewSymmetric(const Vec3f &vec, Mat3f &matX)
{
    matX << 0, -vec(2), vec(1),
          vec(2), 0, -vec(0),
          -vec(1), vec(0), 0;
}

__global__
void constructSDFKernel(float* sdfs_ref, float* wsdfs_ref, float* sdfs_cur, float* wsdfs_cur, Mat4f gt, const Vec3f centroid, const Vec3f voxelSize, const Vec3f size, Vec3i dim)
{

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if(x >= dim[0] || y >= dim[1] || z >= dim[2])
        return;

    size_t off = z*dim[0]*dim[1] + y*dim[0] + x;

    //reference frame
    Vec3f world_point_current = voxelToCamera(Vec3i(x,y,z), centroid, voxelSize, size);
    float phi_cur = sdfs_cur[off];
    float weight_cur = wsdfs_cur[off];

    //current frame : rotate and translate
    Vec3f world_point_reference = transform(world_point_current, gt);
    Vec3i voxel_point_reference = cameraToVoxel(world_point_reference, centroid, voxelSize, size);


    if (voxel_point_reference[0] < 0 || voxel_point_reference[0] > dim[0] - 1 ||
            voxel_point_reference[1] < 0 || voxel_point_reference[1] > dim[1] - 1 ||
            voxel_point_reference[2] < 0 || voxel_point_reference[2] > dim[2] - 1)
        return;


    size_t off_reference = voxel_point_reference[2]*dim[0]*dim[1] + voxel_point_reference[1]*dim[0] + voxel_point_reference[0];
    float phi_ref = interpolate3voxel(voxel_point_reference[0], voxel_point_reference[1], voxel_point_reference[2], sdfs_ref, dim);
    float weight_ref = wsdfs_ref[off_reference];

    wsdfs_ref[off_reference] = weight_ref + weight_cur;
    if(weight_ref + weight_cur == 0) return;
    sdfs_ref[off_reference] = (weight_ref*phi_ref + weight_cur*phi_cur) / (weight_ref + weight_cur);
}


void constructSDFCuda(float* sdfs_ref, float* wsdfs_ref, float* sdfs_cur, float* wsdfs_cur, Mat4f& projection, Vec3f& centroid, Vec3f& voxelSize, Vec3f& size, Vec3i& dim)
{
    dim3 block(32, 8, 1);
    dim3 grid = computeGrid3D(block, dim[0], dim[1], dim[2]);

    constructSDFKernel <<<grid, block>>> (sdfs_ref, wsdfs_ref, sdfs_cur, wsdfs_cur, projection, centroid, voxelSize, size, dim);
    CUDA_CHECK;
}


__global__
void computeGradientByXKernel(Vec3f *gradientByX, const float *sdfs, Vec3i dim, Vec3f voxelSize)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= dim[0] || y >= dim[1] || z >= dim[2])
        return;

    Vec3f dX = voxelSize * 2;

    size_t off = z*dim[0]*dim[1] + y*dim[0] + x;
    size_t offX2 = z*dim[0]*dim[1] + y*dim[0] + x + ((x == dim[0] - 1) ? 0 : 1);
    size_t offX1 = z*dim[0]*dim[1] + y*dim[0] + x - ((x == 0) ? 0 : 1);
    size_t offY2 = z*dim[0]*dim[1] + (y + ((y == dim[1] - 1) ? 0 : 1))*dim[0] + x;
    size_t offY1 = z*dim[0]*dim[1] + (y - ((y == 0) ? 0 : 1))*dim[0] + x;
    size_t offZ2 = (z + ((z == dim[2] - 1) ? 0 : 1))*dim[0]*dim[1] + y*dim[0] + x;
    size_t offZ1 = (z - ((z == 0) ? 0 : 1))*dim[0]*dim[1] + y*dim[0] + x;

    gradientByX[off][0] = (sdfs[offX2] - sdfs[offX1]) / dX[0];
    gradientByX[off](1) = (sdfs[offY2] - sdfs[offY1]) / dX[1];
    gradientByX[off](2) = (sdfs[offZ2] - sdfs[offZ1]) / dX[2];

    normalize(gradientByX[off]);
}


__global__
void computeGradientKernel(Vec6f *gradient, Vec3f *gradientByX, Mat4f projection, Mat4f projectionInverse, Vec3f centroid, Vec3f voxelSize, Vec3f size, Vec3i dim)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= dim[0] || y >= dim[1] || z >= dim[2])
        return;

    //reference frame
    Vec3f world_point_reference = voxelToCamera(Vec3i(x,y,z),centroid, voxelSize, size);

    //current frame : rotate and translate
    Vec3f world_point_current = transform(world_point_reference, projection);

    Vec3f ksiV = (projectionInverse * world_point_current.homogeneous()).head(3);
    Mat3f ksiVX;
    skewSymmetric(ksiV, ksiVX);

    Eigen::Matrix<float, 3, 6> concatenation;
    concatenation << Mat3f::Identity(), -1 * ksiVX;

    size_t off_current = z*dim[0]*dim[1] + y*dim[0] + x;
    gradient[off_current] = gradientByX[off_current].transpose() * concatenation;
}

__global__
void computeStepParametersKernel(
        const float *tsdf_cur,
        const float *wsdf_cur,
        const float *tsdf_ref,
        const float *wsdf_ref,
        Mat4f projection_current,
        Vec6f *gradient,
        Vec3f centroid,
        Vec3f voxelSize,
        Vec3f size,
        Vec3i dim,
        Vec3i completeDim,
        int offsetZ,
        Vec6f ksi_cur,
        // results:
        float *A,
        float *b,
        float *error_function
)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= dim[0] || y >= dim[1] || z >= dim[2])
        return;

    size_t completeOff = (z + offsetZ)*completeDim[0]*completeDim[1] + (y)*completeDim[0] + (x);


    //reference frame
    Vec3f world_point_reference = voxelToCamera(Vec3i(x,y,z + offsetZ), centroid, voxelSize, size);
    float phi_ref = tsdf_ref[completeOff];
    float weight_ref = wsdf_ref[completeOff];


    //current frame : rotate and translate
    Vec3f world_point_current = transform(world_point_reference, projection_current);
    Vec3i voxel_point_current = cameraToVoxel(world_point_current, centroid, voxelSize, size);

    if (voxel_point_current[0] < 0 || voxel_point_current[0] > completeDim[0] - 1 ||
        voxel_point_current[1] < 0 || voxel_point_current[1] > completeDim[1] - 1 ||
        voxel_point_current[2] < 0 || voxel_point_current[2] > completeDim[2] - 1)
        return;


    //current frame : sdf and weight
    size_t off_current = voxel_point_current[2]*completeDim[0]*completeDim[1] + voxel_point_current[1]*completeDim[0] + voxel_point_current[0];
    float phi_current = interpolate3voxel(
            voxel_point_current[0],
            voxel_point_current[1],
            voxel_point_current[2],
            tsdf_cur,
            completeDim
    );
    float weight_current = wsdf_cur[off_current];

    //computation
    Mat6f A_mat = weight_ref * weight_current * computeA_cuda(gradient[off_current]);
    Vec6f b_vec = weight_ref * weight_current * computeb_cuda(gradient[off_current], ksi_cur, phi_current, phi_ref);
    // populate output matricies
    //compute global offset
    size_t off = z*dim[0]*dim[1] + y*dim[0] + x;
    size_t sizeAll = dim[0] * dim[1] * dim[2];
    for (int i = 0; i < 6 * 6; i++)
        A[off + sizeAll * i] = A_mat(i / 6, i % 6);
    for (int i = 0; i < 6; i++)
        b[off + sizeAll * i] = b_vec[i];
    error_function[off] = 0.5 * weight_ref * weight_current * (phi_current - phi_ref) * (phi_current - phi_ref);
}


void computeGradientByXCuda(Vec3f *gradientByX, const float *sdfs, Vec3f voxelSize, Vec3i dim)
{
    if (!gradientByX || !sdfs)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid3D(block, dim[0], dim[1], dim[2]);

    computeGradientByXKernel <<<grid, block>>> (gradientByX, sdfs, dim, voxelSize);

    CUDA_CHECK;
}

void computeGradientCuda(Vec6f *gradient, Vec3f *gradientByX, Mat4f projection, Vec3f centroid, Vec3f voxelSize, Vec3f size, Vec3i dim)
{
    if (!gradient || !gradientByX)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid3D(block, dim[0], dim[1], dim[2]);

    Mat4f projectionInverse = projection.inverse();
    ensure(projectionInverse(3,3) == 1);

    computeGradientKernel <<<grid, block>>> (gradient, gradientByX, projection, projectionInverse, centroid, voxelSize, size, dim);

    CUDA_CHECK;
}

void computeStepParametersCuda(
    const float *tsdf_cur,
    const float *wsdf_cur,
    const float *tsdf_ref,
    const float *wsdf_ref,
    Mat4f projection_current,
    Vec6f *gradient,
    Vec3f centroid,
    Vec3f voxelSize,
    Vec3f size,
    Vec3i dim,
    Vec3i completeDim,
    int offsetZ,
    Vec6f ksi_cur,
    // results:
    float *A,
    float *b,
    float *error_function
)
{
    // calculate block and grid size
    dim3 block(32, 32, 1);
    dim3 grid = computeGrid3D(block, dim[0], dim[1], dim[2]);

    computeStepParametersKernel <<<grid, block>>> (
        tsdf_cur,
        wsdf_cur,
        tsdf_ref,
        wsdf_ref,
        projection_current,
        gradient,
        centroid,
        voxelSize,
        size,
        dim,
        completeDim,
        offsetZ,
        ksi_cur,
        // results:
        A,
        b,
        error_function
    );

    CUDA_CHECK;
}

__global__
void initKernel(float *g_idata, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    g_idata[i] = 1.0;
}

void initCuda(float *input_d, int n) {
    // calculate block and grid size
    dim3 block(1024, 1, 1);
    dim3 grid = computeGrid1D(block, n);

    // run cuda kernel
    initKernel<<<grid, block, sizeof(float)>>>(input_d, n);

    CUDA_CHECK;
}

