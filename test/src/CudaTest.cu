#include <gtest/gtest.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <af/Helper.cuh>
#include <numeric>

__global__ void kernelTest(Vec3f* vec, Vec3i dim) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= dim[0] || y >= dim[1] || z >= dim[2])
        return;

    size_t off_current = z * dim[0] * dim[1] + y * dim[0] + x;
    vec[off_current]   = Vec3f(1, 1, 1);
}

void runKernelTest(Vec3f* vec, Vec3i dim, int blockSizeX, int blockSizeY, int blockSizeZ) {
    // calculate block and grid size
    dim3 block(blockSizeX, blockSizeY, blockSizeZ);
    dim3 grid = computeGrid3D(block, dim[0], dim[1], dim[2]);

    kernelTest<<<grid, block>>>(vec, dim);

    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void pushBackKernel(float* output, unsigned int* outputSize, float* vec, int vecSize) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= vecSize)
        return;

    if (cIdx % 4 == 0) {
        unsigned int lastSize = atomicAdd(outputSize, 1);
        output[lastSize]      = vec[cIdx];
    }
}

void runPushBackKernel(float* output, unsigned int* outputSize, float* vec, int vecSize) {
    // calculate block and grid size
    dim3 block, grid;
    setupBlockGrid(block, grid, vecSize);

    pushBackKernel<<<grid, block>>>(output, outputSize, vec, vecSize);

    CUDA_CHECK;
    cudaDeviceSynchronize();
}

__global__ void markKernel(float* output, unsigned int* outputSize, float* vec, int vecSize) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= vecSize)
        return;

    if (cIdx % 4 == 0) {
        output[cIdx] = vec[cIdx];
    } else {
        output[cIdx] = 0.f;
    }
}

void runMarkKernel(float* output, unsigned int* outputSize, float* vec, int vecSize) {
    // calculate block and grid size
    dim3 block, grid;
    setupBlockGrid(block, grid, vecSize);

    markKernel<<<grid, block>>>(output, outputSize, vec, vecSize);

    CUDA_CHECK;
    cudaDeviceSynchronize();
}

TEST(CudaTest, PushBackKernelTest) {
    Timer timer;
    std::vector<float> vec(100000000);
    std::vector<float> out(100000000);
    unsigned int outSize = 0;
    std::iota(vec.begin(), vec.end(), 0);

    float* vec_d            = allocateDeviceArrayCopy(vec.data(), vec.size());
    float* out_d            = allocateDeviceArray<float>(vec.size());
    unsigned int* outSize_d = allocateDeviceArrayCopy(&outSize, 1);
    Time("pushback kernel : ", timer, runPushBackKernel(out_d, outSize_d, vec_d, vec.size()); copyToHost(&outSize, outSize_d, 1);
         copyToHost(out.data(), out_d, outSize););
    std::cout << "outSize : \n" << outSize << "\n";
    std::cout << "out[0] : \n" << out[0] << "\n";
    std::cout << "out[outSize - outSize / 2] : \n" << out[outSize - outSize / 2] << "\n";
    std::cout << "out[outSize - 1] : \n" << out[outSize - 1] << "\n";

    Time("mark kernel : ", timer, runMarkKernel(out_d, outSize_d, vec_d, vec.size()); copyToHost(out.data(), out_d, out.size()););
    Time(
        "mark kernel post process: ", timer, std::vector<float> newVec; for (std::size_t i = 0; i < out.size(); ++i) {
            if (out[i] > 0.1f)
                newVec.push_back(out[i]);
        });
    std::cout << "newVec.size() : \n" << newVec.size() << "\n";
}

TEST(CudaTest, BasicOperationsSpeed) {
    Timer timer;

    std::vector<Vec3f> data;
    Vec3i dim(256, 256, 256);
    std::size_t size = dim[0] * dim[1] * dim[2];
    data.resize(size);

    timer.start();
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = Vec3f(2, 2, 2);
    }
    timer.end();
    auto cpuTime = timer.get();

    Time("cudaMalloc", timer, Vec3f* data_d = allocateDeviceArray<Vec3f>(size); CUDA_CHECK;);

    Time("cudaMemcpy", timer, cudaMemcpy(data_d, data.data(), sizeof(Vec3f) * size, cudaMemcpyHostToDevice););
    Time("runKernelTest32/32/1", timer, runKernelTest(data_d, dim, 32, 32, 1););
    Time("runKernelTest16/16/4", timer, runKernelTest(data_d, dim, 16, 16, 4););
    Time("runKernelTest1024/1/1", timer, runKernelTest(data_d, dim, 1024, 1, 1););
    Time("runKernelTest512/1/1", timer, runKernelTest(data_d, dim, 512, 1, 1););
    EXPECT_THROW(runKernelTest(data_d, dim, 2048, 1, 1), std::runtime_error);
    Time("cudaMemcpy", timer, cudaMemcpy(data.data(), data_d, sizeof(Vec3f) * size, cudaMemcpyDeviceToHost););
}

TEST(ThrustTest, Sort) {
    Timer timer;
    std::size_t size = 50000000;
    std::vector<float> vec(size);
    std::generate(vec.begin(), vec.end(), []() { return rand(); });
    float* vec_d = allocateDeviceArrayCopy(vec.data(), size);
    Time("std::sort : ", timer, std::sort(vec.begin(), vec.end()););
    Time("thrust::sort : ", timer, thrust::device_ptr<float> vec_d_ptr(vec_d); thrust::sort(vec_d_ptr, vec_d_ptr + size););
    if (size > 50)
        return;

    std::cout << "vec : ";
    for (std::size_t i = 0; i < size; ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << "\n";
    copyToHost(vec.data(), vec_d, size);
    std::cout << "vec_d : ";
    for (std::size_t i = 0; i < size; ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << "\n";
}

#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

__global__ void sortKernel(bool* results, Veci<6>* indicies, int size) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= size)
        return;

    Veci<6> iDxs = indicies[cIdx];

    thrust::device_ptr<int> iDxsPtr(&iDxs[0]);
    thrust::sort(thrust::device, iDxsPtr, iDxsPtr + 6);
    results[cIdx]  = thrust::is_sorted(thrust::device, iDxsPtr, iDxsPtr + 6);
    indicies[cIdx] = iDxs;
}

void runSortKernel(bool* results, Veci<6>* indicies, int size) {
    dim3 block, grid;
    setupBlockGrid(block, grid, size);
    sortKernel<<<grid, block>>>(results, indicies, size);
}

TEST(ThrustTest, SortInKernel) {
    std::vector<Veci<6>> vec;
    vec.push_back(Veci<6>::Random());
    vec.push_back(Veci<6>::Random());
    vec.push_back(Veci<6>::Random());
    vec.push_back(Veci<6>::Random());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i].transpose() << "\n";
    }
    bool* isSorted = new bool[vec.size()];

    Veci<6>* vec_d = allocateDeviceArrayCopy(vec.data(), vec.size());
    bool* res_d    = allocateDeviceArray<bool>(vec.size());
    runSortKernel(res_d, vec_d, vec.size());
    copyToHost(vec.data(), vec_d, vec.size());
    copyToHost(isSorted, res_d, vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::cout << "isSorted " << isSorted[i] << " : " << vec[i].transpose() << "\n";
    }
}

void combinations(std::vector<std::pair<int, int>>& pairs, const std::vector<int>& indicies) {
    for (std::size_t i = 0; i < indicies.size(); ++i) {
        for (std::size_t j = 0; j < indicies.size(); ++j) {
            if (indicies[j] > indicies[i])
                pairs.push_back({indicies[i], indicies[j]});
        }
    }
}

void combinationsSort(std::vector<std::pair<int, int>>& pairs, std::vector<int>& indicies) {
    std::sort(indicies.begin(), indicies.end());
    for (std::size_t i = 0; i < indicies.size(); ++i) {
        for (std::size_t j = i + 1; j < indicies.size(); ++j) {
            pairs.push_back({indicies[i], indicies[j]});
        }
    }
}

TEST(General, CombinationsGenerationSpeed) {
    std::vector<int> data(10000);
    std::vector<std::pair<int, int>> vCombinations;
    std::iota(data.begin(), data.end(), 0);
    std::reverse(data.begin(), data.end());
    Timer timer;
    Time("combinations : ", timer, combinations(vCombinations, data););
    vCombinations.clear();
    Time("combinationsSort : ", timer, combinationsSort(vCombinations, data););
}

// namespace thrust {
// __device__ __host__ bool operator<(const Vec2i& a, const Vec2i& b) {
//     if (a[0] < b[0])
//         return true;
//     if (b[0] < a[0])
//         return false;
//     return a[1] < b[1];
// }
// }  // namespace thrust

void print(const Vec2i& key, int value) { std::cout << "(" << key[0] << "," << key[1] << ") - " << value << "\n"; }
void print(const long long& key, int value) { std::cout << "(" << key << ") - " << value << "\n"; }

TEST(ThrustTest, SortByKeyVec2i) {
    Timer timer;
    std::vector<Vec2i> keys;
    std::vector<int> values;
    for (std::size_t i = 0; i < 800000; ++i) {
        keys.push_back({rand() % 20, rand() % 20});
        values.push_back(rand() % 20);
    }

    Vec2i* keys_d             = allocateDeviceArrayCopy(keys.data(), keys.size());
    int* values_d             = allocateDeviceArrayCopy(values.data(), values.size());
    long long* keysLongLong_d = allocateDeviceArrayCopy<Vec2i, long long>(keys.data(), keys.size());
    int* valuesLongLong_d     = allocateDeviceArrayCopy(values.data(), values.size());

    std::cout << "sizeof(Vec2i) : \n" << sizeof(Vec2i) << "\n";
    std::cout << "sizeof(int) : \n" << sizeof(int) << "\n";
    std::cout << "sizeof(long long) : \n" << sizeof(long long) << "\n";
    std::cout << "sizeof(std::size_t) : \n" << sizeof(std::size_t) << "\n";

    thrust::device_ptr<Vec2i> keys_d_ptr(keys_d);
    thrust::device_ptr<int> values_d_ptr(values_d);
    Time("sort by Vec2i key : ", timer, thrust::sort_by_key(keys_d_ptr, keys_d_ptr + keys.size(), values_d_ptr););

    thrust::device_ptr<long long> keysLongLong_d_ptr(keysLongLong_d);
    thrust::device_ptr<int> valuesLongLong_d_ptr(valuesLongLong_d);
    Time("sort by long long key : ", timer,
         thrust::sort_by_key(keysLongLong_d_ptr, keysLongLong_d_ptr + keys.size(), valuesLongLong_d_ptr););

    // Sorting by long long instead of Vec2i doesn't produce valid results and is actually slower the Vec2i on sizes up to 800000.
    // Better us Vec2i as a key.
    EXPECT_FALSE(thrust::equal(values_d_ptr, values_d_ptr + values.size(), valuesLongLong_d_ptr));
}

TEST(ThrustTest, ReduceSum) {
    Timer timer;
    std::vector<Mat6f> mats;
    std::vector<Vec2i> keys;
    std::vector<int> simple_values;
    std::vector<Vec2i> simple_keys;
    for (int i = 0; i < 80000; ++i) {
        mats.push_back(Mat6f::Random());
        keys.push_back(Vec2i(i % 1000, i % 10));
        simple_values.push_back(rand());
        simple_keys.push_back(Vec2i(i % 1000, i % 10));
    }
    // mats.push_back(Mat6f::Identity());
    // mats.push_back(Mat6f::Identity());
    // mats.push_back(Mat6f::Identity());
    // mats.push_back(Mat6f::Identity());
    // keys.push_back(Vec2i(0, 1));
    // keys.push_back(Vec2i(0, 1));
    // keys.push_back(Vec2i(0, 1));
    // keys.push_back(Vec2i(0, 2));

    thrust::device_vector<Mat6f> matsVec_d       = mats;
    thrust::device_vector<Vec2i> keysVec_d       = keys;
    thrust::device_vector<int> simpleValuesVec_d = simple_values;
    thrust::device_vector<Vec2i> simpleKeysVec_d = simple_keys;
    Vec2i* keys_d                                = allocateDeviceArrayCopy(keys.data(), keys.size());
    Mat6f* mats_d                                = allocateDeviceArrayCopy(mats.data(), mats.size());
    Vec2i* outputKeys_d                          = allocateDeviceArray<Vec2i>(keys.size());
    Mat6f* outputMats_d                          = allocateDeviceArray<Mat6f>(mats.size());
    thrust::device_ptr<Vec2i> Keys_d_Ptr(keys_d);
    thrust::device_ptr<Mat6f> Mats_d_Ptr(mats_d);
    thrust::device_ptr<Vec2i> OutputKeys_d_Ptr(outputKeys_d);
    thrust::device_ptr<Mat6f> OutputMats_d_Ptr(outputMats_d);

    thrust::device_vector<Vec2i> outputKeysVec_d(mats.size());
    thrust::device_vector<Mat6f> outputMatsVec_d(mats.size());

    Time("sort_by_key (Vec2i, int)", timer,
         thrust::sort_by_key(simpleKeysVec_d.begin(), simpleKeysVec_d.end(), simpleValuesVec_d.begin()););

    Time("sort_by_key (Vec2i, Mat6f)", timer, thrust::sort_by_key(Keys_d_Ptr, Keys_d_Ptr + keys.size(), Mats_d_Ptr););
    Time("reduce_by_key", timer,
         auto itEndPair = thrust::reduce_by_key(Keys_d_Ptr, Keys_d_Ptr + keys.size(), Mats_d_Ptr, OutputKeys_d_Ptr,
                                                OutputMats_d_Ptr););

    std::cout << "mats.size() : \n" << mats.size() << "\n";
    std::cout << "outputKeysVec_d.size() : \n" << outputKeysVec_d.size() << "\n";
    std::cout << "outputMatsVec_d.size() : \n" << outputMatsVec_d.size() << "\n";
    std::cout << "result Keys size : " << (itEndPair.first - OutputKeys_d_Ptr) << "\n";
    std::cout << "result Mats size : " << (itEndPair.second - OutputMats_d_Ptr) << "\n";
}

TEST(ThrustTest, ReduceSumCompareCpu) {
    Timer timer;
    std::vector<Vec3f> points;
    for (int i = 0; i < 20000; ++i) {
        if (i % 2)
            points.push_back(Vec3f(0, 0, 0));
        else
            points.push_back(Vec3f::Random());
    }
    thrust::device_vector<Vec3f> points_d = points;
    for (int j = 0; j < 10; ++j) {
        Vec3f sumCpu(0, 0, 0);
        int countCpu = 0;
        for (int i = 0; i < 20000; ++i) {
            sumCpu += points[i];
            if (hasPositiveDepth()(points[i]))
                countCpu++;
        }

        Vec3f sumThrust = thrust::reduce(points_d.begin(), points_d.end(), Vec3f(0, 0, 0));
        int countThrust = thrust::count_if(thrust::device, points_d.begin(), points_d.end(), hasPositiveDepth());

        std::cout << "sumCpu : " << sumCpu << "\n";
        std::cout << "countCpu : " << countCpu << "\n";
        std::cout << "sumThrust : " << sumThrust << "\n";
        std::cout << "countThrust : " << countThrust << "\n";
        std::cout << "sumCpu / countCpu : " << sumCpu / countCpu << "\n";
        std::cout << "sumThrust / countThrust : " << sumThrust / countThrust << "\n";
    }
}

#include <af/DebugHelper.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolverSp.h>

#include <af/LinearSolver.cuh>
TEST(CuSolver, Cholesky_Small) {
    Timer timer;

    const int m = 4;
    Mat4f A;
    // clang-format off
    A << 1., 0., 0., 0.1, 0., 2., 0., 0.1, 0., 0., 3., 0.1, 0.1, 0.1, 0.1, 4.;
    Vec4f b;
    b << 1., 1., 1., 1.;
    Time("A.inverse() * b", timer,
        Vec4f expectedX = A.inverse() * b;
    );

    // GPU Cholesky Dense

    thrust::device_vector<float> ADense_d(A.size());
    thrust::device_vector<float> BDense_d(b.size());
    thrust::copy_n(&(A(0, 0)), A.size(), ADense_d.begin());
    thrust::copy_n(b.begin(), b.size(), BDense_d.begin());

    Time("cusolverDnCreate", timer,
    cusolverDnHandle_t handleDn = af::createHandleDn();
    CUDA_CHECK;
    );

    Time("linSolvCholDn", timer,
        af::linSolvCholDn(handleDn, ADense_d.data().get(), BDense_d.data().get(), m);
    );

    // GPU LU Dense

    thrust::device_vector<float> ADenseLU_d(A.size());
    thrust::device_vector<float> BDenseLU_d(b.size());
    thrust::copy_n(&(A(0, 0)), A.size(), ADenseLU_d.begin());
    thrust::copy_n(b.begin(), b.size(), BDenseLU_d.begin());

    // Time("cusolverDnCreate", timer,
    // cusolverDnHandle_t handleDn = af::createHandleDn();
    // CUDA_CHECK;
    // );

    Time("linSolvCholDn", timer,
        af::linSolvLUDn(handleDn, ADenseLU_d.data().get(), BDenseLU_d.data().get(), m);
    );

    // GPU Cholesky Sparse

    cusolverSpHandle_t handleSp = NULL;
    csrqrInfo_t info          = NULL;
    cusparseMatDescr_t descrA = NULL;

    const float tol                         = 1.e-4;
    const int nnzA                          = 7;
    thrust::device_vector<int> csrRowPtrA_d = std::vector<int>{1, 2, 3, 4, nnzA + 1};
    thrust::device_vector<int> csrColIndA_d = std::vector<int>{1, 2, 3, 1, 2, 3, 4};
    thrust::device_vector<float> csrValA_d  = std::vector<float>{1.0, 2.0, 3.0, 0.1, 0.1, 0.1, 4.0};
    thrust::device_vector<float> b_d        = std::vector<float>{1.0, 1.0, 1.0, 1.0};
    thrust::device_vector<float> x_d(m);
    thrust::device_vector<int> singularity_d(1);
    int singularity = 0;

    assert(cusolverSpCreate(&handleSp) == cudaSuccess);

    assert(cusparseCreateMatDescr(&descrA) == cudaSuccess);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);  // base-1

    Time("cusolverSpScsrlsvchol", timer,
         cusolverSpScsrlsvchol(handleSp, m, nnzA, descrA, csrValA_d.data().get(), csrRowPtrA_d.data().get(),
                               csrColIndA_d.data().get(), b_d.data().get(), tol, 0, x_d.data().get(), &singularity););

    std::cout << "expectedX.transpose() : " << expectedX.transpose() << "\n";

    std::cout << "result dense chol : ";
    debug::print(BDense_d, 1, BDense_d.size());

    std::cout << "result dense lu : ";
    debug::print(BDenseLU_d, 1, BDenseLU_d.size());

    std::cout << "result sparse : ";
    debug::print(x_d, 1, x_d.size());
    std::cout << "singularity : " << singularity << "\n";
}

TEST(CuSolver, CholeskySparse_Speed) {
    Timer timer;

    cusolverSpHandle_t cusolverH = NULL;
    csrqrInfo_t info          = NULL;
    cusparseMatDescr_t descrA = NULL;

    const int m = 10000;

    const float tol = 1.e-4;
    const int nnzA  = 2 * m - 1;
    std::vector<int> csrRowPtrA;
    std::vector<int> csrColIndA;
    std::vector<float> csrValA;
    std::vector<float> b(m, 1.f);
    for (std::size_t i = 0; i < m - 1; ++i) {
        csrRowPtrA.push_back(i + 1);
        csrColIndA.push_back(i + 1);
        csrValA.push_back(i + 1);
    }
    csrRowPtrA.push_back(m);
    for (std::size_t i = 0; i < m; ++i) {
        csrColIndA.push_back(i + 1);
        csrValA.push_back(0.1);
    }
    csrValA.back() = m;
    csrRowPtrA.push_back(2 * m);

    thrust::device_vector<int> csrRowPtrA_d = csrRowPtrA;
    thrust::device_vector<int> csrColIndA_d = csrColIndA;
    thrust::device_vector<float> csrValA_d  = csrValA;
    thrust::device_vector<float> b_d        = b;
    thrust::device_vector<float> x_d(m);
    thrust::device_vector<int> singularity_d(1);
    int singularity = 0;

    assert(cusolverSpCreate(&cusolverH) == cudaSuccess);

    assert(cusparseCreateMatDescr(&descrA) == cudaSuccess);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);  // base-1

    cusolverSpCreateCsrqrInfo(&info);

    Time("cusolverSpScsrlsvchol", timer,
         cusolverSpScsrlsvchol(cusolverH, m, nnzA, descrA, csrValA_d.data().get(), csrRowPtrA_d.data().get(),
                               csrColIndA_d.data().get(), b_d.data().get(), tol, 0, x_d.data().get(), &singularity););
    Time("cusolverSpScsrlsvqr", timer,
         cusolverSpScsrlsvqr(cusolverH, m, nnzA, descrA, csrValA_d.data().get(), csrRowPtrA_d.data().get(),
                             csrColIndA_d.data().get(), b_d.data().get(), tol, 0, x_d.data().get(), &singularity););

    std::cout << "singularity : " << singularity << "\n";
}

TEST(CuSolver, CholeskyDense_Speed) {
    Timer timer;

    cusolverDnHandle_t handleDn = NULL;
    const cublasFillMode_t uplo  = CUBLAS_FILL_MODE_LOWER;
    const int nrhs               = 1;

    const int m = 10000;

    std::vector<float> ADense(m * m, 0.f);
    std::vector<float> BDense(m, 1.f);
    for (std::size_t i = 0; i < m; ++i) {
        ADense[i * m + i]       = i + 1;
        ADense[(m - 1) * m + i] = 0.1;
        ADense[i * m + m - 1]   = 0.1;
    }
    ADense[m * m - 1] = m;

    thrust::device_vector<float> ADense_d = ADense;
    thrust::device_vector<float> BDense_d = BDense;
    CUDA_CHECK;

    Time("cusolverDnCreate", timer,
    assert(cusolverDnCreate(&handleDn) == cudaSuccess);
    );

    int workSize = 0;
    Time("cusolverDnSpotrf_bufferSize", timer,
    cusolverDnSpotrf_bufferSize(handleDn, uplo, m, ADense_d.data().get(), m, &workSize);
    std::cout << "workSize : " << workSize << "\n";
    CUDA_CHECK;
    );
    Time("workspace_d", timer,
    thrust::device_vector<float> workspace_d(workSize);
    thrust::device_vector<int> devInfo_d(1, 0);
    );

    Time("cusolverDnSpotrf", timer,
        cusolverDnSpotrf(handleDn, uplo, m, ADense_d.data().get(), m, workspace_d.data().get(), workSize, devInfo_d.data().get());
    CUDA_CHECK;
    );
    Time("cusolverDnSpotrs", timer,
    cusolverDnSpotrs(handleDn, uplo, m, nrhs, ADense_d.data().get(), m, BDense_d.data().get(), m, devInfo_d.data().get());
    CUDA_CHECK;
    );

    std::cout << "devInfo_d[0] : " << devInfo_d[0] << "\n";

}

/*
 *  solve A*x = b by LU with partial pivoting
 *
 */
int linearSolverLU(
    cusolverDnHandle_t handle,
    int n,
    const float *Acopy,
    int lda,
    const float *b,
    float *x)
{
    int bufferSize = 0;
    int *info = NULL;
    float *buffer = NULL;
    float *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;
    // float start, stop;
    // float time_solve;

    cusolverDnSgetrf_bufferSize(handle, n, n, (float*)Acopy, lda, &bufferSize);

    cudaMalloc(&info, sizeof(int));
    cudaMalloc(&buffer, sizeof(float)*bufferSize);
    cudaMalloc(&A, sizeof(float)*lda*n);
    cudaMalloc(&ipiv, sizeof(int)*n);


    // prepare a copy of A because getrf will overwrite A with L
    cudaMemcpy(A, Acopy, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice);
    cudaMemset(info, 0, sizeof(int));

    // start = second();
    // start = second();

    cusolverDnSgetrf(handle, n, n, A, lda, buffer, ipiv, info);
    cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    cudaMemcpy(x, b, sizeof(float)*n, cudaMemcpyDeviceToDevice);
    cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info);
    cudaDeviceSynchronize();
    // stop = second();

    // time_solve = stop - start;
    // fprintf (stdout, "timing: LU = %10.6f sec\n", time_solve);

    if (info  ) { cudaFree(info  ); }
    if (buffer) { cudaFree(buffer); }
    if (A     ) { cudaFree(A); }
    if (ipiv  ) { cudaFree(ipiv);}

    return 0;
}

/*
 *  solve A*x = b by LU with partial pivoting
 *
 */
int linearSolverLU_Alt(
    cusolverDnHandle_t handle,
    int n,
    float *A,
    int lda,
    float *b,
    int* ipiv)
{
    int bufferSize = 0;
    int *info = NULL;
    float *buffer = NULL;
    // float *A = NULL;
    // int *ipiv = NULL; // pivoting sequence
    int h_info = 0;
    // float start, stop;
    // float time_solve;

    cusolverDnSgetrf_bufferSize(handle, n, n, (float*)A, lda, &bufferSize);

    cudaMalloc(&info, sizeof(int));
    cudaMalloc(&buffer, sizeof(float)*bufferSize);
    // cudaMalloc(&A, sizeof(float)*lda*n);
    // cudaMalloc(&ipiv, sizeof(int)*n);


    // prepare a copy of A because getrf will overwrite A with L
    // cudaMemcpy(A, Acopy, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice);
    cudaMemset(info, 0, sizeof(int));

    // start = second();
    // start = second();

    cusolverDnSgetrf(handle, n, n, A, lda, buffer, ipiv, info);
    cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, b, n, info);
    cudaDeviceSynchronize();
    // stop = second();

    // time_solve = stop - start;
    // fprintf (stdout, "timing: LU = %10.6f sec\n", time_solve);

    if (info  ) { cudaFree(info  ); }
    if (buffer) { cudaFree(buffer); }
    // if (A     ) { cudaFree(A); }
    // if (ipiv  ) { cudaFree(ipiv);}

    return 0;
}

TEST(CuSolver, DenseLUGithubChangeDims) {
    Timer timer;
    Time("cusolverDnCreate", timer,
        cusolverDnHandle_t handleDn = af::createHandleDn();
        CUDA_CHECK;
    );

    int start = 2;
    int stop = 50;
    int step = 1;
    thrust::device_vector<float> ADense_d(stop*stop);
    thrust::device_vector<float> BDense_d(stop);
    thrust::device_vector<int> ipiv_d(stop);
    // for (int count = 1; count < 30; ++count) {
    //     int m = 12;
    for (int m = start; m < stop; m += step) {
        Eigen::MatrixXf A(m, m);
        Eigen::VectorXf b(m);
        thrust::device_vector<float> dv_X(m);
        thrust::fill_n(ADense_d.begin(), m*m, 0.f);
        thrust::fill_n(BDense_d.begin(), m, 0.f);
        runInitIdentityKernel(ADense_d.data().get(), stop);
        cudaDeviceSynchronize();
        A.fill(0.f);
        for (int i = 0; i < m - 1; ++i) {
            A(i, i) = 1.f;
            A(i, i + 1) = 1.f;
        }
        A(m - 1, m - 1) = 1.f;
        b.fill(m * 2);
        b[m - 1] = m;

        // std::cout << "A : " << A << "\n";
        // std::cout << "b : " << b << "\n";

        thrust::copy_n(A.data(), m*m, ADense_d.begin());
        thrust::copy_n(b.data(), m, BDense_d.begin());
        cudaDeviceSynchronize();

        af::linSolvLUDn(handleDn, ADense_d.data().get(), BDense_d.data().get(), m);
        // linearSolverLU(handleDn, m, ADense_d.data().get(), m, BDense_d.data().get(), dv_X.data().get());
        cudaDeviceSynchronize();

        thrust::copy_n(BDense_d.begin(), m, b.data());
        cudaDeviceSynchronize();
        std::cout << "b.transpose() : " << b.transpose() << "\n";

        //////////

        af::linearSolverLU(handleDn, m, ADense_d.data().get(), m, BDense_d.data().get(), ipiv_d.data().get());

        thrust::copy_n(BDense_d.begin(), m, b.data());
        cudaDeviceSynchronize();
        std::cout << "b.transpose() _alt : " << b.transpose() << "\n";
        cudaDeviceSynchronize();

    }
    cusolverDnDestroy(handleDn);
}

TEST(CuSolver, DenseLUChangeDims) {
    Timer timer;
    Time("cusolverDnCreate", timer,
        cusolverDnHandle_t handleDn = af::createHandleDn();
        CUDA_CHECK;
    );

    int start = 2;
    int stop = 30;
    int step = 1;
    thrust::device_vector<float> ADense_d(stop*stop);
    thrust::device_vector<float> BDense_d(stop);
    for (int count = 1; count < 30; ++count) {
        int m = 12;
    // for (int m = start; m < stop; m += step) {
        Eigen::MatrixXf A(m, m);
        Eigen::VectorXf b(m);
        thrust::fill_n(ADense_d.begin(), m*m, 0.f);
        thrust::fill_n(BDense_d.begin(), m, 0.f);
        runInitIdentityKernel(ADense_d.data().get(), stop);
        cudaDeviceSynchronize();
        A.fill(0.f);
        for (int i = 0; i < m - 1; ++i) {
            A(i, i) = 1.f;
            A(i, i + 1) = 1.f;
        }
        A(m - 1, m - 1) = 1.f;
        b.fill(count * 2);
        b[m - 1] = count;

        // std::cout << "A : " << A << "\n";
        // std::cout << "b : " << b << "\n";

        thrust::copy_n(A.data(), m*m, ADense_d.begin());
        thrust::copy_n(b.data(), m, BDense_d.begin());
        cudaDeviceSynchronize();

        af::linSolvLUDn(handleDn, ADense_d.data().get(), BDense_d.data().get(), m);
        cudaDeviceSynchronize();

        thrust::copy_n(BDense_d.begin(), m, b.data());
        cudaDeviceSynchronize();
        std::cout << "b.transpose() : " << b.transpose() << "\n";
    }
    cusolverDnDestroy(handleDn);
}
TEST(CuSolver, DenseQRChangeDims) {
    Timer timer;
    Time("cublasCreate", timer,
        cublasHandle_t handleCublas;
        cublasCreate(&handleCublas);
        CUDA_CHECK;
    );
    Time("cusolverDnCreate", timer,
        cusolverDnHandle_t handleDn = af::createHandleDn();
        CUDA_CHECK;
    );

    int start = 2;
    int stop = 50;
    int step = 1;
    thrust::device_vector<float> ADense_d(stop*stop);
    thrust::device_vector<float> BDense_d(stop);
    // for (int count = 0; count < 30; ++count) {
    //     int m = 7;
    for (int m = start; m < stop; m += step) {
        Eigen::MatrixXf A(m, m);
        Eigen::VectorXf b(m);
        thrust::fill_n(ADense_d.begin(), m*m, 0.f);
        thrust::fill_n(BDense_d.begin(), m, 0.f);
        A.fill(0.0f);
        for (int i = 0; i < m - 1; ++i) {
            A(i, i) = 1.f;
            A(i, i + 1) = 1.f;
        }
        A(m - 1, m - 1) = 1.f;
        b.fill(1.f);
        b[m - 1] = 0.5f;

        thrust::copy_n(A.data(), m*m, ADense_d.begin());
        thrust::copy_n(b.data(), m, BDense_d.begin());

        af::linSolvQRDn(handleDn, handleCublas, ADense_d.data().get(), BDense_d.data().get(), m);
        cudaDeviceSynchronize();

        thrust::copy_n(BDense_d.begin(), m, b.data());
        std::cout << "b.transpose() : " << b.transpose() << "\n";
    }
}

TEST(DELETEEEE, SIMPLE_RUN) {
    thrust::device_vector<float> dv_A(40);
    std::cout << "dv_A.size() : " << dv_A.size() << "\n";
}

#include <af/DeviceBuffer.cuh>

__global__ void testKernel(af::Device_Buff_View<float> view, float value) {
    view.push_back(value);
}

TEST(DELETEEEE, TEST_DEVICE_BUFFER) {
    af::DeviceBufferCounted<float>mesh_d(100, 0);
    std::cout << "mesh_d : ";
    for (auto&& v:mesh_d) {
        std::cout << v << " ";
    }
    std::cout << "\n";

    testKernel<<<10, 1>>>(af::createView(mesh_d), 4.f);
   mesh_d.syncHostSize();

    std::cout << "mesh_d after kernel: ";
    for (auto&& v:mesh_d) {
        std::cout << v << " ";
    }
    std::cout << "\n";
}

__global__ void ReadWriteTestKernel2d(float* data, std::size_t size, int dims) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= size)
        return;

    for (int i = 0; i < dims; ++i) {
        float x = data[cIdx + size * i];
        data[cIdx + size * i] = x * 2;
    }
}

__global__ void ReadWriteTestKernel2d2d(float* data, std::size_t size, int dims) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    const int cIdy = threadIdx.y + blockDim.y * blockIdx.y;
    if (cIdx >= size || cIdy >= dims)
        return;

    float x = data[cIdx + size * cIdy];
    data[cIdx + size * cIdy] = x * 2;
}

template<int K>
__global__ void ReadWriteTestKernelEigen(Vecf<K>* data, std::size_t size) {
    const int cIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (cIdx >= size)
        return;

    Vecf<K> x = data[cIdx];
    data[cIdx] = x * 2;
}

template<int K>
void ReadWriteSpeedTest() {
    Timer timer;
    std::cout << "K : " << K << "\n";
    int aIterations = 20;
    for (std::size_t size = 10000; size < 1000000000; size *= 10) {
        std::cout << "size : " << size << "\n";
        thrust::device_vector<Vecf<K>> vecEigen(size, Vecf<K>::Zero());
        thrust::device_vector<float> vec2d(size * K, 0.f);

        dim3 block, grid;
        setupBlockGrid(block, grid, size);

        timer.start();
        for (size_t vLoopIter = 0; vLoopIter < aIterations; ++vLoopIter) {
            ReadWriteTestKernelEigen<<<grid, block>>>(vecEigen.data().get(), size);
            CUDA_CHECK;
            cudaDeviceSynchronize();
        }
        timer.end();
        std::cout << "Time: " << "eigen vec" << ": " << timer.get()/aIterations << "s.\n";


        timer.start();
        for (size_t vLoopIter = 0; vLoopIter < aIterations; ++vLoopIter) {
            ReadWriteTestKernel2d<<<grid, block>>>(vec2d.data().get(), size, K);
            CUDA_CHECK;
            cudaDeviceSynchronize();
        }
        timer.end();
        std::cout << "Time: " << "2d array" << ": " << timer.get()/aIterations << "s.\n";

        dim3 block2, grid2;
        setupBlockGrid(block2, grid2, size, K);
        timer.start();
        for (size_t vLoopIter = 0; vLoopIter < aIterations; ++vLoopIter) {
            ReadWriteTestKernel2d2d<<<grid2, block2>>>(vec2d.data().get(), size, K);
            CUDA_CHECK;
            cudaDeviceSynchronize();
        }
        timer.end();
        std::cout << "Time: " << "2d array and grid" << ": " << timer.get()/aIterations << "s.\n";

    }
}

TEST(EigenVs2dArrayTest, ReadWriteSpeed) {
    ReadWriteSpeedTest<2>();
    ReadWriteSpeedTest<3>();
    ReadWriteSpeedTest<4>();
    ReadWriteSpeedTest<5>();
    ReadWriteSpeedTest<6>();
}