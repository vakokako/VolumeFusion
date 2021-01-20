#include <af/LinearSolver.cuh>

namespace af {

cusolverDnHandle_t createHandleDn() {
    cusolverDnHandle_t handleDn = NULL;
    cusolverDnCreate(&handleDn);
    CUDA_CHECK;
    return handleDn;
}

bool linSolvCholDn(cusolverDnHandle_t handleDn, float* ADense, float* bDense, const int dim) {
    if (ADense == NULL || bDense == NULL)
        return false;
    // throw std::runtime_error("af::linSolvCholDn(): input arrays cannot be empty.");

    Timer timer;

    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    const int nrhs              = 1;
    const int lda               = dim;
    const int ldb               = dim;

    int workSize = 0;
    cusolverDnSpotrf_bufferSize(handleDn, uplo, dim, ADense, lda, &workSize);
    CUDA_CHECK;

    thrust::device_vector<float> workspace_d(workSize);
    thrust::device_vector<int> devInfo_d(1, 0);

    // Cholesky Factorization
    cusolverDnSpotrf(handleDn, uplo, dim, ADense, lda, workspace_d.data().get(), workSize, devInfo_d.data().get());
    if (devInfo_d[0] != 0) {
        std::cout << "af::linSolvCholDn(): cusolverDnSpotrf failed, devinfo : " + std::to_string(devInfo_d[0]) << "\n";
        return false;
        // throw std::runtime_error("af::linSolvCholDn(): cusolverDnSpotrf failed, devinfo : " + std::to_string(devInfo_d[0]));
    }
    CUDA_CHECK;

    // Solve Ax = b
    cusolverDnSpotrs(handleDn, uplo, dim, nrhs, ADense, lda, bDense, ldb, devInfo_d.data().get());
    if (devInfo_d[0] != 0) {
        std::cout << "af::cusolverDnSpotrs(): cusolverDnSpotrf failed, devinfo : " + std::to_string(devInfo_d[0]) << "\n";
        return false;
        // throw std::runtime_error("af::linSolvCholDn(): cusolverDnSpotrs failed, devinfo : " + std::to_string(devInfo_d[0]));
    }
    CUDA_CHECK;

    return true;
}

bool linSolvLUDn(cusolverDnHandle_t handleDn, float* ADense, float* bDense, const int dim) {
    if (ADense == NULL || bDense == NULL)
        return false;
    // throw std::runtime_error("af::linSolvLUDn(): input arrays cannot be empty.");
    cudaStream_t stream = NULL;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(handleDn, stream);

    const int nrhs = 1;
    const int lda  = dim;
    const int ldb  = dim;

    int workSize = 0;
    cusolverDnSgetrf_bufferSize(handleDn, dim, dim, ADense, lda, &workSize);
    cudaDeviceSynchronize();
    CUDA_CHECK;

    thrust::device_vector<float> workspace_d(workSize);
    thrust::device_vector<int> devInfo_d(1, 0);

    // LU Factorization
    cusolverDnSgetrf(handleDn, dim, dim, ADense, lda, workspace_d.data().get(), NULL, devInfo_d.data().get());
    cudaDeviceSynchronize();
    if (devInfo_d[0] != 0)
        return false;
    // throw std::runtime_error("af::linSolvLUDn(): cusolverDnSgetrf failed, devinfo : " + std::to_string(devInfo_d[0]));
    CUDA_CHECK;

    // Solve Ax = b
    cusolverDnSgetrs(handleDn, CUBLAS_OP_N, dim, nrhs, ADense, lda, NULL, bDense, ldb, devInfo_d.data().get());
    cudaDeviceSynchronize();
    if (devInfo_d[0] != 0)
        throw std::runtime_error("af::linSolvLUDn(): cusolverDnSgetrs failed, devinfo : " + std::to_string(devInfo_d[0]));
    CUDA_CHECK;

    if (stream)
        cudaStreamDestroy(stream);

    return true;
}

bool linSolvQRDn(cusolverDnHandle_t handleDn, cublasHandle_t handleCublas, float* ADense, float* bDense, const int dim) {
    if (ADense == NULL || bDense == NULL)
        return false;

    const int m    = dim;
    const int lda  = m;
    const int ldb  = m;
    const int nrhs = 1;

    thrust::device_vector<float> tau_d(m);

    int workSize = 0;
    cusolverDnSgeqrf_bufferSize(handleDn, dim, dim, ADense, lda, &workSize);
    cudaDeviceSynchronize();
    CUDA_CHECK;

    thrust::device_vector<float> workspace_d(workSize);
    thrust::device_vector<int> devInfo_d(1, 0);

    cusolverDnSgeqrf(handleDn, dim, dim, ADense, lda, tau_d.data().get(), workspace_d.data().get(), workSize,
                     devInfo_d.data().get());
    cudaDeviceSynchronize();
    if (devInfo_d[0] != 0) {
        std::cout << "af::linSolvQRDn(): cusolverDnSgeqrf failed, devinfo : " + std::to_string(devInfo_d[0]) << "\n";
        return false;
        // throw std::runtime_error("af::linSolvCholDn(): cusolverDnSpotrf failed, devinfo : " + std::to_string(devInfo_d[0]));
    }
    CUDA_CHECK;

    cusolverDnSormqr(handleDn, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m, ADense, lda, tau_d.data().get(), bDense, ldb,
                     workspace_d.data().get(), workSize, devInfo_d.data().get());
    cudaDeviceSynchronize();
    if (devInfo_d[0] != 0) {
        std::cout << "af::linSolvQRDn(): cusolverDnSormqr failed, devinfo : " + std::to_string(devInfo_d[0]) << "\n";
        return false;
        // throw std::runtime_error("af::linSolvCholDn(): cusolverDnSpotrf failed, devinfo : " + std::to_string(devInfo_d[0]));
    }
    CUDA_CHECK;

    const float one = 1.f;
    cublasStrsm(handleCublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, nrhs, &one, ADense,
                lda, bDense, ldb);
    cudaDeviceSynchronize();

    return true;
}

/*
 *  solve A*x = b by LU with partial pivoting
 *
 */
bool linearSolverLU(cusolverDnHandle_t handle, int n, float* A, int lda, float* b, int* ipiv) {
    int bufferSize = 0;
    int* info      = NULL;
    float* buffer  = NULL;
    int h_info     = 0;

    cusolverDnSgetrf_bufferSize(handle, n, n, (float*)A, lda, &bufferSize);

    cudaMalloc(&info, sizeof(int));
    cudaMalloc(&buffer, sizeof(float) * bufferSize);

    cudaMemset(info, 0, sizeof(int));

    // getrf will overwrite A with L
    cusolverDnSgetrf(handle, n, n, A, lda, buffer, ipiv, info);
    cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);

    if (0 != h_info) {
        fprintf(stderr, "Error: LU factorization failed\n");
        return false;
    }

    cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, b, n, info);
    cudaDeviceSynchronize();

    if (info) {
        cudaFree(info);
    }
    if (buffer) {
        cudaFree(buffer);
    }

    return true;
}

bool linearSolverLUStable(cusolverDnHandle_t handle, int n, const float* Acopy, int lda, const float* b, int* ipiv, float* x) {
    int bufferSize = 0;
    int* info      = NULL;
    float* buffer  = NULL;
    float* A       = NULL;
    // int* ipiv      = NULL;  // pivoting sequence
    int h_info     = 0;

    cusolverDnSgetrf_bufferSize(handle, n, n, (float*)Acopy, lda, &bufferSize);

    cudaMalloc(&info, sizeof(int));
    cudaMalloc(&buffer, sizeof(float) * bufferSize);
    cudaMalloc(&A, sizeof(float) * lda * n);
    cudaMalloc(&ipiv, sizeof(int) * n);

    // prepare a copy of A because getrf will overwrite A with L
    cudaMemcpy(A, Acopy, sizeof(float) * lda * n, cudaMemcpyDeviceToDevice);
    cudaMemset(info, 0, sizeof(int));

    cusolverDnSgetrf(handle, n, n, A, lda, buffer, ipiv, info);
    cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);

    if (0 != h_info) {
        fprintf(stderr, "Error: LU factorization failed\n");
        return false;
    }

    cudaMemcpy(x, b, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info);
    cudaDeviceSynchronize();

    if (info) {
        cudaFree(info);
    }
    if (buffer) {
        cudaFree(buffer);
    }
    if (A) {
        cudaFree(A);
    }
    // if (ipiv) {
    //     cudaFree(ipiv);
    // }

    return true;
}

}  // namespace af