#ifndef LINEARSOLVER_CUH
#define LINEARSOLVER_CUH

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>

#include <af/Helper.cuh>

namespace af {

cusolverDnHandle_t createHandleDn();

bool linSolvCholDn(cusolverDnHandle_t handleDn, float* ADense, float* bDense, const int dim);

bool linSolvLUDn(cusolverDnHandle_t handleDn, float* ADense, float* bDense, const int dim);

bool linSolvQRDn(cusolverDnHandle_t handleDn, cublasHandle_t handleCublas, float* ADense, float* bDense, const int dim);

bool linearSolverLU(cusolverDnHandle_t handle, int n, float* A, int lda, float* b, int* ipiv);

bool linearSolverLUStable(cusolverDnHandle_t handle, int n, const float* Acopy, int lda, const float* b, int* ipiv, float* x);

}  // namespace af

#endif