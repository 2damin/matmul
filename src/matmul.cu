#include "matmul.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

__global__ void matmul_kernel()
{

}

void Matmul::matmult_cuda(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c){

}