#include "matmul.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

inline __host__ __device__ int iDivUp(const int& a, const int& b)
{
    int result = a % b != 0 ? a / b + 1 : a / b;
    return result;
}

__global__ void matmul_kernel(float* d_a, float* d_b, float* d_c, const int& out_w, const int& out_h, const int& k)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if( x > out_w - 1 || y > out_h -1)
        return;

    float dst = 0.0f;

    for(int i = 0; i < k; ++i)
    {
        dst += d_a[out_w * y + i] * d_b[i * k + x];
    }

    d_c[y * out_w + x] = dst;

}

void Matmul::matmult_cuda(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c){

    float *d_a, *d_b, *d_c;

    const int a_buffersize = sizeof(float)*m*k;
    const int b_buffersize = sizeof(float)*n*k;
    const int c_buffersize = sizeof(float)*m*n;

    cudaMalloc((void**)&d_a, a_buffersize);
    cudaMalloc((void**)&d_b, b_buffersize);
    cudaMalloc((void**)&d_c, c_buffersize);

    cudaMemcpy(d_a, mat_a, a_buffersize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, mat_b, b_buffersize, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_c, mat_c, a_buffersize, cudaMemcpyHostToDevice);

    const dim3 blockDim(32,32);
    const dim3 gridDim(iDivUp(m,blockDim.x),iDivUp(n,blockDim.y));

    matmul_kernel<<<gridDim,blockDim>>>(d_a, d_b, d_c, m, n, k);

    cudaMemcpy(mat_c, d_c, c_buffersize, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}