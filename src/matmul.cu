#include "matmul.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include <vector>

inline __host__ __device__ int iDivUp(const int& a, const int& b)
{
    int result = a % b != 0 ? (a < b ? 1 : a / b + 1) : a / b;
    return result;
}

__global__ void matmul_kernel(float* d_a,float* d_b, float* d_c, int out_w, int out_h, int k)
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

void Matmul::matmult_cuda(int m, int k, int n, float** d_a, float** d_b, float** d_c, cudaStream_t& stream){
    assert(*d_a && *d_b && *d_c);
    const dim3 blockDim(16,16);
    const dim3 gridDim(iDivUp(m,blockDim.x),iDivUp(n,blockDim.y));

    matmul_kernel<<<gridDim,blockDim,0,stream>>>(*d_a, *d_b, *d_c, m, n, k);
    cudaGetLastError();
    //cudaDeviceSynchronize();
}

void Matmul::upload(const long long& buffersize, const float* src, float** dst){
    cudaMalloc((void**)dst, buffersize);
    cudaMemcpy(*dst, src, buffersize, cudaMemcpyHostToDevice);
}

void Matmul::download(const long long& buffersize, const float* src, float* dst){
    cudaMemcpy(dst, src, buffersize, cudaMemcpyDeviceToHost);
}

void Matmul::cudafree(float *src)
{
    cudaFree(src);
}