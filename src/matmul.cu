#include "matmul.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include <vector>

#define BLOCK_SIZE 16

inline __host__ __device__ int iDivUp(const int& a, const int& b)
{
    int result = a % b != 0 ? (a < b ? 1 : a / b + 1) : a / b;
    return result;
}

inline __device__ __host__ float2 operator*(const float2 &a, const float2 &b){
    return make_float2( a.x * b.x, a.y * b.y );
}

inline __device__ __host__ float2 operator+(const float2 &a, const float2 &b){
    return make_float2( a.x + b.x, a.y + b.y );
}

//matrix_multiplication kernel 
__global__ void matmul_kernel(float* d_a,float* d_b, float* d_c, int out_h, int out_w, int k)
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

//matrix_multiplication kernel using shared_memory
__global__ void matmul_kernel_shared(float* d_a,float* d_b, float* d_c, int out_h, int out_w, int k)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    const int t_x = threadIdx.x;
    const int t_y = threadIdx.y;

    __shared__ float ds_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float ds_b[BLOCK_SIZE][BLOCK_SIZE];

    float dst = 0.0;
    const int gridStride = (BLOCK_SIZE + k - 1)/BLOCK_SIZE;

    for(int i = 0; i < gridStride; ++i)
    {
        if( i * BLOCK_SIZE + t_x < k && y < out_h )
            ds_a[t_y][t_x] = d_a[y * k + i * BLOCK_SIZE + t_x];
        else
            ds_a[t_y][t_x] = 0.0;

        if( t_y + i * BLOCK_SIZE < k && x < out_w )
            ds_b[t_y][t_x] = d_b[(t_y + i * BLOCK_SIZE) * out_w + x];
        else
            ds_b[t_y][t_x] = 0.0;

        __syncthreads();
    
        for(int i = 0; i < BLOCK_SIZE; ++i)
            dst += ds_a[t_y][i] * ds_b[i][t_x];
        __syncthreads();
    }

    if( x < out_w && y < out_h )
        d_c[y * out_w + x] = dst;
}

void Matmul::matmult_cuda(int m, int k, int n, float** d_a, float** d_b, float** d_c, cudaStream_t& stream){
    assert(*d_a && *d_b && *d_c);
    const dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE);
    const dim3 gridDim(iDivUp(n,blockDim.x),iDivUp(m,blockDim.y));

    matmul_kernel_shared<<<gridDim,blockDim,BLOCK_SIZE*BLOCK_SIZE*sizeof(float),stream>>>(*d_a, *d_b, *d_c, m, n, k);
    //matmul_kernel<<<gridDim,blockDim,0,stream>>>(*d_a, *d_b, *d_c, m, n, k);
    cudaGetLastError();
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