#include<iostream>
#include<vector>
#include<cmath>

#include "matmul.hpp"

#include<cuda.h>
#include<cuda_runtime.h>

float error_check(const float *mat_a, const float *mat_b, int m, int n)
{
    float error = 0;
    uint64_t cnt = 0;
    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < m; ++i)
        {
            const int pos = j * m + i;
            error += std::fabs(mat_a[pos] - mat_b[pos]);
            ++cnt;
        }
    }
    error /= cnt;

    return error;
}

int main(int argc, char** argv)
{
    Matmul mm;
    Timer tt;

    auto mat_a = std::vector<float>();
    auto mat_b = std::vector<float>();
    auto mat_c = std::vector<float>();

    int m = 100;
    int k = 100;
    int n = 100;
    const int iteration = 10000;

    mm.genMat(m,k,&mat_a);
    mm.genMat(k,n,&mat_b);
    mm.genMat(m,n,&mat_c);

    std::cout << "-------- naive C/C++ ---------" << std::endl;
    std::cout << " " << std::endl;

    double elapsed = 0;
    for (int i=0; i<iteration; i++)
    {
        tt.tic();
        mm.matmult(m,k,n,&mat_a[0],&mat_b[0],&mat_c[0]);
        elapsed += tt.toc();
    }
    //mm.dumpMat(m,n,mat_c);

    printf("naive %lf ms\n", 1000.0 * elapsed / iteration);
    elapsed = 0;

    std::cout << "-------- opencv ---------" << std::endl;
    std::cout << " " << std::endl;

    auto mat_c_cv = std::vector<float>();
    mm.genMat(m,n,&mat_c_cv);
    for (int i=0; i<iteration; i++)
    {
        tt.tic();
        mm.matmult_opencv2(m,k,n,&mat_a[0],&mat_b[0],&mat_c_cv[0]);
        elapsed += tt.toc();
    }
    //mm.dumpMat(m,n,mat_c_cv);

    printf("opencv %lf ms\n", 1000.0 * elapsed / iteration);
    elapsed = 0;

    std::cout << "-------- cuda ---------" << std::endl;
    std::cout << " " << std::endl;

    auto mat_c_cuda = std::vector<float>(m*n,0);

    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    long long a_buffer = m * k * sizeof(float);
    long long b_buffer = k * n * sizeof(float);
    long long c_buffer = m * n * sizeof(float);
    mm.upload(a_buffer, &mat_a[0], &d_a);
    mm.upload(b_buffer, &mat_b[0], &d_b);
    mm.upload(c_buffer, &mat_c_cuda[0], &d_c);
    for(int i = 0; i < iteration; i++)
    {
        tt.tic();
        mm.matmult_cuda(m,k,n,&d_a,&d_b,&d_c);
        elapsed += tt.toc();
    }
    mm.download(c_buffer, d_c, &mat_c_cuda[0]);
    //cudaMemcpy(&mat_c_cuda[0], d_c, c_buffer, cudaMemcpyDeviceToHost);

    //mm.dumpMat(m,n,mat_c_cuda);
    printf("cuda %lf ms\n", 1000.0 * elapsed / iteration);

    mm.cudafree(d_a);
    mm.cudafree(d_b);
    mm.cudafree(d_c);

    std::cout << "---------- error check ----------" << std::endl;
    std::cout << "naive vs opencv : " << error_check(&mat_c[0], &mat_c_cv[0],m,n) << std::endl;
    std::cout << "naive vs cuda : " << error_check(&mat_c[0], &mat_c_cuda[0],m,n) << std::endl;

    return 0;
}