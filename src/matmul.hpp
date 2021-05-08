#include <iostream>
#include <time.h>
#include <stdlib.h>

#include <vector>

namespace cv{
    class Mat;
};

class Matmul{

public:
    void matmult(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c);

    void matmult_cuda(int m, int n, int k, float** d_a, float** d_b, float** d_c);

    void matmult_cuda2(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c);

    void matmult_opencv(int m, int n, int k, const cv::Mat& mat_a, const cv::Mat& mat_b, cv::Mat& mat_c);

    void matmult_opencv2(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c);

    // flag [0 : naive , 1 : cuda, 2 : opencv]
    void genMat(int n, int m, void* _mat, int flag = 0);

    void upload(const long long& buffersize, const float* src, float** dst);

    void download(const long long& buffersize, const float* src, float* dst);

    void cudafree(float* src);

    void dumpMat(int n, int m, std::vector<float>& mat);
};

class Timer {
    struct timespec s_;
public:
    Timer() { tic(); }
    void tic() {
        clock_gettime(CLOCK_REALTIME, &s_);
    }

    double toc() {
        struct timespec e;
        clock_gettime(CLOCK_REALTIME, &e);
        return (double)(e.tv_sec - s_.tv_sec) + 1e-9 * (double)(e.tv_nsec - s_.tv_nsec);
    }
};