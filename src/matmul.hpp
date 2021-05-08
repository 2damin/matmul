#include <iostream>
#include <time.h>
#include <stdlib.h>

#include <vector>

class Matmul{

public:
    void matmult(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c);

    void genMat(int n, int m, std::vector<float>& mat);

    void dumpMat(int n, int m, std::vector<float>& mat);

    void matmult_cuda(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c);
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