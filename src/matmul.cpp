#include "matmul.hpp"

#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

void Matmul::genMat(int n, int m, std::vector<float>& mat){
    srand(time(0));
    mat.resize(n * m);
    for (int i=0; i < mat.size(); i++) mat[i] = rand() % 100;
}

void Matmul::dumpMat(int n, int m, std::vector<float>& mat)
{
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<m; j++)
            printf("%f ", mat[i * m + j]);
        printf("\n");
    }
}

void Matmul::matmult(int m, int n, int k, const float* mat_a, const float* mat_b, float* mat_c){
    /*
        == input ==
        mat_a: m x k matrix
        mat_b: k x n matrix

        == output ==
        mat_c: m x n matrix (output)
    */

    for (int i1=0; i1<m; i1++) {
        for (int i2=0; i2<n; i2++) {
            mat_c [n*i1 + i2] = 0;
            for (int i3=0; i3<k; i3++) {
                mat_c[n*i1 + i2] += mat_a[i1 * k + i3] * mat_b[i3 * n + i2];
            }
        }
    }
}