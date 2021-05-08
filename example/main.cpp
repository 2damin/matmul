#include<iostream>
#include<vector>

#include "matmul.hpp"


int main(int argc, char** argv)
{
    Matmul mm;
    Timer tt;

    auto mat_a = std::vector<float>();
    auto mat_b = std::vector<float>();
    auto mat_c = std::vector<float>();

    mm.genMat(10,10,&mat_a);
    mm.genMat(10,10,&mat_b);
    mm.genMat(10,10,&mat_c);

    double elapsed=0;
    const int iteration = 10000;
    for (int i=0; i<iteration; i++)
    {
        tt.tic();
        mm.matmult(10,10,10,&mat_a[0],&mat_b[0],&mat_c[0]);
        elapsed += tt.toc();
    }

    //mm.dumpMat(10,10,mat_a);
    std::cout << " " << std::endl;
    //mm.dumpMat(10,10,mat_b);
    std::cout << " " << std::endl;
    mm.dumpMat(10,10,mat_c);

    printf("naive %lf ms\n", 1000.0 * elapsed / iteration);

    //mm.genMat(10,10,&mat_c);
    mat_c.resize(10,10);
    for (int i=0; i<iteration; i++)
    {
        tt.tic();
        mm.matmult_opencv2(10,10,10,&mat_a[0],&mat_b[0],&mat_c[0]);
        elapsed += tt.toc();
    }

    //mm.dumpMat(10,10,mat_a);
    std::cout << " " << std::endl;
    //mm.dumpMat(10,10,mat_b);
    std::cout << " " << std::endl;
    mm.dumpMat(10,10,mat_c);

    printf("opencv %lf ms\n", 1000.0 * elapsed / iteration);

    //mm.genMat(10,10,&mat_c);
    mat_c.resize(10,10);
    for(int i = 0; i < iteration; i++)
    {
        tt.tic();
        mm.matmult_cuda(10,10,10,&mat_a[0],&mat_b[0],&mat_c[0]);
        elapsed += tt.toc();
    }
    //mm.dumpMat(10,10,mat_a);
    std::cout << " " << std::endl;
    //mm.dumpMat(10,10,mat_b);
    std::cout << " " << std::endl;
    mm.dumpMat(10,10,mat_c);

    printf("cuda %lf ms\n", 1000.0 * elapsed / iteration);

    return 0;
}