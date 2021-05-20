# matmul
matrix multiplication using naive C/C++, OpenCV, CUDA

![image](https://user-images.githubusercontent.com/29620595/118991874-12c93380-b9bf-11eb-8247-825b876f04c9.png)


## Dependency

- cmake >= 3.10.0
- cuda >= 11.1
- opencv

## Build

### Linux

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..;cmake --build . --config "Release" -j;
cd ..
```

### Run
```bash
./bin/matmul.exe
```

## Result

#### Environment

- CPU : Intel Zeon E5-2643 (6core, 3.4GHz)
- GPU : NVIDIA Titan XP
- OS : ubuntu 18.04
- CUDA : 11.1
- OpenCV : 4.5.0
- C++ : 17

#### Performance (ms)

average running time when iterating 10000.

| Matrix size(N*N) | naive C/C++ | OpenCV | CUDA |
| --- | --- | ------ | ------ |
| 10 | 0.001814 |  0.00211 | 0.00251 |
| 100 | 0.829002 |  0.512281 | 0.005964 |
| 1000  | 1041.928947 | 366.799433 | 1.132146 |
