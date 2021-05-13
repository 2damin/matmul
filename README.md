# matmul
matrix multiplication using naive C/C++, OpenCV, CUDA

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
