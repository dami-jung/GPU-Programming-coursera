#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper/helper_functions.h"
#include "helper/helper_cuda.h"

#include <ctime>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cufft.h>
#include <fstream>
#include <stdlib.h>

using namespace std;
typedef float2 Complex;

__device__ Complex complexScaleMult(Complex a, Complex b, int scalar);
__global__ void complexProcess(Complex *a, Complex *b, Complex *c, int size, int scalar);
__host__ std::tuple<int, int> parseCommandLineArguments(int argc, char** argv);
__host__ Complex *generateComplexPointer(int SIZE);
__host__ void printComplexPointer(Complex *complex, int N);
__host__ cufftComplex *generateCuFFTComplexPointerFromHostComplex(int mem_size, Complex *hostComplex);
__host__ cufftHandle transformFromTimeToSignalDomain(int N, cufftComplex *d_a, cufftComplex *d_b, cufftComplex *d_c);
