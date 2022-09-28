#include "cufft_example.h"

//Based on example found at http://techqa.info/programming/question/36889333/cuda-cufft-2d-example

__device__ Complex complexScaleMult(Complex a, Complex b, int scalar)
{
    //TODO Create a variable of type Complex named c
    Complex c;
    //TODO Calculate the x value for c by scalar * (a.x * b.x)
    c.x = scalar * (a.x * b.x);
    //TODO Calculate the y value for c by scalar * (a.y * b.y)
    c.y = scalar * (a.y * b.y);
    return c;
}

__global__ void complexProcess(Complex *a, Complex *b, Complex *c, int size, int scalar)
{
    // TODO calculate threadId variable
    int threadID = threadIdx.x + blockIdx.x * blockDim.x ;
    // TODO process complexScalarMult on values in a and b at index threadID and the passed scalar, place the result in c[threadId]
    c[threadID] = complexScaleMult(a[threadID], b[threadID], scalar);

}

__host__ std::tuple<int, int> parseCommandLineArguments(int argc, char** argv) 
{
    // parse command line input for argument -n and place in variable N
    int N = 16;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = 0;
            switch(arg) {
                case 'n':
                    N = (unsigned int) strtol(argv[i], 0, 10);
                    i++;
                    break;
            }
            if (toSet) {
                i++;
                *toSet = (unsigned int) strtol(argv[i], 0, 10);
            }
        }
    }
    // TODO Set variable SIZE equal to N squared
    int SIZE = N^2;

    return {N, SIZE};
}

__host__ Complex *generateComplexPointer(int SIZE)
{
    Complex *complex = new Complex[SIZE];
    // TODO populate properties x and y of variable complex at index i to 2 and 3 respectively
    for (int i = 0; i < SIZE; i++){
        complex[i].x = 2;
        complex[i].y = 3;
    }
    return complex;
}

__host__ void printComplexPointer(Complex *complex, int N)
{
    for (int i = 0; i < N * N; i = i + N)
    {
        for (int j=0; j < N; j++){
            cout << complex[i+j].x << " ";
        }
        cout << endl;
    }
    cout << "----------------" << endl;
}

__host__ cufftComplex *generateCuFFTComplexPointerFromHostComplex(int mem_size, Complex *hostComplex)
{
    // Complex *complex = new Complex(SIZE);
    // TODO populate properties x and y of variable complex at index i to 2 and 3 respectively

    cufftComplex *d_complex;
    checkCudaErrors(cudaMalloc((void **) &d_complex, mem_size)); 
    checkCudaErrors(cudaMemcpy(d_complex, hostComplex, mem_size, cudaMemcpyHostToDevice));

    return d_complex;
}

__host__ cufftHandle transformFromTimeToSignalDomain(int N, cufftComplex *d_a, cufftComplex *d_b, cufftComplex *d_c)
{
    //TODO create a cufftHandle of size N*N and from Complex input to Complex output
    cufftHandle plan;
    cufftPlan2d(&plan, N, N, CUFFT_C2C);
    //TODO execute Complex 2 Complex Forward Transformation based on the cufftHandle for d_a, d_b, d_c 
    printf("Performing Forward Transformation of a, b, and c");
    cufftExecC2C(plan, (cufftComplex *)d_a, (cufftComplex *)d_a, CUFFT_FORWARD);
    cufftExecC2C(plan, (cufftComplex *)d_b, (cufftComplex *)d_b, CUFFT_FORWARD);
    cufftExecC2C(plan, (cufftComplex *)d_c, (cufftComplex *)d_c, CUFFT_FORWARD);
    // TODO return cufftHandle for later use
    return plan;
}

__host__ Complex *transformFromSignalToTimeDomain(cufftHandle plan, int SIZE, cufftComplex *d_c)
{
    // TODO Initialize a Complex pointer with name results of size SIZE
    Complex *results = new Complex[SIZE];
    // TODO Perform Complex to Complex INVERSE transformation of cufftComplex using the passed in plan and d_c
    printf("Transforming signal back cufftExecC2C\n");
    cufftExecC2C(plan, (cufftComplex *)d_c, (cufftComplex *)d_c, CUFFT_INVERSE);

    // TODO Perform memory copy from d_c into Complex variable results
    cudaMemcpy(results, d_c, sizeof(Complex)*SIZE, cudaMemcpyDeviceToHost);
    
    return results;
}

int main(int argc, char** argv)
{
    auto[N, SIZE] = parseCommandLineArguments(argc, argv);

    Complex *a = generateComplexPointer(SIZE);
    Complex *b = generateComplexPointer(SIZE);
    Complex *c = generateComplexPointer(SIZE);

    cout << "Input random data a:" << endl;
    printComplexPointer(a, N);
    cout << "Input random data b:" << endl;
    printComplexPointer(b, N);

    int mem_size = sizeof(Complex)* SIZE;

    cufftComplex *d_a = generateCuFFTComplexPointerFromHostComplex(mem_size, a);
    cufftComplex *d_b = generateCuFFTComplexPointerFromHostComplex(mem_size, b);
    cufftComplex *d_c = generateCuFFTComplexPointerFromHostComplex(mem_size, c);

    cufftHandle plan = transformFromTimeToSignalDomain(N, d_a, d_b, d_c);

    printf("Launching Complex Division and Subtraction\n");
    int scalar = (rand() % 5) + 1;
    cout << "Scalar value: " << scalar << endl;
    complexProcess <<< N, N >> >(d_a, d_b, d_c, SIZE, scalar);

    Complex *results = transformFromSignalToTimeDomain(plan, SIZE, d_c);
    cout << "Output data c: " << endl;
    printComplexPointer(results, N);

    delete results, a, b, c;
    cufftDestroy(plan);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}