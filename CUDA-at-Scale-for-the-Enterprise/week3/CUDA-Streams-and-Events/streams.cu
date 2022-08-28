// Based on code found at https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
#include "streams.h"

// Increments all of the values in the input arrays
__global__ void kernelA1(float *dev_mem, int n, float x)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        dev_mem[i] = dev_mem[i] + x;
    }
}

//Doubles all the values in the input arrays
__global__ void kernelB1(float *dev_mem, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        dev_mem[i] = dev_mem[i] * 2;
    }
}

// Decrements all of the values in the input arrays
__global__ void kernelA2(float *dev_mem, int n, float x)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        dev_mem[i] = dev_mem[i] - x;
    }
}

//Halves all the values in the input arrays
__global__ void kernelB2(float *dev_mem, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        dev_mem[i] = dev_mem[i] / 2;
    }
}

// This will generate an array of size numElements of random integers from 0 to 255 in pageable host memory
// The host memory has to be page-locked memory or control of streams is not guaranteed
// Note that I have added an argument for the random seed, so that you can generate the same "random" values
// for multiple runs to see the result of different actions on the same set of "random" values
__host__ float *allocateHostMemory(int numElements, int seed)
{
    seed = seed != -1 ? seed : 0;
    srand(seed);
    size_t size = numElements * sizeof(float);
    float random_max = 255.0f;

    // Allocate the host pinned memory input pointer B
    float *data;
    cudaHostAlloc((void**)&data, size, cudaHostAllocDefault);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        // Feel free to change the max value of the random input data by replacing 255 with a smaller or larger number
        data[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/random_max));
    }

    return data;
}

__host__ float * allocateDeviceMemory(int numElements)
{
    // Allocate the device input vector a
    float *dev_mem = NULL;
    size_t size = numElements * sizeof(float);
    cudaError_t err = cudaMalloc(&dev_mem, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return dev_mem;
}

//Synchronous copy of data from host to device using a default stream
__host__ void copyFromHostToDeviceSync(float *host_mem, float *dev_mem, int numElements)
{
    size_t size = numElements * sizeof(float);
    // Copy the host input vector to the device input vectors
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaError_t err = cudaMemcpy(dev_mem, host_mem, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//Asynchronous copy of data from host to device using a non-default stream
__host__ void copyFromHostToDeviceAsync(float *host_mem, float *dev_mem, int numElements, cudaStream_t stream)
{
    size_t size = numElements * sizeof(float);
    // Copy the host input vector to the device input vectors
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaError_t err = cudaMemcpyAsync(dev_mem, host_mem, size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//Synchronous copy of data from device to host using the default stream
__host__ void copyFromDeviceToHostSync(float *dev_mem, float *host_mem, int numElements)
{
    size_t size = numElements * sizeof(float);
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaError_t err = cudaMemcpy(host_mem, dev_mem, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

//Synchronous copy of data from device to host using a non-default stream
__host__ void copyFromDeviceToHostAsync(float *dev_mem, float *host_mem, int numElements, cudaStream_t stream)
{
    size_t size = numElements * sizeof(float);
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaError_t err = cudaMemcpyAsync(host_mem, dev_mem, size, cudaMemcpyDeviceToHost, stream);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Free device global memory
__host__ void deallocateDevMemory(float *dev_mem)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaFree(dev_mem);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ std::tuple<int, int> determineThreadBlockDimensions(int num_elements)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    return {threadsPerBlock, blocksPerGrid};
}

__host__ float * runStreamsFullAsync(float *host_mem, int num_elements)
{
    // TODO you will need to update this to allow for user input to manage how kernels are executed 
    // and will probably include extra copies to/from host/device memory. The user will input
    // an integer between 0 and 255 as the seed for the random number generator. Be creative there is
    // no wrong answer as long as all kernels are run. Output all random number sequences as a comma-separated
    // list in one line and then the results also as a CSV in one line. The file should have at least 3 pairs of lines for
    // a minimum of 6 lines of results.
    
    // Prepare all streams such that all kernels and memory copies execute asynchronously
    cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6;
    cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream3,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream4,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream5,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream6,cudaStreamNonBlocking);

    // Prepare device memory based on host memory
    float *dev_mem = allocateDeviceMemory(num_elements);
    copyFromHostToDeviceAsync(host_mem, dev_mem, num_elements, stream1);
    
    // Execute 4 kernels asynchronously on independent streams
    // auto[threadsPerBlock, blocksPerGrid] = determineThreadBlockDimensions(num_elements);
    // kernelA1<<<blocksPerGrid,threadsPerBlock,0, stream2>>>(dev_mem, num_elements, s);
    // kernelB1<<<blocksPerGrid,threadsPerBlock,0, stream3>>>(dev_mem, num_elements);
    // kernelA2<<<blocksPerGrid,threadsPerBlock,0, stream4>>>(dev_mem, num_elements, s);
    // kernelB2<<<blocksPerGrid,threadsPerBlock,0, stream5>>>(dev_mem, num_elements);

    // Before A1 kernel ask for user input as s(0-255), which will be the seed for the random generatsor that generates the 
    // variable x which is an argument to A1. Like below do not allow A1 to run until this input is given

    int s = 0;
    srand(s);
    float random_max = 255.0f;
    float x = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/random_max));

    // After B1 runs ask for user input (0-255) and use the input value as the seed, s, to the random number generator. 
    // You will generate the random float variable x, which is the 3rd argument to the A2 kernel.
    // You will want to hold A2 from running until this value is input, either by how you set up streams or by using events
    srand(s);
    float x2 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/random_max));

    auto[threadsPerBlock, blocksPerGrid] = determineThreadBlockDimensions(num_elements);
    kernelA1<<<blocksPerGrid,threadsPerBlock,0, stream2>>>(dev_mem, num_elements, x);
    kernelB1<<<blocksPerGrid,threadsPerBlock,0, stream3>>>(dev_mem, num_elements);
    kernelA2<<<blocksPerGrid,threadsPerBlock,0, stream4>>>(dev_mem, num_elements, x2);
    kernelB2<<<blocksPerGrid,threadsPerBlock,0, stream5>>>(dev_mem, num_elements);

    // Copy device memory back to host asynchronously
    copyFromDeviceToHostAsync(dev_mem, host_mem, num_elements, stream6);
    deallocateDevMemory(dev_mem);

    // Wait for all streams to be completed
    // This might act differently on multiple GPU system without assigning devices
    cudaDeviceSynchronize();

    return host_mem;
}

__host__ float * runStreamsBlockingKernel2StreamsNaive(float *host_mem, int num_elements)
{
    // TODO you will need to update this to allow for user input to manage how kernels are executed 
    // and will probably include extra copies to/from host/device memory. The user will input
    // an integer between 0 and 255 as the seed for the random number generator. Be creative there is
    // no wrong answer as long as all kernels are run. Output all random number sequences as a comma-separated
    // list in one line and then the results also as a CSV in one line. The file should have at least 3 pairs of lines for
    // a minimum of 6 lines of results.

    // Prepare all streams such that all kernels and memory copies execute asynchronously
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream3,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream4,cudaStreamNonBlocking);

    // Prepare device memory based on host memory
    float *dev_mem = allocateDeviceMemory(num_elements);
    copyFromHostToDeviceAsync(host_mem, dev_mem, num_elements, stream1);
    
    // Execute 2 pairs of kernels asynchronous with resepect to their streams
    // The order of execution can have an effect on the blocking behaviours
    // auto[threadsPerBlock, blocksPerGrid] = determineThreadBlockDimensions(num_elements);
    // kernelA1<<<blocksPerGrid,threadsPerBlock,0, stream2>>>(dev_mem, num_elements);
    // kernelA2<<<blocksPerGrid,threadsPerBlock,0, stream3>>>(dev_mem, num_elements);
    // kernelB1<<<blocksPerGrid,threadsPerBlock,0, stream2>>>(dev_mem, num_elements);
    // kernelB2<<<blocksPerGrid,threadsPerBlock,0, stream3>>>(dev_mem, num_elements);

    // Before A1 kernel ask for user input as s(0-255), which will be the seed for the random generatsor that generates the 
    // variable x which is an argument to A1. Like below do not allow A1 to run until this input is given

    int s = 0;
    srand(s);
    float random_max = 255.0f;
    float x = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/random_max));

    // After B1 runs ask for user input (0-255) and use the input value as the seed, s, to the random number generator. 
    // You will generate the random float variable x, which is the 3rd argument to the A2 kernel.
    // You will want to hold A2 from running until this value is input, either by how you set up streams or by using events
    srand(s);
    float x2 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/random_max));

    auto[threadsPerBlock, blocksPerGrid] = determineThreadBlockDimensions(num_elements);
    kernelA1<<<blocksPerGrid,threadsPerBlock,0, stream2>>>(dev_mem, num_elements, x);
    kernelA2<<<blocksPerGrid,threadsPerBlock,0, stream3>>>(dev_mem, num_elements, x2);
    kernelB1<<<blocksPerGrid,threadsPerBlock,0, stream2>>>(dev_mem, num_elements);
    kernelB2<<<blocksPerGrid,threadsPerBlock,0, stream3>>>(dev_mem, num_elements);

    // Copy device memory back to host asynchronously
    copyFromDeviceToHostAsync(dev_mem, host_mem, num_elements, stream4);
    deallocateDevMemory(dev_mem);

    // Wait for all streams to be completed
    // This might act differently on multiple GPU system without assigning devices
    cudaDeviceSynchronize();

    return host_mem;
}

__host__ float * runStreamsBlockingKernel2StreamsOptimal(float *host_mem, int num_elements)
{
    // TODO you will need to update this to allow for user input to manage how kernels are executed 
    // and will probably include extra copies to/from host/device memory. The user will input
    // an integer between 0 and 255 as the seed for the random number generator. Be creative there is
    // no wrong answer as long as all kernels are run. Output all random number sequences as a comma-separated
    // list in one line and then the results also as a CSV in one line. The file should have at least 3 pairs of lines for
    // a minimum of 6 lines of results.

   // Prepare all streams such that all kernels and memory copies execute asynchronously
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream3,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream4,cudaStreamNonBlocking);

    // Prepare device memory based on host memory
    float *dev_mem = allocateDeviceMemory(num_elements);
    copyFromHostToDeviceAsync(host_mem, dev_mem, num_elements, stream1);
    
    // Execute 2 pairs of kernels asynchronous with resepect to their streams
    // The order of execution can have an effect on the blocking behaviours
    auto[threadsPerBlock, blocksPerGrid] = determineThreadBlockDimensions(num_elements);
    // kernelA1<<<blocksPerGrid,threadsPerBlock,0, stream2>>>(dev_mem, num_elements);
    // kernelA2<<<blocksPerGrid,threadsPerBlock,0, stream3>>>(dev_mem, num_elements);
    // kernelB1<<<blocksPerGrid,threadsPerBlock,0, stream2>>>(dev_mem, num_elements);
    // kernelB2<<<blocksPerGrid,threadsPerBlock,0, stream3>>>(dev_mem, num_elements);
    
    // Before A1 kernel ask for user input as s(0-255), which will be the seed for the random generatsor that generates the 
    // variable x which is an argument to A1. Like below do not allow A1 to run until this input is given

    int s = 0;
    srand(s);
    float random_max = 255.0f;
    float x = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/random_max));

    // After B1 runs ask for user input (0-255) and use the input value as the seed, s, to the random number generator. 
    // You will generate the random float variable x, which is the 3rd argument to the A2 kernel.
    // You will want to hold A2 from running until this value is input, either by how you set up streams or by using events
    srand(s);
    float x2 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/random_max));

    kernelA1<<<blocksPerGrid,threadsPerBlock,0, stream2>>>(dev_mem, num_elements, x);
    kernelA2<<<blocksPerGrid,threadsPerBlock,0, stream3>>>(dev_mem, num_elements, x2);
    kernelB1<<<blocksPerGrid,threadsPerBlock,0, stream2>>>(dev_mem, num_elements);
    kernelB2<<<blocksPerGrid,threadsPerBlock,0, stream3>>>(dev_mem, num_elements);

    // Copy device memory back to host asynchronously
    copyFromDeviceToHostAsync(dev_mem, host_mem, num_elements, stream4);
    deallocateDevMemory(dev_mem);

    // Wait for all streams to be completed
    // This might act differently on multiple GPU system without assigning devices
    cudaDeviceSynchronize();

    return host_mem;
}   

__host__ void printHostMemory(float *host_mem, int num_elments)
{
    // Output results
    printf("Host memory: ");
    for(int i = 0; i < num_elments; i++)
    {
        printf("%.6f ",host_mem[i]);
    }
    printf("\n");
}

int main()
{
    int num_elements = 255; // Can be altered but keep it less than 1/2 the memory size of global memory for full concurrency
    int rand_seed = 0; // You can set this to different values for each run but default will be the same to see the effect on data

    float * host_mem = allocateHostMemory(num_elements, rand_seed);
    printHostMemory(host_mem, num_elements);
    host_mem = runStreamsFullAsync(host_mem, num_elements);
    printHostMemory(host_mem, num_elements);

    host_mem = allocateHostMemory(num_elements, 0);
    printHostMemory(host_mem, num_elements);
    host_mem = runStreamsBlockingKernel2StreamsNaive(host_mem, num_elements);
    printHostMemory(host_mem, num_elements);

    host_mem = allocateHostMemory(num_elements, 0);
    printHostMemory(host_mem, num_elements);
    host_mem = runStreamsBlockingKernel2StreamsOptimal(host_mem, num_elements);
    printHostMemory(host_mem, num_elements);

    return 0;
}