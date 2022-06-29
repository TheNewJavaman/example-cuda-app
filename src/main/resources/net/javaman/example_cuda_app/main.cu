// The main function, a.k.a. "kernel". You can add private helper functions using `__device__`, for example:
//     __device__ void helper() {}
extern "C" __global__ void add_vectors_kernel(
    int n,
    double* a,
    double* b,
    double* c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Which element we should process
    if (idx < n) { // Prevent processing an element that doesn't exist!
        c[idx] = a[idx] + b[idx]; // Write to the output array
    }
}