extern "C" __global__ void mainKernel(
    int n,
    float* a,
    float* b,
    float* c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}