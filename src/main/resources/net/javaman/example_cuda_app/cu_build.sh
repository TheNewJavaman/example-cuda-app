# Make sure to replace `86` with your GPU's architecture! https://en.wikipedia.org/wiki/CUDA#GPUs_supported
nvcc "main.cu" --cubin -o "main.cubin" -lineinfo -gencode arch=compute_86,code=sm_86