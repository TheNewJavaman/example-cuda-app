package net.javaman.example_cuda_app;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUlimit;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.nvrtc.JNvrtc;

import java.io.IOException;
import java.util.Objects;
import java.util.Random;

public class Main {
    private static final int BLOCK_SIZE = 512; // How many elements to process in one pass on the GPU
    private static final long STDOUT_BUFFER_SIZE = 4096L; // Allow console output from the GPU
    private static final int N = 100_000_000; // Number of elements being added

    public static void main(String[] args) throws IOException {
        // Show errors being thrown (if any)
        JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);

        // Init the system and get a device + context with which to access the device
        JCudaDriver.cuInit(0);
        var device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);
        var context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);

        // Load the Cuda binary you've compiled from main.cu
        var module = new CUmodule();
        var cubin = Objects.requireNonNull(Main.class.getResourceAsStream("main.cubin")).readAllBytes();
        JCudaDriver.cuModuleLoadData(module, cubin);
        var kernel = new CUfunction();
        JCudaDriver.cuModuleGetFunction(kernel, module, "add_vectors_kernel");

        // Enable console output
        JCudaDriver.cuCtxSetLimit(CUlimit.CU_LIMIT_PRINTF_FIFO_SIZE, STDOUT_BUFFER_SIZE);

        // If multithreading, make sure to switch the context to this thread!
        JCudaDriver.cuCtxSetCurrent(context);

        // Randomize vectors a and b with n elements each
        var rand = new Random();
        var a = rand.doubles(N).toArray();
        var b = rand.doubles(N).toArray();

        // Allocate memory on the GPU for these vectors, and get pointers to that memory on the device
        var aPtr = allocMem(Sizeof.DOUBLE * N, Pointer.to(a));
        var bPtr = allocMem(Sizeof.DOUBLE * N, Pointer.to(b));
        var cPtr = allocMem(Sizeof.DOUBLE * N, null);
        var kernelParams = Pointer.to(
                Pointer.to(new int[]{N}), // A little hack to pass a primitive as a parameter to the kernel
                Pointer.to(aPtr),
                Pointer.to(bPtr),
                Pointer.to(cPtr)
        );

        // Launch the kernel on the GPU and wait for it to finish
        var gridSize = (int) Math.ceil((float) N / BLOCK_SIZE);
        JCudaDriver.cuLaunchKernel(kernel,
                gridSize, 1, 1,
                BLOCK_SIZE, 1, 1,
                0, null,
                kernelParams, null
        );
        JCudaDriver.cuCtxSynchronize();

        // Get the output and free the device memory from earlier
        var c = new double[N];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(c), cPtr, Sizeof.DOUBLE * N);
        JCudaDriver.cuMemFree(aPtr);
        JCudaDriver.cuMemFree(bPtr);
        JCudaDriver.cuMemFree(cPtr);

        // Print the results! The vectors should have been added in c
        System.out.printf("First 5 (of %d) elements: %n", N);
        for (int i = 0; i < 5; i++) {
            System.out.printf("Element %d: %1.3f + %1.3f = %1.3f %n", i, a[i], b[i], c[i]);
        }
    }

    // Helper to allocate memory and get a device pointer in one go
    private static CUdeviceptr allocMem(long size, Pointer hostPtr) {
        var devicePtr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(devicePtr, size);
        if (hostPtr != null) {
            JCudaDriver.cuMemcpyHtoD(devicePtr, hostPtr, size);
        }
        return devicePtr;
    }
}
