package net.javaman;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.nvrtc.JNvrtc;

import java.io.IOException;
import java.util.Objects;
import java.util.Random;

public class Main {
    private static final int BLOCK_SIZE = 256;
    private static final long STDOUT_BUFFER_SIZE = 4096L;

    public static void main(String[] args) throws IOException {
        JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);

        JCudaDriver.cuInit(0);
        var device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);
        var context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        var cubin = Objects.requireNonNull(Main.class.getResourceAsStream("main.cubin")).readAllBytes();
        JCudaDriver.cuModuleLoadData(module, cubin);
        var kernel = new CUfunction();
        JCudaDriver.cuModuleGetFunction(kernel, module, "mainKernel");

        JCudaDriver.cuCtxSetLimit(CUlimit.CU_LIMIT_PRINTF_FIFO_SIZE, STDOUT_BUFFER_SIZE);

        JCudaDriver.cuCtxSetCurrent(context);

        final var n = 1000;
        var a = new float[n];
        var b = new float[n];
        var c = new float[n];
        var rand = new Random();
        for (int i = 0; i < n; i++) {
            a[i] = rand.nextFloat();
            b[i] = rand.nextFloat();
        }

        var aPtr = allocMem(Sizeof.FLOAT * n, Pointer.to(a));
        var bPtr = allocMem(Sizeof.FLOAT * n, Pointer.to(b));
        var cPtr = allocMem(Sizeof.FLOAT * n, Pointer.to(c));
        var kernelParams = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(aPtr),
                Pointer.to(bPtr),
                Pointer.to(cPtr)
        );

        var gridSize = (int) Math.ceil((float) n / BLOCK_SIZE);
        JCudaDriver.cuLaunchKernel(kernel,
                gridSize, 1, 1,
                BLOCK_SIZE, 1, 1,
                0, null,
                kernelParams, null
        );
        JCudaDriver.cuCtxSynchronize();

        JCudaDriver.cuMemcpyDtoH(Pointer.to(c), cPtr, Sizeof.FLOAT * n);

        JCudaDriver.cuMemFree(aPtr);
        JCudaDriver.cuMemFree(bPtr);
        JCudaDriver.cuMemFree(cPtr);

        for (int i = 0 ; i < 5; i++) {
            System.out.printf("Element %d: %1.3f + %1.3f = %1.3f %n", i, a[i], b[i], c[i]);
        }
    }

    private static CUdeviceptr allocMem(long size, Pointer hostPtr) {
        var devicePtr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(devicePtr, size);
        if (hostPtr != null) {
            JCudaDriver.cuMemcpyHtoD(devicePtr, hostPtr, size);
        }
        return devicePtr;
    }
}
