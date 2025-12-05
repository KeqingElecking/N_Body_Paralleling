#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// --- CUDA Stubs ---
#ifndef __CUDACC__
typedef struct { int x, y, z; } dim3;
extern dim3 blockIdx; extern dim3 blockDim; extern dim3 threadIdx;
static inline void __syncthreads() {}
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
#endif

// --- Simulation Constants ---
#define N 4096            
#define DT 0.005f         
#define STEPS 1000         
#define G 1.0f            
#define SOFTENING 0.1f    

typedef struct { float x, y, z, vx, vy, vz, mass; } Body;
typedef struct { float x, y, z, w; } float4_out;

static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// --- Init Galaxy (Giữ nguyên) ---
void initGalaxy(Body* p) {
    for (int i = 0; i < N; ++i) {
        if (i == 0) {
            p[i].x = 0; p[i].y = 0; p[i].z = 0;
            p[i].vx = 0; p[i].vy = 0; p[i].vz = 0;
            p[i].mass = 5000.0f;
            continue;
        }
        float angle = (i * 0.1f);
        float dist = 100.0f + (i * 10.0f / N);
        p[i].x = cos(angle) * dist;
        p[i].y = sin(angle) * dist;
        p[i].z = ((rand() / (float)RAND_MAX) - 0.5f) * 1.0f;
        float vel = sqrt(5000.0f) / sqrt(dist);
        p[i].vx = -sin(angle) * vel;
        p[i].vy = cos(angle) * vel;
        p[i].vz = 0.0f;
        p[i].mass = 1.0f + ((rand() / (float)RAND_MAX));
    }
}

// --- Kernels (Giữ nguyên logic) ---
__global__ void bodyForceKernel(Body* p, float dt, int n) {
    extern __shared__ float sh[];
    float* shx = sh; float* shy = shx + blockDim.x;
    float* shz = shy + blockDim.x; float* shm = shz + blockDim.x;

    const float eps2 = SOFTENING * SOFTENING;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    float xi, yi, zi, mi;

    if (i < n) { xi = p[i].x; yi = p[i].y; zi = p[i].z; mi = p[i].mass; }

    int numTiles = (n + blockDim.x - 1) / blockDim.x;

    for (int t = 0; t < numTiles; ++t) {
        int idx = t * blockDim.x + threadIdx.x;
        if (idx < n) {
            shx[threadIdx.x] = p[idx].x; shy[threadIdx.x] = p[idx].y;
            shz[threadIdx.x] = p[idx].z; shm[threadIdx.x] = p[idx].mass;
        }
        else {
            shx[threadIdx.x] = 0; shy[threadIdx.x] = 0; shz[threadIdx.x] = 0; shm[threadIdx.x] = 0;
        }
        __syncthreads();

        if (i < n) {
#pragma unroll 8
            for (int k = 0; k < blockDim.x; ++k) {
                float dx = shx[k] - xi; float dy = shy[k] - yi; float dz = shz[k] - zi;
                float distSqr = dx * dx + dy * dy + dz * dz + eps2;
                float invDist = rsqrtf(distSqr);
                float f = shm[k] * (invDist * invDist * invDist);
                Fx += f * dx; Fy += f * dy; Fz += f * dz;
            }
        }
        __syncthreads();
    }
    if (i < n) {
        p[i].vx += (Fx * G) * dt;
        p[i].vy += (Fy * G) * dt;
        p[i].vz += (Fz * G) * dt;
    }
}

__global__ void integrateKernel(Body* p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
}

int main() {
    Body* bodies = (Body*)malloc(N * sizeof(Body));
    float4_out* output_buffer = (float4_out*)malloc(N * sizeof(float4_out));

    srand(1337);
    initGalaxy(bodies);

    Body* d_bodies = NULL;
    cudaCheck(cudaMalloc((void**)&d_bodies, N * sizeof(Body)), "cudaMalloc");
    cudaCheck(cudaMemcpy(d_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice), "H2D");

    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;
    const size_t shMemSize = 4 * BLOCK * sizeof(float);

    FILE* fp = fopen("particles.bin", "wb");
    if (!fp) { perror("fopen"); return 1; }
    int n_out = N; int steps_out = STEPS;
    fwrite(&n_out, sizeof(int), 1, fp); fwrite(&steps_out, sizeof(int), 1, fp);

    printf("Running Simulation N=%d, Steps=%d\n", N, STEPS);
    printf("Block Size: %d, Grid Size: %d, Total Threads: %d\n", BLOCK, GRID, GRID * BLOCK);
    printf("------------------------------------------------\n");

    // --- VARIABLES FOR TIMING ---
    cudaEvent_t startCompute, stopCompute, startTransfer, stopTransfer;
    cudaEventCreate(&startCompute); cudaEventCreate(&stopCompute);
    cudaEventCreate(&startTransfer); cudaEventCreate(&stopTransfer);

    float totalKernelMs = 0.0f;
    float totalTransferMs = 0.0f;
    float milliseconds = 0.0f;

    clock_t wallStart = clock(); // Đo tổng thời gian (bao gồm cả I/O)

    for (int step = 0; step < STEPS; step++) {

        // 1. Đo thời gian tính toán GPU (Kernels)
        cudaEventRecord(startCompute);
        bodyForceKernel << <GRID, BLOCK, shMemSize >> > (d_bodies, DT, N);
        integrateKernel << <GRID, BLOCK >> > (d_bodies, DT, N);
        cudaEventRecord(stopCompute);

        // Chờ GPU tính xong để lấy thời gian chính xác (Chỉ làm khi benchmark, thực tế có thể bỏ để chạy nhanh hơn)
        cudaEventSynchronize(stopCompute);
        cudaEventElapsedTime(&milliseconds, startCompute, stopCompute);
        totalKernelMs += milliseconds;

        // 2. Đo thời gian copy dữ liệu (Device to Host)
        cudaEventRecord(startTransfer);
        cudaCheck(cudaMemcpy(bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost), "D2H");
        cudaEventRecord(stopTransfer);

        cudaEventSynchronize(stopTransfer);
        cudaEventElapsedTime(&milliseconds, startTransfer, stopTransfer);
        totalTransferMs += milliseconds;

        // 3. Ghi file (CPU - Không tính vào thời gian GPU)
        for (int i = 0; i < N; ++i) {
            output_buffer[i].x = bodies[i].x;
            output_buffer[i].y = bodies[i].y;
            output_buffer[i].z = bodies[i].z;
            output_buffer[i].w = bodies[i].mass;
        }
        fwrite(output_buffer, sizeof(float4_out), N, fp);

        if (step % 50 == 0) { printf("\rStep %d/%d", step, STEPS); fflush(stdout); }
    }

    clock_t wallEnd = clock();
    double wallSecs = (double)(wallEnd - wallStart) / CLOCKS_PER_SEC;

    // --- REPORTING ---
    printf("\n\n=== PERFORMANCE REPORT ===\n");
    printf("1. Total Particles (N):       %d\n", N);
    printf("2. Total GPU Threads:         %d (Utilization: %.2f%%)\n", GRID * BLOCK, (float)N / (GRID * BLOCK) * 100.0f);
    printf("3. Wall Clock Time (Total):   %.4f s (Includes File I/O)\n", wallSecs);
    printf("\n--- GPU TIMINGS ---\n");
    printf("4. Total Compute Time:        %.4f ms (%.4f s)\n", totalKernelMs, totalKernelMs / 1000.0f);
    printf("5. Total Memory Transfer:     %.4f ms (%.4f s)\n", totalTransferMs, totalTransferMs / 1000.0f);

    float avgStepTime = totalKernelMs / STEPS;
    printf("6. Avg Compute Time/Step:     %.4f ms\n", avgStepTime);

    // Thời gian trung bình để 1 thread (đại diện 1 hạt) hoàn thành 1 bước tính toán
    // Đây là thời gian thực tế GPU cần để xử lý 1 hạt trong 1 frame
    printf("7. Avg Time per Thread/Step:  %.6f ms (%.2f ns)\n", avgStepTime / N, (avgStepTime / N) * 1e6);

    // Metrics nâng cao
    double interactionsPerStep = (double)N * N;
    double gflops = (interactionsPerStep * 20.0 * STEPS) / (totalKernelMs / 1000.0) / 1e9;
    // *20 vì khoảng 20 phép tính float trong vòng lặp force
    printf("\n--- THROUGHPUT ---\n");
    printf("8. Estimated GFLOPS:          %.2f GFLOPS\n", gflops);
    printf("==========================\n");

    fclose(fp);
    cudaFree(d_bodies);
    free(bodies);
    free(output_buffer);
    cudaEventDestroy(startCompute); cudaEventDestroy(stopCompute);
    cudaEventDestroy(startTransfer); cudaEventDestroy(stopTransfer);
    return 0;
}