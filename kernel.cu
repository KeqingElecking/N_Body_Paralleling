#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// --- CUDA Stubs for Editor/IntelliSense ---
#ifndef __CUDACC__
typedef struct { int x, y, z; } dim3;
extern dim3 blockIdx;
extern dim3 blockDim;
extern dim3 threadIdx;
static inline void __syncthreads() {}
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
#endif

// --- Simulation Constants ---
#define N 4096            // Số lượng hạt (Tăng lên vì N^2 kernel chịu được tốt)
#define DT 0.005f         // Time step
#define STEPS 500         // Số bước mô phỏng
#define G 1.0f            // Hằng số hấp dẫn chuẩn hóa
#define SOFTENING 0.1f    // Làm mềm để tránh chia cho 0

// Cấu trúc dữ liệu cho Kernel N^2
typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float mass;
} Body;

// Cấu trúc dùng để xuất file (tương thích với Python script cũ)
typedef struct {
    float x, y, z, w; // w = mass
} float4_out;

static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// --- Galaxy Initialization ---
void initGalaxy(Body* p) {
    // Tạo cấu trúc xoắn ốc quanh tâm
    for (int i = 0; i < N; ++i) {
        // Hạt tâm (Hố đen)
        if (i == 0) {
            p[i].x = 0; p[i].y = 0; p[i].z = 0;
            p[i].vx = 0; p[i].vy = 0; p[i].vz = 0;
            p[i].mass = 5000.0f; // Rất nặng
            continue;
        }

        float angle = (i * 0.1f);
        // Bán kính tăng dần để tạo đĩa
        float dist = 2.0f + (i * 10.0f / N);

        p[i].x = cos(angle) * dist;
        p[i].y = sin(angle) * dist;
        p[i].z = ((rand() / (float)RAND_MAX) - 0.5f) * 1.0f; // Độ dày đĩa

        // Vận tốc quỹ đạo: v = sqrt(G * M_center / r)
        // M_center = 5000, G = 1
        float vel = sqrt(5000.0f) / sqrt(dist);

        // Vận tốc tiếp tuyến (-sin, cos)
        p[i].vx = -sin(angle) * vel;
        p[i].vy = cos(angle) * vel;
        p[i].vz = 0.0f;

        p[i].mass = 1.0f + ((rand() / (float)RAND_MAX));
    }
}

// --- CUDA Kernel: O(N^2) Tiled Shared Memory ---
__global__ void bodyForceKernel(Body* p, float dt, int n) {
    extern __shared__ float sh[]; // Dynamic shared mem
    float* shx = sh;
    float* shy = shx + blockDim.x;
    float* shz = shy + blockDim.x;
    float* shm = shz + blockDim.x;

    const float eps2 = SOFTENING * SOFTENING;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    float xi, yi, zi, mi;

    // Load body i data into registers
    if (i < n) {
        xi = p[i].x; yi = p[i].y; zi = p[i].z; mi = p[i].mass;
    }

    int numTiles = (n + blockDim.x - 1) / blockDim.x;

    for (int t = 0; t < numTiles; ++t) {
        int idx = t * blockDim.x + threadIdx.x;

        // Load tile into shared memory
        if (idx < n) {
            shx[threadIdx.x] = p[idx].x;
            shy[threadIdx.x] = p[idx].y;
            shz[threadIdx.x] = p[idx].z;
            shm[threadIdx.x] = p[idx].mass;
        }
        else {
            shx[threadIdx.x] = 0; shy[threadIdx.x] = 0; shz[threadIdx.x] = 0; shm[threadIdx.x] = 0;
        }
        __syncthreads();

        if (i < n) {
#pragma unroll 8
            for (int k = 0; k < blockDim.x; ++k) {
                // Tính toán lực với tất cả các hạt trong tile (bao gồm cả chính nó vì eps > 0 sẽ xử lý việc chia 0)
                float dx = shx[k] - xi;
                float dy = shy[k] - yi;
                float dz = shz[k] - zi;
                float distSqr = dx * dx + dy * dy + dz * dz + eps2;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;

                float f = shm[k] * invDist3; // G=1 implicitly here or multiply by G later
                Fx += f * dx;
                Fy += f * dy;
                Fz += f * dz;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        p[i].vx += (Fx * G) * dt; // F/m * m_other -> mi triệt tiêu khi tính a = F/mi
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

    // Setup GPU
    Body* d_bodies = NULL;
    cudaCheck(cudaMalloc((void**)&d_bodies, N * sizeof(Body)), "cudaMalloc");
    cudaCheck(cudaMemcpy(d_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice), "H2D");

    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;
    const size_t shMemSize = 4 * BLOCK * sizeof(float);

    // Open Binary Output
    FILE* fp = fopen("particles.bin", "wb");
    if (!fp) { perror("fopen"); return 1; }

    int n_out = N;
    int steps_out = STEPS;
    fwrite(&n_out, sizeof(int), 1, fp);
    fwrite(&steps_out, sizeof(int), 1, fp);

    printf("Running N-Body simulation (O(N^2))...\n");
    printf("N=%d, Steps=%d, Output=particles.bin\n", N, STEPS);

    clock_t start = clock();

    for (int step = 0; step < STEPS; step++) {
        // 1. Compute Forces & Update Velocity
        bodyForceKernel << <GRID, BLOCK, shMemSize >> > (d_bodies, DT, N);

        // 2. Integrate Position
        integrateKernel << <GRID, BLOCK >> > (d_bodies, DT, N);

        // 3. Copy back for IO (CPU waits for GPU here)
        cudaCheck(cudaMemcpy(bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost), "D2H");

        // 4. Pack data to float4 format (x,y,z,mass) for Python
        for (int i = 0; i < N; ++i) {
            output_buffer[i].x = bodies[i].x;
            output_buffer[i].y = bodies[i].y;
            output_buffer[i].z = bodies[i].z;
            output_buffer[i].w = bodies[i].mass;
        }

        // 5. Write frame
        fwrite(output_buffer, sizeof(float4_out), N, fp);

        if (step % 10 == 0) { printf("\rStep %d/%d", step, STEPS); fflush(stdout); }
    }

    clock_t end = clock();
    double secs = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nDone. Total time: %.2fs (%.2f ms/step)\n", secs, (secs * 1000) / STEPS);

    fclose(fp);
    cudaFree(d_bodies);
    free(bodies);
    free(output_buffer);
    return 0;
}