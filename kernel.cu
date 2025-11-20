#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// If not compiling with nvcc, provide minimal stubs so static analyzers don't error on CUDA builtins.
#ifndef __CUDACC__
typedef struct { int x, y, z; } dim3;
extern dim3 blockIdx;
extern dim3 blockDim;
extern dim3 threadIdx;
static inline void __syncthreads() {}
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
#endif

// Parameters
#define N 1000            // Number of bodies
#define DT 0.01f          // Time step
#define STEPS 1000        // Number of simulation steps
#define G 9.8f            // Gravitational constant (normalized)
#define SOFTENING 1e-2f   // Softening length (eps). We'll add eps^2 to distance squared.

typedef struct {
    float x, y, z;  // Position
    float vx, vy, vz;   // Velocity
    float mass;     // Mass
} Body;

// Simple CUDA error check
static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void randomizeBodies(Body* p) {
    for (int i = 0; i < N; i++) {
        // Random position between -1.0 and 1.0
        p[i].x = 2.0f * (float)rand() / RAND_MAX - 1.0f;
        p[i].y = 2.0f * (float)rand() / RAND_MAX - 1.0f;
        p[i].z = 2.0f * (float)rand() / RAND_MAX - 1.0f;

        // Random velocity (small initial push)
        p[i].vx = 0.0f;
        p[i].vy = 0.0f;
        p[i].vz = 0.0f;

        // Random mass
        p[i].mass = (float)rand() / RAND_MAX + 0.1f;
    }
}

// CUDA kernel: tiled/shared-memory version. Each thread computes force for one body i.
// Shared memory layout: [x0..xB-1][y0..yB-1][z0..zB-1][m0..mB-1]
__global__ void bodyForceKernel(Body* p, float dt, int n) {
    extern __shared__ float sh[];
    float* shx = sh;                            // blockDim.x entries
    float* shy = shx + blockDim.x;              // blockDim.x entries
    float* shz = shy + blockDim.x;              // blockDim.x entries
    float* shm = shz + blockDim.x;              // blockDim.x entries

    const float eps2 = SOFTENING * SOFTENING;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    float xi = p[i].x;
    float yi = p[i].y;
    float zi = p[i].z;
    float mi = p[i].mass;

    int numTiles = (n + blockDim.x - 1) / blockDim.x;

    for (int t = 0; t < numTiles; ++t) {
        int idx = t * blockDim.x + threadIdx.x;

        if (idx < n) {
            shx[threadIdx.x] = p[idx].x;
            shy[threadIdx.x] = p[idx].y;
            shz[threadIdx.x] = p[idx].z;
            shm[threadIdx.x] = p[idx].mass;
        } else {
            shx[threadIdx.x] = 0.0f;
            shy[threadIdx.x] = 0.0f;
            shz[threadIdx.x] = 0.0f;
            shm[threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Loop over entries in this tile
        int tileSize = blockDim.x;
        for (int k = 0; k < tileSize; ++k) {
            int j = t * blockDim.x + k;
            if (j >= n) break;
            if (j == i) continue;

            float dx = shx[k] - xi;
            float dy = shy[k] - yi;
            float dz = shz[k] - zi;
            float distSqr = dx * dx + dy * dy + dz * dz + eps2;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            float f = G * mi * shm[k] * invDist3;
            Fx += f * dx;
            Fy += f * dy;
            Fz += f * dz;
        }

        __syncthreads();
    }

    // Update velocity
    p[i].vx += (Fx / mi) * dt;
    p[i].vy += (Fy / mi) * dt;
    p[i].vz += (Fz / mi) * dt;
}

// CUDA kernel: integrate positions
__global__ void integrateKernel(Body* p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
}

int main() {
    Body* bodies = (Body*)malloc(N * sizeof(Body));
    if (!bodies) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    srand((unsigned)time(NULL));
    randomizeBodies(bodies);

    // Allocate device memory
    Body* d_bodies = NULL;
    size_t size = N * sizeof(Body);
    cudaCheck(cudaMalloc((void**)&d_bodies, size), "cudaMalloc failed");

    // Copy initial data to device
    cudaCheck(cudaMemcpy(d_bodies, bodies, size, cudaMemcpyHostToDevice), "cudaMemcpy H2D failed");

    const int BLOCK = 128;
    const int GRID = (N + BLOCK - 1) / BLOCK;
    const size_t sharedMemBytes = 4 * BLOCK * sizeof(float); // x,y,z,m

    // Simulation Loop
    for (int step = 0; step < STEPS; step++) {
        // Compute forces (tiled/shared-memory)
#ifdef __CUDACC__
        bodyForceKernel<<<GRID, BLOCK, sharedMemBytes>>>(d_bodies, DT, N);
        cudaCheck(cudaGetLastError(), "bodyForceKernel launch failed");

        // Integrate positions
        integrateKernel<<<GRID, BLOCK>>>(d_bodies, DT, N);
        cudaCheck(cudaGetLastError(), "integrateKernel launch failed");

        // Wait for kernels to finish
        cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

        // Copy back to host for printing
        cudaCheck(cudaMemcpy(bodies, d_bodies, size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H failed");
#else
        // If not compiling with nvcc, run CPU fallback (3D)
        const float eps2 = SOFTENING * SOFTENING;
        for (int i = 0; i < N; ++i) {
            float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
            float xi = bodies[i].x;
            float yi = bodies[i].y;
            float zi = bodies[i].z;
            float mi = bodies[i].mass;
            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                float dx = bodies[j].x - xi;
                float dy = bodies[j].y - yi;
                float dz = bodies[j].z - zi;
                float distSqr = dx * dx + dy * dy + dz * dz + eps2;
                float invDist = 1.0f / sqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                float f = G * mi * bodies[j].mass * invDist3;
                Fx += f * dx;
                Fy += f * dy;
                Fz += f * dz;
            }
            bodies[i].vx += (Fx / mi) * DT;
            bodies[i].vy += (Fy / mi) * DT;
            bodies[i].vz += (Fz / mi) * DT;
        }

        for (int i = 0; i < N; ++i) {
            bodies[i].x += bodies[i].vx * DT;
            bodies[i].y += bodies[i].vy * DT;
            bodies[i].z += bodies[i].vz * DT;
        }
#endif

        // Output formatted for Python (expects the line "timestep")
        printf("timestep\n");
        for (int i = 0; i < N; i++) {
            printf("%f %f %f\n", bodies[i].x, bodies[i].y, bodies[i].z);
        }
    }

    // Cleanup
    cudaCheck(cudaFree(d_bodies), "cudaFree failed");
    free(bodies);
    return 0;
}