#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <SDL3/SDL.h>

// --- CUDA Stubs ---
#ifndef __CUDACC__
typedef struct { int x, y, z; } dim3;
extern dim3 blockIdx; extern dim3 blockDim; extern dim3 threadIdx;
static inline void __syncthreads() {}
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
#endif

// --- CONSTANTS (Original Settings) ---
#define N 10000            
#define DT 0.005f         
#define G 1.0f              
#define SOFTENING 0.1f      
#define BLOCK_SIZE 256

// --- GRAPHICS CONSTANTS ---
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
#define SCALE 3.5f       

typedef struct { float x, y, z, vx, vy, vz, mass; } Body;

static inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// --- Init Galaxy ---
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
        p[i].z = ((rand() / (float)RAND_MAX) - 0.5f) * 10.0f;

        float vel = sqrt(5000.0f) / sqrt(dist);
        p[i].vx = -sin(angle) * vel;
        p[i].vy = cos(angle) * vel;
        p[i].vz = 0.0f;
        p[i].mass = 1.0f + ((rand() / (float)RAND_MAX));
    }
}

// --- Kernels (Original Euler Logic) ---
__global__ void bodyForceKernel(Body* p, float dt, int n) {
    extern __shared__ float sh[];
    float* shx = sh; float* shy = shx + blockDim.x;
    float* shz = shy + blockDim.x; float* shm = shz + blockDim.x;

    const float eps2 = SOFTENING * SOFTENING;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    float xi, yi, zi;

    if (i < n) { xi = p[i].x; yi = p[i].y; zi = p[i].z; }

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

// --- Main ---
int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("CUDA N-Body (Euler + Stats)", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);

    // Init Data
    Body* h_bodies = (Body*)malloc(N * sizeof(Body));
    srand(1337);
    initGalaxy(h_bodies);

    Body* d_bodies = NULL;
    cudaCheck(cudaMalloc((void**)&d_bodies, N * sizeof(Body)), "cudaMalloc");
    cudaCheck(cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body), cudaMemcpyHostToDevice), "H2D");

    const int GRID = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t shMemSize = 4 * BLOCK_SIZE * sizeof(float);

    int quit = 0;
    SDL_Event event;

    // Stats Variables
    Uint64 lastTime = SDL_GetTicks();
    int frames = 0;
    char titleBuffer[256];
    long totalSteps = 0;

    // --- ACCUMULATORS ---
    Uint64 startWallTime = SDL_GetTicks();
    double totalComputeMs = 0.0f;

    cudaEvent_t startEvt, stopEvt;
    cudaEventCreate(&startEvt);
    cudaEventCreate(&stopEvt);

    printf("Simulation Running (Euler Mode). N=%d\n", N);
    printf("Press ESC to see Performance Report.\n");

    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) quit = 1;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) quit = 1;
        }

        // --- Compute Step (Timed) ---
        cudaEventRecord(startEvt);

        bodyForceKernel << <GRID, BLOCK_SIZE, shMemSize >> > (d_bodies, DT, N);
        integrateKernel << <GRID, BLOCK_SIZE >> > (d_bodies, DT, N);

        cudaEventRecord(stopEvt);
        cudaEventSynchronize(stopEvt);

        float physTimeMs = 0;
        cudaEventElapsedTime(&physTimeMs, startEvt, stopEvt);
        totalComputeMs += physTimeMs; // Accumulate

        // --- Data Transfer ---
        cudaCheck(cudaMemcpy(h_bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost), "D2H Frame");

        // --- Render ---
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        for (int i = 0; i < N; ++i) {
            float sx = h_bodies[i].x * SCALE + WINDOW_WIDTH / 2.0f;
            float sy = h_bodies[i].y * SCALE + WINDOW_HEIGHT / 2.0f;
            if (sx >= 0 && sx < WINDOW_WIDTH && sy >= 0 && sy < WINDOW_HEIGHT) {
                SDL_RenderPoint(renderer, sx, sy);
            }
        }
        SDL_RenderPresent(renderer);

        // --- Title Stats ---
        frames++;
        totalSteps++;
        Uint64 currentTime = SDL_GetTicks();
        if (currentTime - lastTime >= 1000) {
            float fps = (float)frames * 1000.0f / (float)(currentTime - lastTime);
            snprintf(titleBuffer, sizeof(titleBuffer),
                "Step: %ld | GPU Phys: %.3f ms | FPS: %.1f | N: %d",
                totalSteps, physTimeMs, fps, N);
            SDL_SetWindowTitle(window, titleBuffer);
            frames = 0;
            lastTime = currentTime;
        }
    }

    // --- FINAL REPORT ---
    Uint64 endWallTime = SDL_GetTicks();
    double totalWallSecs = (double)(endWallTime - startWallTime) / 1000.0;
    double totalComputeSecs = totalComputeMs / 1000.0;

    printf("\n=== PERFORMANCE REPORT ===\n");
    printf("Total steps executed: %ld\n", totalSteps);
    printf("Total time (simulation): %.6f seconds\n", totalWallSecs);
    printf("Total compute time (GPU): %.6f seconds\n", totalComputeSecs);

    if (totalSteps > 0) {
        printf("Average total time/step: %.6f s (%.3f ms)\n",
            totalWallSecs / totalSteps, (totalWallSecs * 1000.0) / totalSteps);
        printf("Average compute time/step: %.6f s (%.3f ms)\n",
            totalComputeSecs / totalSteps, totalComputeMs / totalSteps);
    }

    printf("Overhead (Draw/Memcpy): %.6f seconds\n", totalWallSecs - totalComputeSecs);
    printf("==========================\n");

    cudaEventDestroy(startEvt);
    cudaEventDestroy(stopEvt);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    cudaFree(d_bodies);
    free(h_bodies);

    return 0;
}