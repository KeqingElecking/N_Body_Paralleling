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

// --- CONSTANTS ---
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
void initGalaxy(Body* p, int n) {
    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            p[i].x = 0; p[i].y = 0; p[i].z = 0;
            p[i].vx = 0; p[i].vy = 0; p[i].vz = 0;
            p[i].mass = 5000.0f;
            continue;
        }
        float angle = (i * 0.1f);
        float dist = 100.0f + (i * 10.0f / n);
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

// --- Kernels ---
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
        } else {
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
        p[i].vx += (Fx * G) * dt; p[i].vy += (Fy * G) * dt; p[i].vz += (Fz * G) * dt;
    }
}

__global__ void integrateKernel(Body* p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].x += p[i].vx * dt; p[i].y += p[i].vy * dt; p[i].z += p[i].vz * dt;
}

// --- Main ---
int main(int argc, char* argv[]) {
    int N;
    long targetSteps;
    int choiceGraphics;

    printf("=== NHAP THONG SO BENCHMARK ===\n");
    printf("Nhap so luong hat (N): ");
    scanf("%d", &N);
    printf("Nhap so buoc (time steps) muon chay: ");
    scanf("%ld", &targetSteps);
    printf("Co hien thi do hoa khong? (1: Co, 0: Khong): ");
    scanf("%d", &choiceGraphics);
    bool enableGraphics = (choiceGraphics == 1);

    if (N <= 0) { fprintf(stderr, "So luong hat khong hop le!\n"); return 1; }

    // Initialize SDL
    if (SDL_Init(enableGraphics ? SDL_INIT_VIDEO : 0) < 0) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = NULL;
    SDL_Renderer* renderer = NULL;

    if (enableGraphics) {
        window = SDL_CreateWindow("CUDA N-Body Benchmark", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
        renderer = SDL_CreateRenderer(window, NULL);
    }

    // Init Data
    Body* h_bodies = (Body*)malloc(N * sizeof(Body));
    if (!h_bodies) { fprintf(stderr, "Khong du RAM de cap phat cho CPU!\n"); return 1; }
    srand(1337);
    initGalaxy(h_bodies, N);

    Body* d_bodies = NULL;
    cudaCheck(cudaMalloc((void**)&d_bodies, N * sizeof(Body)), "cudaMalloc");
    cudaCheck(cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body), cudaMemcpyHostToDevice), "H2D Initial");

    // Tinh toan GRID sau khi biet N
    const int GRID = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t shMemSize = 4 * BLOCK_SIZE * sizeof(float);

    int quit = 0;
    SDL_Event event;
    long totalSteps = 0;
    Uint64 lastTime = SDL_GetTicks();
    int frames = 0;

    Uint64 startWallTime = SDL_GetTicks();
    double totalComputeMs = 0.0f;

    cudaEvent_t startEvt, stopEvt;
    cudaEventCreate(&startEvt);
    cudaEventCreate(&stopEvt);

    printf("\nSimulation started. Workload: %d bodies, %ld steps.\n", N, targetSteps);

    while (!quit && totalSteps < targetSteps) {
        if (enableGraphics) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_EVENT_QUIT) quit = 1;
                if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) quit = 1;
            }
        }

        // --- Compute Step (Timed) ---
        cudaEventRecord(startEvt);
        bodyForceKernel << <GRID, BLOCK_SIZE, shMemSize >> > (d_bodies, DT, N);
        integrateKernel << <GRID, BLOCK_SIZE >> > (d_bodies, DT, N);
        cudaEventRecord(stopEvt);
        cudaEventSynchronize(stopEvt);

        float physTimeMs = 0;
        cudaEventElapsedTime(&physTimeMs, startEvt, stopEvt);
        totalComputeMs += physTimeMs;

        // --- Data Transfer & Render ---
        if (enableGraphics) {
            cudaCheck(cudaMemcpy(h_bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost), "D2H Frame");

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

            frames++;
            Uint64 currentTime = SDL_GetTicks();
            if (currentTime - lastTime >= 1000) {
                char title[128];
                snprintf(title, sizeof(title), "Step: %ld/%ld | N: %d | FPS: %.1f", totalSteps, targetSteps, N, (float)frames * 1000.0f / (currentTime - lastTime));
                SDL_SetWindowTitle(window, title);
                frames = 0; lastTime = currentTime;
            }
        }
        totalSteps++;
    }

    // --- FINAL REPORT ---
    Uint64 endWallTime = SDL_GetTicks();
    double totalWallSecs = (double)(endWallTime - startWallTime) / 1000.0;
    double totalComputeSecs = totalComputeMs / 1000.0;

    printf("\n=== PERFORMANCE REPORT ===\n");
    printf("Total particles (N): %d\n", N);
    printf("Total steps executed: %ld\n", totalSteps);
    printf("Total time (simulation): %.6f seconds\n", totalWallSecs);
    printf("Total compute time (GPU): %.6f seconds\n", totalComputeSecs);

    if (totalSteps > 0) {
        printf("Average compute time/step: %.6f s (%.3f ms)\n",
            totalComputeSecs / totalSteps, totalComputeMs / totalSteps);
    }
    printf("Overhead (Draw/Memcpy): %.6f seconds\n", totalWallSecs - totalComputeSecs);
    printf("==========================\n");

    // Cleanup
    cudaEventDestroy(startEvt); cudaEventDestroy(stopEvt);
    if (enableGraphics) {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
    }
    SDL_Quit();
    cudaFree(d_bodies); free(h_bodies);
    return 0;
}