#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cuda_runtime.h>
#include <SDL3/SDL.h>

// --- CUDA Stubs ---
#ifndef __CUDACC__
struct dim3 { int x, y, z; };
extern dim3 blockIdx; extern dim3 blockDim; extern dim3 threadIdx;
static inline void __syncthreads() {}
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
static inline float3 make_float3(float x, float y, float z) { return float3{ x,y,z }; }
static inline float4 make_float4(float x, float y, float z, float w) { return float4{ x,y,z,w }; }
#endif

// --- CONSTANTS ---
#define N 5000              // Moderate N for CPU Tree Build
#define THETA 0.5f          // Accuracy
#define CAPACITY 32         // Leaf capacity
#define MAX_DEPTH 64
#define DT 0.005f           // Time step

// Updated Constants to match kernel_SDL.cu scale
#define EPS 0.1f            // Softening (increased for larger scale)
#define SCALE 3.5f          // Scale (decreased because coordinates are ~100.0 now)

// Graphics
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800     

// --- DATA STRUCTURES (From kernel_tree.cu) ---

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

// Padded Node for GPU Alignment (64 bytes)
struct NodeDev {
    float3 com;
    float mass;
    float3 center;
    float size;
    int more;       // "Down" pointer
    int next;       // "Sideways" pointer
    int start;      // Particle index start
    int end;        // Particle index end
    int isLeaf;
    int padding[3];
};

struct HostNode {
    float cx, cy, cz;
    float h;
    int start, end;
    std::vector<int> children;
    int parent;
    float comx, comy, comz;
    float mass;
    bool isLeaf;
};

static inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA Error %s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

// --- HOST: TREE BUILDER (CPU Logic) ---

void compute_bounds(const std::vector<Particle>& P, float& cx, float& cy, float& cz, float& h) {
    float minx = P[0].x, maxx = P[0].x;
    float miny = P[0].y, maxy = P[0].y;
    float minz = P[0].z, maxz = P[0].z;
    for (size_t i = 1; i < P.size(); ++i) {
        if (P[i].x < minx) minx = P[i].x; if (P[i].x > maxx) maxx = P[i].x;
        if (P[i].y < miny) miny = P[i].y; if (P[i].y > maxy) maxy = P[i].y;
        if (P[i].z < minz) minz = P[i].z; if (P[i].z > maxz) maxz = P[i].z;
    }
    cx = 0.5f * (minx + maxx);
    cy = 0.5f * (miny + maxy);
    cz = 0.5f * (minz + maxz);
    float dx = maxx - minx, dy = maxy - miny, dz = maxz - minz;
    h = 0.5f * std::max(std::max(dx, dy), dz) + 1e-5f;
}

int build_node(std::vector<HostNode>& nodes, std::vector<int>& indices, const std::vector<Particle>& P,
    int start, int end, float cx, float cy, float cz, float h, int parent, int depth)
{
    HostNode node;
    node.start = start; node.end = end;
    node.cx = cx; node.cy = cy; node.cz = cz; node.h = h;
    node.parent = parent; node.children.clear();
    node.mass = 0.0f; node.comx = 0; node.comy = 0; node.comz = 0;
    node.isLeaf = true;

    for (int i = start; i < end; ++i) {
        const Particle& pt = P[indices[i]];
        node.mass += pt.mass;
        node.comx += pt.mass * pt.x;
        node.comy += pt.mass * pt.y;
        node.comz += pt.mass * pt.z;
    }
    if (node.mass > 0.0f) {
        node.comx /= node.mass; node.comy /= node.mass; node.comz /= node.mass;
    }
    else {
        node.comx = cx; node.comy = cy; node.comz = cz;
    }

    int myIndex = (int)nodes.size();
    nodes.push_back(node);

    int count = end - start;
    if (count <= CAPACITY || depth >= MAX_DEPTH) {
        return myIndex;
    }

    float nh = 0.5f * h;
    int bucketCounts[8] = { 0 };
    std::vector<int> buckets[8];

    for (int i = start; i < end; ++i) {
        const Particle& pt = P[indices[i]];
        int oct = 0;
        if (pt.x > cx) oct |= 1;
        if (pt.y > cy) oct |= 2;
        if (pt.z > cz) oct |= 4;
        buckets[oct].push_back(indices[i]);
        ++bucketCounts[oct];
    }

    int nonEmpty = 0;
    for (int b = 0; b < 8; ++b) if (!buckets[b].empty()) ++nonEmpty;
    if (nonEmpty <= 1) return myIndex;

    int pos = start;
    for (int b = 0; b < 8; ++b) for (int id : buckets[b]) indices[pos++] = id;

    pos = start;
    for (int b = 0; b < 8; ++b) {
        int cnt = bucketCounts[b];
        if (cnt == 0) continue;
        float ncx = cx + ((b & 1) ? nh : -nh);
        float ncy = cy + ((b & 2) ? nh : -nh);
        float ncz = cz + ((b & 4) ? nh : -nh);
        int childIdx = build_node(nodes, indices, P, pos, pos + cnt, ncx, ncy, ncz, nh, myIndex, depth + 1);
        nodes[myIndex].children.push_back(childIdx);
        pos += cnt;
    }
    nodes[myIndex].isLeaf = false;
    return myIndex;
}

static void flatten_dfs(const std::vector<HostNode>& H, int hidx, std::vector<NodeDev>& outNodes, std::vector<int>& flatHostIndex) {
    int flatIdx = (int)outNodes.size();
    NodeDev nd;
    nd.com = make_float3(H[hidx].comx, H[hidx].comy, H[hidx].comz);
    nd.center = make_float3(H[hidx].cx, H[hidx].cy, H[hidx].cz);
    nd.mass = H[hidx].mass;
    nd.size = H[hidx].h;
    nd.more = -1; nd.next = -1;
    nd.start = H[hidx].start; nd.end = H[hidx].end;
    nd.isLeaf = H[hidx].isLeaf ? 1 : 0;
    nd.padding[0] = 0; nd.padding[1] = 0; nd.padding[2] = 0;

    outNodes.push_back(nd);
    flatHostIndex.push_back(hidx);

    for (int childHost : H[hidx].children) flatten_dfs(H, childHost, outNodes, flatHostIndex);
}

void flatten_tree(const std::vector<HostNode>& H, int rootIdx, std::vector<NodeDev>& outNodes) {
    outNodes.clear();
    std::vector<int> flatHostIndex;
    flatten_dfs(H, rootIdx, outNodes, flatHostIndex);

    int F = (int)outNodes.size();
    std::vector<int> hostToFlat(H.size(), -1);
    for (int i = 0; i < F; ++i) hostToFlat[flatHostIndex[i]] = i;

    // Stackless Links (More/Next)
    for (int i = 0; i < F; ++i) {
        int hidx = flatHostIndex[i];
        if (!H[hidx].children.empty()) {
            outNodes[i].more = hostToFlat[H[hidx].children[0]];
        }
        int parent = H[hidx].parent;
        int nextFlat = -1;
        int currentHost = hidx;
        while (parent != -1) {
            const auto& siblings = H[parent].children;
            auto it = std::find(siblings.begin(), siblings.end(), currentHost);
            if (it != siblings.end() && (it + 1) != siblings.end()) {
                nextFlat = hostToFlat[*(it + 1)];
                break;
            }
            currentHost = parent;
            parent = H[parent].parent;
        }
        outNodes[i].next = nextFlat;
    }
}

// --- DEVICE: KERNELS (GPU Logic) ---

__device__ float length_eps(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z + EPS * EPS);
}

__device__ float3 calc_node_force_p2m(const float3& pos, const NodeDev& node) {
    float3 d = make_float3(node.com.x - pos.x, node.com.y - pos.y, node.com.z - pos.z);
    float r = length_eps(d);
    float invR3 = 1.0f / (r * r * r);
    float s = node.mass * invR3;
    return make_float3(d.x * s, d.y * s, d.z * s);
}

__global__ void compute_force_kernel(
    const float4* __restrict__ d_particles,
    const NodeDev* __restrict__ d_nodes,
    float4* __restrict__ d_forces,
    int num_particles,
    float theta)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    float4 p4 = d_particles[tid];
    float3 pos = make_float3(p4.x, p4.y, p4.z);
    float3 acc = make_float3(0.0f, 0.0f, 0.0f);

    int idx = 0; // Start at root
    // Stackless Traversal
    while (idx != -1) {
        NodeDev node = d_nodes[idx];
        float3 d = make_float3(node.center.x - pos.x, node.center.y - pos.y, node.center.z - pos.z);
        float dist = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z) + 1e-12f;

        bool should_open = (node.more != -1) && ((node.size / dist) > theta);

        if (should_open) {
            idx = node.more;
        }
        else {
            if (node.isLeaf) {
                for (int j = node.start; j < node.end; ++j) {
                    if (j == tid) continue;
                    float4 q = d_particles[j];
                    float3 dq = make_float3(q.x - pos.x, q.y - pos.y, q.z - pos.z);
                    float r = length_eps(dq);
                    float invR3 = 1.0f / (r * r * r);
                    float s = q.w * invR3;
                    acc.x += dq.x * s; acc.y += dq.y * s; acc.z += dq.z * s;
                }
            }
            else {
                float3 a = calc_node_force_p2m(pos, node);
                acc.x += a.x; acc.y += a.y; acc.z += a.z;
            }
            idx = node.next;
        }
    }
    d_forces[tid] = make_float4(acc.x, acc.y, acc.z, 0.0f);
}

__global__ void integrate_kernel(
    float4* particles,
    float4* velocities,
    const float4* forces,
    int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 p = particles[i];
    float4 v = velocities[i];
    float4 f = forces[i];

    float invMass = 1.0f / p.w;
    v.x += f.x * invMass * dt;
    v.y += f.y * invMass * dt;
    v.z += f.z * invMass * dt;

    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;

    particles[i] = p;
    velocities[i] = v;
}

// --- MAIN LOOP ---

int main(int argc, char** argv) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL Init Failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("CUDA Barnes-Hut Tree + Stats", WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    // 1. Initial Data
    std::vector<Particle> P(N);
    std::vector<float4> ph(N), vh(N);

    // --- GALAXY SETUP (From kernel_SDL.cu) ---
    srand(1337);
    for (int i = 0; i < N; ++i) {
        if (i == 0) {
            P[i].x = 0; P[i].y = 0; P[i].z = 0;
            P[i].vx = 0; P[i].vy = 0; P[i].vz = 0;
            P[i].mass = 5000.0f;
        }
        else {
            float angle = (i * 0.1f);
            float dist = 100.0f + (i * 10.0f / N);
            P[i].x = cos(angle) * dist;
            P[i].y = sin(angle) * dist;
            P[i].z = ((rand() / (float)RAND_MAX) - 0.5f) * 10.0f;

            float vel = sqrt(5000.0f) / sqrt(dist); // sqrt(G * M_center / r)
            P[i].vx = -sin(angle) * vel;
            P[i].vy = cos(angle) * vel;
            P[i].vz = 0.0f;
            P[i].mass = 1.0f + ((rand() / (float)RAND_MAX));
        }

        // Copy to Float4 format for GPU
        ph[i] = make_float4(P[i].x, P[i].y, P[i].z, P[i].mass);
        vh[i] = make_float4(P[i].vx, P[i].vy, P[i].vz, 0.0f);
    }

    // 2. Device Memory
    float4* d_particles = nullptr, * d_forces = nullptr, * d_velocities = nullptr;
    NodeDev* d_nodes = nullptr;

    checkCuda(cudaMalloc(&d_particles, N * sizeof(float4)), "Alloc P");
    checkCuda(cudaMalloc(&d_velocities, N * sizeof(float4)), "Alloc V");
    checkCuda(cudaMalloc(&d_forces, N * sizeof(float4)), "Alloc F");
    checkCuda(cudaMemcpy(d_velocities, vh.data(), N * sizeof(float4), cudaMemcpyHostToDevice), "Copy V");

    int quit = 0;
    SDL_Event event;

    // --- Stats Variables ---
    Uint64 lastTime = SDL_GetTicks();
    int frames = 0;
    char titleBuffer[256];
    long totalSteps = 0;

    // Accumulators for Report
    Uint64 startWallTime = SDL_GetTicks();
    double totalComputeMs = 0.0f;

    // Events for timing GPU part
    cudaEvent_t startEvt, stopEvt;
    cudaEventCreate(&startEvt);
    cudaEventCreate(&stopEvt);

    printf("Starting Stackless Barnes-Hut. N=%d\n", N);
    printf("Press ESC to see Performance Report.\n");

    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) quit = 1;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) quit = 1;
        }

        // --- A. CPU Tree Build (Bottleneck) ---
        float cx, cy, cz, h;
        compute_bounds(P, cx, cy, cz, h);

        std::vector<int> indices(N);
        for (int i = 0; i < N; ++i) indices[i] = i;

        std::vector<HostNode> hostNodes;
        hostNodes.reserve(N * 2);
        build_node(hostNodes, indices, P, 0, N, cx, cy, cz, h, -1, 0);

        std::vector<NodeDev> nodesDev;
        flatten_tree(hostNodes, 0, nodesDev);

        std::vector<float4> ph_reordered(N), vh_reordered(N);
        for (int i = 0; i < N; ++i) {
            int oldIdx = indices[i];
            ph_reordered[i] = ph[oldIdx];
            vh_reordered[i] = vh[oldIdx];
        }

        // --- B. GPU Compute Step (Timed) ---
        cudaEventRecord(startEvt);

        checkCuda(cudaMemcpy(d_particles, ph_reordered.data(), N * sizeof(float4), cudaMemcpyHostToDevice), "H2D P");
        checkCuda(cudaMemcpy(d_velocities, vh_reordered.data(), N * sizeof(float4), cudaMemcpyHostToDevice), "H2D V");

        // Realloc nodes if needed
        static size_t currentNodeSize = 0;
        if (nodesDev.size() > currentNodeSize) {
            if (d_nodes) cudaFree(d_nodes);
            currentNodeSize = nodesDev.size() * 1.5;
            checkCuda(cudaMalloc(&d_nodes, currentNodeSize * sizeof(NodeDev)), "Alloc Nodes");
        }
        checkCuda(cudaMemcpy(d_nodes, nodesDev.data(), nodesDev.size() * sizeof(NodeDev), cudaMemcpyHostToDevice), "H2D Nodes");

        int block = 128;
        int grid = (N + block - 1) / block;

        compute_force_kernel << <grid, block >> > (d_particles, d_nodes, d_forces, N, THETA);
        integrate_kernel << <grid, block >> > (d_particles, d_velocities, d_forces, N, DT);

        cudaEventRecord(stopEvt);
        cudaEventSynchronize(stopEvt);

        float physTimeMs = 0;
        cudaEventElapsedTime(&physTimeMs, startEvt, stopEvt);
        totalComputeMs += physTimeMs;

        // --- C. Retrieve Data ---
        checkCuda(cudaMemcpy(ph.data(), d_particles, N * sizeof(float4), cudaMemcpyDeviceToHost), "D2H P");
        checkCuda(cudaMemcpy(vh.data(), d_velocities, N * sizeof(float4), cudaMemcpyDeviceToHost), "D2H V");

        // Update Host P for next tree build
        for (int i = 0; i < N; ++i) {
            P[i].x = ph[i].x; P[i].y = ph[i].y; P[i].z = ph[i].z;
            P[i].mass = ph[i].w;
            P[i].vx = vh[i].x; P[i].vy = vh[i].y; P[i].vz = vh[i].z;
        }

        // --- D. Render ---
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

        for (int i = 0; i < N; ++i) {
            // New Scaling Logic for larger Galaxy
            float sx = ph[i].x * SCALE + WINDOW_WIDTH / 2.0f;
            float sy = ph[i].y * SCALE + WINDOW_HEIGHT / 2.0f;
            if (sx >= 0 && sx < WINDOW_WIDTH && sy >= 0 && sy < WINDOW_HEIGHT) {
                SDL_RenderPoint(renderer, sx, sy);
            }
        }
        SDL_RenderPresent(renderer);

        // --- Stats ---
        frames++;
        totalSteps++;
        Uint64 currentTime = SDL_GetTicks();
        if (currentTime - lastTime >= 1000) {
            float fps = (float)frames * 1000.0f / (float)(currentTime - lastTime);
            snprintf(titleBuffer, sizeof(titleBuffer), "Step: %ld | GPU: %.3f ms | FPS: %.1f | N: %d", totalSteps, physTimeMs, fps, N);
            SDL_SetWindowTitle(window, titleBuffer);
            frames = 0;
            lastTime = currentTime;
        }
    }

    // --- FINAL REPORT ---
    Uint64 endWallTime = SDL_GetTicks();
    double totalWallSecs = (double)(endWallTime - startWallTime) / 1000.0;
    double totalComputeSecs = totalComputeMs / 1000.0;

    printf("\n=== BARNES-HUT PERFORMANCE REPORT ===\n");
    printf("Total steps executed: %ld\n", totalSteps);
    printf("Total time (simulation): %.6f seconds\n", totalWallSecs);
    printf("Total compute time (GPU): %.6f seconds\n", totalComputeSecs);

    if (totalSteps > 0) {
        printf("Average total time/step: %.6f s (%.3f ms)\n",
            totalWallSecs / totalSteps, (totalWallSecs * 1000.0) / totalSteps);
        printf("Average compute time/step: %.6f s (%.3f ms)\n",
            totalComputeSecs / totalSteps, totalComputeMs / totalSteps);
    }

    printf("Overhead (TreeBuild/Draw/PCIe): %.6f seconds\n", totalWallSecs - totalComputeSecs);
    printf("=====================================\n");

    cudaFree(d_particles); cudaFree(d_velocities); cudaFree(d_forces); cudaFree(d_nodes);
    cudaEventDestroy(startEvt); cudaEventDestroy(stopEvt);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}