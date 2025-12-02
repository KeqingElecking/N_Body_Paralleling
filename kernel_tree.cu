/*
 Stackless Barnes-Hut style N-body (3D) demo using CUDA.
 - Optimized: Padding NodeDev to 64 bytes for cache alignment.
 - Output: Binary file (particles.bin) for fast visualization.
 - Integration: Fully on GPU.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <array>

#include <cuda_runtime.h>

// --- CUDA Stubs for IntelliSense ---
#ifndef __CUDACC__
struct dim3 { int x, y, z; };
extern dim3 blockIdx;
extern dim3 blockDim;
extern dim3 threadIdx;
static inline void __syncthreads() {}
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
static inline float3 make_float3(float x, float y, float z) { return float3{ x,y,z }; }
static inline float4 make_float4(float x, float y, float z, float w) { return float4{ x,y,z,w }; }
#else
#endif

#define N_DEFAULT 1000 // Tăng default lên để thấy sức mạnh GPU
#define THETA 0.5f
#define EPS 1e-2f
#define CAPACITY 32     // Số lượng hạt tối đa trong 1 leaf
#define MAX_DEPTH 64

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

// --- OPTIMIZATION: Padded to 64 bytes ---
// Cũ: 52 bytes. Mới: 64 bytes. Giúp khớp với Cache Line của GPU.
struct NodeDev {
    float3 com;     // 12
    float mass;     // 4
    float3 center;  // 12
    float size;     // 4
    int more;       // 4
    int next;       // 4
    int start;      // 4
    int end;        // 4
    int isLeaf;     // 4
    int padding[3]; // 12 bytes padding -> Total 64 bytes
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

/* --- Host: Octree Builder (Giữ nguyên logic) --- */
void compute_bounds(const std::vector<Particle>& P, float& cx, float& cy, float& cz, float& h) {
    float minx = P[0].x, maxx = P[0].x;
    float miny = P[0].y, maxy = P[0].y;
    float minz = P[0].z, maxz = P[0].z;
    for (size_t i = 1; i < P.size(); ++i) {
        minx = std::min(minx, P[i].x); maxx = std::max(maxx, P[i].x);
        miny = std::min(miny, P[i].y); maxy = std::max(maxy, P[i].y);
        minz = std::min(minz, P[i].z); maxz = std::max(maxz, P[i].z);
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
    // Padding init
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

    // Build stackless links
    for (int i = 0; i < F; ++i) {
        int hidx = flatHostIndex[i];

        // MORE pointer
        if (!H[hidx].children.empty()) {
            outNodes[i].more = hostToFlat[H[hidx].children[0]];
        }

        // NEXT pointer
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

/* --- GPU Kernels ----------------------------------------------------------- */
#ifdef __CUDACC__
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

    int idx = 0;
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
#endif

/* --- Simulation Loop ------------------------------------------------------- */
void host_simulate(int N, int steps, float dt) {
    // 1. Initial Data Generation
    std::vector<Particle> P(N);
    std::vector<float4> ph(N), vh(N);

    // Tạo cấu hình "Galaxy" xoắn ốc để đẹp hơn random
    for (int i = 0; i < N; ++i) {
        float angle = (i * 0.1f);
        float dist = 0.1f + (i * 1.0f / N);
        P[i].x = cos(angle) * dist;
        P[i].y = sin(angle) * dist;
        P[i].z = ((rand() / (float)RAND_MAX) - 0.5f) * 0.1f;

        // Vận tốc tiếp tuyến để tạo quỹ đạo
        float vel = sqrt(1.0f) / sqrt(dist + 0.1f);
        P[i].vx = -sin(angle) * vel;
        P[i].vy = cos(angle) * vel;
        P[i].vz = 0.0f;
        P[i].mass = 1.0f;

        ph[i] = make_float4(P[i].x, P[i].y, P[i].z, P[i].mass);
        vh[i] = make_float4(P[i].vx, P[i].vy, P[i].vz, 0.0f);
    }
    // Hạt nặng ở tâm
    ph[0] = make_float4(0, 0, 0, 100.0f); vh[0] = make_float4(0, 0, 0, 0); P[0].mass = 100.0f;

    // 2. Setup Device Memory
    float4* d_particles = nullptr, * d_forces = nullptr, * d_velocities = nullptr;
    NodeDev* d_nodes = nullptr;

#ifdef __CUDACC__
    checkCuda(cudaMalloc(&d_particles, N * sizeof(float4)), "Alloc P");
    checkCuda(cudaMalloc(&d_velocities, N * sizeof(float4)), "Alloc V");
    checkCuda(cudaMalloc(&d_forces, N * sizeof(float4)), "Alloc F");
    checkCuda(cudaMemcpy(d_velocities, vh.data(), N * sizeof(float4), cudaMemcpyHostToDevice), "Copy V");
#endif

    // 3. Open Binary Output File
    const char* filename = "particles.bin";
    FILE* fp = fopen(filename, "wb");
    if (!fp) { perror("fopen"); exit(1); }

    // Write Header: [N] [STEPS]
    fwrite(&N, sizeof(int), 1, fp);
    fwrite(&steps, sizeof(int), 1, fp);

    printf("Starting simulation: N=%d Steps=%d. Output: %s\n", N, steps, filename);

    for (int step = 0; step < steps; ++step) {
        // --- A. Build Tree on Host ---
        float cx, cy, cz, h;
        compute_bounds(P, cx, cy, cz, h);

        std::vector<int> indices(N);
        for (int i = 0; i < N; ++i) indices[i] = i; // Reset indices

        std::vector<HostNode> hostNodes;
        build_node(hostNodes, indices, P, 0, N, cx, cy, cz, h, -1, 0);

        std::vector<NodeDev> nodesDev;
        flatten_tree(hostNodes, 0, nodesDev);

        // --- B. Reorder Arrays for Coalescing ---
        // Sắp xếp lại hạt và vận tốc theo thứ tự lá cây để GPU đọc tuần tự
        std::vector<float4> ph_reordered(N), vh_reordered(N);
        for (int i = 0; i < N; ++i) {
            int oldIdx = indices[i];
            ph_reordered[i] = ph[oldIdx];
            vh_reordered[i] = vh[oldIdx];
        }

#ifdef __CUDACC__
        // --- C. Copy to GPU ---
        checkCuda(cudaMemcpy(d_particles, ph_reordered.data(), N * sizeof(float4), cudaMemcpyHostToDevice), "H2D P");
        checkCuda(cudaMemcpy(d_velocities, vh_reordered.data(), N * sizeof(float4), cudaMemcpyHostToDevice), "H2D V");

        checkCuda(cudaMalloc(&d_nodes, nodesDev.size() * sizeof(NodeDev)), "Alloc Nodes"); // Re-allocating per frame for simplicity (can optimize)
        checkCuda(cudaMemcpy(d_nodes, nodesDev.data(), nodesDev.size() * sizeof(NodeDev), cudaMemcpyHostToDevice), "H2D Nodes");

        // --- D. Run Kernels ---
        int block = 128;
        int grid = (N + block - 1) / block;

        compute_force_kernel << <grid, block >> > (d_particles, d_nodes, d_forces, N, THETA);
        integrate_kernel << <grid, block >> > (d_particles, d_velocities, d_forces, N, dt);
        checkCuda(cudaDeviceSynchronize(), "Sync");

        // --- E. Get Data Back ---
        // Thay vì lấy Forces, ta lấy vị trí mới nhất
        checkCuda(cudaMemcpy(ph.data(), d_particles, N * sizeof(float4), cudaMemcpyDeviceToHost), "D2H P");
        checkCuda(cudaMemcpy(vh.data(), d_velocities, N * sizeof(float4), cudaMemcpyDeviceToHost), "D2H V");

        checkCuda(cudaFree(d_nodes), "Free Nodes");
#else
        // CPU Fallback (simplified)
        for (int i = 0; i < N; ++i) { // Dummy move
            ph_reordered[i].x += vh_reordered[i].x * dt;
            ph[i] = ph_reordered[i];
        }
#endif

        // --- F. Write Binary Frame ---
        // Ghi toàn bộ mảng vị trí (N * float4) ra file. Cực nhanh.
        fwrite(ph.data(), sizeof(float4), N, fp);

        // Update P array for next Host Tree Build
        for (int i = 0; i < N; ++i) {
            P[i].x = ph[i].x; P[i].y = ph[i].y; P[i].z = ph[i].z;
            P[i].mass = ph[i].w;
            // Lưu ý: Vận tốc cũng cần cập nhật từ vh để lần reorder sau đúng
            P[i].vx = vh[i].x; P[i].vy = vh[i].y; P[i].vz = vh[i].z;
        }

        if (step % 10 == 0) { printf("\rStep %d/%d", step, steps); fflush(stdout); }
    }
    printf("\nDone. Saved to particles.bin\n");
    fclose(fp);

#ifdef __CUDACC__
    cudaFree(d_particles); cudaFree(d_velocities); cudaFree(d_forces);
#endif
}

int main(int argc, char** argv) {
    srand(1337);
    int N = (argc >= 2) ? atoi(argv[1]) : N_DEFAULT;
    int STEPS = (argc >= 3) ? atoi(argv[2]) : 10000;
    host_simulate(N, STEPS, 1e-5f);
    return 0;
}