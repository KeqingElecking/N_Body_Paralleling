#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Parameters
#define N 1000            // Number of bodies
#define DT 0.01f          // Time step
#define STEPS 1000        // Number of simulation steps
#define G 9.8f            // Gravitational constant (normalized)
#define SOFTENING 1e-2f   // Softening length (eps)

typedef struct {
    float x, y, z;      // Position
    float vx, vy, vz;   // Velocity
    float mass;         // Mass
} Body;

// Initialize bodies with random positions and velocities
void randomizeBodies(Body* p) {
    for (int i = 0; i < N; i++) {
        // Random position between -1.0 and 1.0
        p[i].x = 2.0f * (float)rand() / RAND_MAX - 1.0f;
        p[i].y = 2.0f * (float)rand() / RAND_MAX - 1.0f;
        p[i].z = 2.0f * (float)rand() / RAND_MAX - 1.0f;

        // Initial velocity (starting at rest)
        p[i].vx = 0.0f;
        p[i].vy = 0.0f;
        p[i].vz = 0.0f;

        // Random mass
        p[i].mass = (float)rand() / RAND_MAX + 0.1f;
    }
}

// Compute forces and update velocities (sequential version)
void computeForces(Body* bodies, float dt) {
    const float eps2 = SOFTENING * SOFTENING;

    for (int i = 0; i < N; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        float xi = bodies[i].x;
        float yi = bodies[i].y;
        float zi = bodies[i].z;
        float mi = bodies[i].mass;

        // Calculate force from all other bodies
        for (int j = 0; j < N; j++) {
            if (j == i) continue;  // Skip self-interaction

            float dx = bodies[j].x - xi;
            float dy = bodies[j].y - yi;
            float dz = bodies[j].z - zi;

            // Distance squared with softening
            float distSqr = dx * dx + dy * dy + dz * dz + eps2;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            // Force magnitude: F = G * m1 * m2 / r^2
            float f = G * mi * bodies[j].mass * invDist3;

            // Accumulate force components
            Fx += f * dx;
            Fy += f * dy;
            Fz += f * dz;
        }

        // Update velocity: v = v + (F/m) * dt
        bodies[i].vx += (Fx / mi) * dt;
        bodies[i].vy += (Fy / mi) * dt;
        bodies[i].vz += (Fz / mi) * dt;
    }
}

// Update positions based on velocities
void integratePositions(Body* bodies, float dt) {
    for (int i = 0; i < N; i++) {
        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }
}

int main() {
    // Allocate memory for bodies
    Body* bodies = (Body*)malloc(N * sizeof(Body));
    if (!bodies) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize random seed and bodies
    srand((unsigned)time(NULL));
    randomizeBodies(bodies);

    printf("N-Body Simulation (Sequential Version)\n");
    printf("Number of bodies: %d\n", N);
    printf("Time step: %f\n", DT);
    printf("Total steps: %d\n", STEPS);
    printf("Starting simulation...\n\n");

    // Start timing
    clock_t start = clock();

    // Simulation loop
    for (int step = 0; step < STEPS; step++) {
        // Compute forces and update velocities
        computeForces(bodies, DT);

        // Update positions
        integratePositions(bodies, DT);

        // Output positions for visualization
        printf("timestep\n");
        for (int i = 0; i < N; i++) {
            printf("%f %f %f\n", bodies[i].x, bodies[i].y, bodies[i].z);
        }

        // Progress indicator (every 100 steps)
        if ((step + 1) % 100 == 0) {
            fprintf(stderr, "Step %d/%d completed\n", step + 1, STEPS);
        }
    }

    // End timing
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    fprintf(stderr, "\nSimulation completed!\n");
    fprintf(stderr, "Total time: %.2f seconds\n", elapsed);
    fprintf(stderr, "Average time per step: %.4f seconds\n", elapsed / STEPS);

    // Cleanup
    free(bodies);
    return 0;
}
