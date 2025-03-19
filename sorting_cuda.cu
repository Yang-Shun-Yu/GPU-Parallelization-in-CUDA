#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include "sorting.h"

// -----------------------------------------------------------------------------
// Kernel 1: Initialize the counts array to 0
// -----------------------------------------------------------------------------
__global__
void initCountsKernel(int* d_count, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_count[idx] = 0;
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: Count the occurrences of each key using atomicAdd
// -----------------------------------------------------------------------------
__global__
void countKeysKernel(const Particle* d_particles, int* d_count, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int key = d_particles[idx].cell_key;
        // Use atomicAdd to increment the global count for this key
        atomicAdd(&d_count[key], 1);
    }
}

// -----------------------------------------------------------------------------
// Kernel 3: Prefix Sum (Inclusive) - NAIVE single-block approach
//           (Here, we assume max_key+1 <= blockDim.x for simplicity.)
//           If max_key is large, you'd need a proper multi-block prefix sum.
// -----------------------------------------------------------------------------
__global__
void prefixSumInclusiveKernel(int* d_count, int size)
{
    // For demonstration, we'll run this on a single thread (threadIdx.x == 0).
    // A naive sequential approach on the GPU (inefficient, but straightforward).
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 1; i < size; i++) {
            d_count[i] += d_count[i - 1];
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel 4: Reorder pass (Stable). We'll do a naive single-thread approach
//           from end to start to exactly mirror the CPU logic's stability.
// -----------------------------------------------------------------------------
__global__
void reorderStableKernel(const Particle* d_particles,
                         Particle*       d_sorted,
                         int*            d_count,
                         int             n)
{
    // Again, run this with just one thread for clarity and guaranteed order
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // Traverse in reverse to maintain stable ordering
        for (int i = n - 1; i >= 0; --i) {
            int key = d_particles[i].cell_key;
            // Decrement the prefix sum for this key to find correct position
            int pos = --d_count[key];
            d_sorted[pos] = d_particles[i];
        }
    }
}

// -----------------------------------------------------------------------------
// GPU Counting Sort Function
// -----------------------------------------------------------------------------
void countingSortCUDA(std::vector<Particle>& particles, int max_key)
{
    int n = static_cast<int>(particles.size());
    if (n == 0) return;

    // 1. Allocate device memory
    Particle* d_particles = nullptr;
    Particle* d_sorted    = nullptr;
    int*      d_count     = nullptr;

    cudaMalloc(&d_particles, n * sizeof(Particle));
    cudaMalloc(&d_sorted,    n * sizeof(Particle));
    cudaMalloc(&d_count,     (max_key + 1) * sizeof(int));

    // 2. Copy particles to device
    cudaMemcpy(d_particles, particles.data(), n * sizeof(Particle), cudaMemcpyHostToDevice);

    // 3. Initialize d_count array to 0
    int blockSize = 128;
    int gridSize  = (max_key + 1 + blockSize - 1) / blockSize;
    initCountsKernel<<<gridSize, blockSize>>>(d_count, max_key + 1);

    // 4. Count occurrences via atomicAdd
    gridSize = (n + blockSize - 1) / blockSize;
    countKeysKernel<<<gridSize, blockSize>>>(d_particles, d_count, n);

    // 5. Prefix sum (inclusive) on the counts array
    //    For simplicity, we launch it with 1 block, and a blockDim >= (max_key+1).
    prefixSumInclusiveKernel<<<1, max_key + 1>>>(d_count, max_key + 1);

    // 6. Reorder pass (stable). We use a single-thread kernel for the exact logic.
    reorderStableKernel<<<1, 1>>>(d_particles, d_sorted, d_count, n);

    // 7. Copy the sorted particles back to host
    cudaMemcpy(particles.data(), d_sorted, n * sizeof(Particle), cudaMemcpyDeviceToHost);

    // 8. Free device memory
    cudaFree(d_particles);
    cudaFree(d_sorted);
    cudaFree(d_count);
}


// Helper function: nextPowerOf2 should be accessible here.
// You can either implement it here or include it from another common header.

// Revised GPU Kernel Implementation of Bitonic Sort for Particle objects
__global__ void bitonicSortGPU(Particle* arr, int j, int k, bool ascending)
{
    // Compute the unique global index for this thread.
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    // Determine the partner index using bitwise XOR.
    unsigned int ij = i ^ j;

    // Process each pair only once.
    if (ij > i)
    {
        // Compare based on the cell_key of each Particle.
        bool compare = (arr[i].cell_key > arr[ij].cell_key);
        
        if (ascending) {
            // For ascending order:
            if ((i & k) == 0) {
                // In the first half of the bitonic sequence, swap if arr[i] > arr[ij].
                if (compare) {
                    Particle temp = arr[i];
                    arr[i] = arr[ij];
                    arr[ij] = temp;
                }
            } else {
                // In the second half, swap if arr[i] < arr[ij].
                if (!compare) {
                    Particle temp = arr[i];
                    arr[i] = arr[ij];
                    arr[ij] = temp;
                }
            }
        } else {
            // For descending order, reverse the comparisons.
            if ((i & k) == 0) {
                // In the first half, swap if arr[i] < arr[ij].
                if (!compare) {
                    Particle temp = arr[i];
                    arr[i] = arr[ij];
                    arr[ij] = temp;
                }
            } else {
                // In the second half, swap if arr[i] > arr[ij].
                if (compare) {
                    Particle temp = arr[i];
                    arr[i] = arr[ij];
                    arr[ij] = temp;
                }
            }
        }
    }
}


// -----------------------------------------------------------------------------
// Wrapper Function: Launches Bitonic Sort on the GPU
// -----------------------------------------------------------------------------
void bitonicSortCUDA(std::vector<Particle>& particles, bool ascending) {
    int n = particles.size();
    if (n == 0) return;

    // Determine the padded size (next power of 2)
    unsigned int padded_size = nextPowerOf2(n);
    std::vector<Particle> padded_particles = particles;
    if (padded_size > static_cast<unsigned>(n)) {
        Particle dummy_particle;
        dummy_particle.id = -1;
        dummy_particle.x = dummy_particle.y = std::numeric_limits<float>::infinity();
        dummy_particle.cell_key = std::numeric_limits<int>::max();
        padded_particles.resize(padded_size, dummy_particle);
    }

    // Allocate device memory
    Particle* d_particles = nullptr;
    cudaMalloc(&d_particles, padded_size * sizeof(Particle));

    // Copy data from host to device
    cudaMemcpy(d_particles, padded_particles.data(), padded_size * sizeof(Particle), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (padded_size + blockSize - 1) / blockSize;

    // Launch Bitonic Sort kernels iteratively.
    for (unsigned int k = 2; k <= padded_size; k <<= 1) {
        for (unsigned int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortGPU<<<gridSize, blockSize>>>(d_particles, j, k, ascending);
            cudaDeviceSynchronize(); // Ensure kernel completion
        }
    }

    // Copy the sorted array back to host
    cudaMemcpy(padded_particles.data(), d_particles, padded_size * sizeof(Particle), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_particles);

    // Remove dummy particles if they were added and update the original vector
    particles.assign(padded_particles.begin(), padded_particles.begin() + n);
}