#include "sorting.h"
#include <algorithm>

// Standard sort using std::sort (Introsort: QuickSort + HeapSort + InsertionSort)
void standardSort(std::vector<Particle>& particles) {
    std::sort(particles.begin(), particles.end(), 
              [](const Particle& a, const Particle& b) {
                  return a.cell_key < b.cell_key;
              });
}

// Counting Sort (O(n) time complexity, only works if cell_key values are small)
void countingSort(std::vector<Particle>& particles, int max_key) {
    std::vector<int> count(max_key + 1, 0);

    // Count occurrences of each cell_key
    for (const auto& p : particles) {
        count[p.cell_key]++;
    }

    // Convert to prefix sums
    for (int i = 1; i <= max_key; ++i) {
        count[i] += count[i - 1];
    }

    // Sort into a new array (stable sort)
    std::vector<Particle> sorted_particles(particles.size());
    for (int i = particles.size() - 1; i >= 0; --i) {
        sorted_particles[--count[particles[i].cell_key]] = particles[i];
    }

    particles = sorted_particles;
}


void compAndSwap(std::vector<Particle>& particles, int i, int j, bool dir) {
    if (dir == (particles[i].cell_key > particles[j].cell_key)) {
        std::swap(particles[i], particles[j]);
    }
}

void bitonicMerge(std::vector<Particle>& particles, int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            compAndSwap(particles, i, i + k, dir);
        }
        bitonicMerge(particles, low, k, dir);
        bitonicMerge(particles, low + k, k, dir);
    }
}

void bitonicSortUtil(std::vector<Particle>& particles, int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSortUtil(particles, low, k, true);  // Sort in ascending order
        bitonicSortUtil(particles, low + k, k, false); // Sort in descending order
        bitonicMerge(particles, low, cnt, dir);    // Merge the whole sequence in ascending order
    }
}

void bitonicSort(std::vector<Particle>& particles, bool ascending) {
    bitonicSortUtil(particles, 0, particles.size(), ascending);
}


// Merge Sort for Particle objects based on cell_key
void mergeSort(std::vector<Particle>& particles) {
    if (particles.size() <= 1) return;

    // Divide the vector into two halves.
    size_t mid = particles.size() / 2;
    std::vector<Particle> left(particles.begin(), particles.begin() + mid);
    std::vector<Particle> right(particles.begin() + mid, particles.end());

    // Recursively sort both halves.
    mergeSort(left);
    mergeSort(right);

    // Merge the sorted halves.
    size_t i = 0, j = 0, k = 0;
    while (i < left.size() && j < right.size()) {
        if (left[i].cell_key <= right[j].cell_key) {
            particles[k++] = left[i++];
        } else {
            particles[k++] = right[j++];
        }
    }
    while (i < left.size()) {
        particles[k++] = left[i++];
    }
    while (j < right.size()) {
        particles[k++] = right[j++];
    }
}


// Bucket Sort for Particle objects based on cell_key
void bucketSort(std::vector<Particle>& particles) {
    if (particles.empty()) return;

    // Determine the maximum cell_key value.
    int max_key = particles[0].cell_key;
    for (const auto &p : particles) {
        if (p.cell_key > max_key) {
            max_key = p.cell_key;
        }
    }

    // Create a bucket for each possible cell_key from 0 to max_key.
    std::vector<std::vector<Particle>> buckets(max_key + 1);
    for (const auto &p : particles) {
        buckets[p.cell_key].push_back(p);
    }

    // Optionally, sort each bucket individually.
    // If many particles share the same cell_key, you can sort each bucket;
    // here, we use mergeSort on each bucket.
    for (auto &bucket : buckets) {
        if (!bucket.empty()) {
            mergeSort(bucket);
        }
    }

    // Concatenate the buckets back into the original vector.
    std::vector<Particle> sorted_particles;
    sorted_particles.reserve(particles.size());
    for (size_t key = 0; key < buckets.size(); ++key) {
        for (const auto &p : buckets[key]) {
            sorted_particles.push_back(p);
        }
    }
    particles = sorted_particles;
}

unsigned int nextPowerOf2(unsigned int n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}
