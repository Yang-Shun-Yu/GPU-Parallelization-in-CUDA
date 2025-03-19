#ifndef SORTING_H
#define SORTING_H

#include <vector>
#include "particle.h"

void standardSort(std::vector<Particle>& particles);
void countingSort(std::vector<Particle>& particles, int max_key);
void bitonicSort(std::vector<Particle>& arr, bool ascending = true);
void mergeSort(std::vector<Particle>& particles);
void bucketSort(std::vector<Particle>& particles);
void countingSortCUDA(std::vector<Particle>& particles, int max_key);
void bitonicSortCUDA(std::vector<Particle>& particles, bool ascending);
unsigned int nextPowerOf2(unsigned int n);
#ifdef __CUDACC__
#ifdef USE_BITONIC_SORT_CUDA
extern __global__ void bitonicSortGPU(Particle* arr, int j, int k, bool ascending);
#endif
#endif

#endif // SORTING_H