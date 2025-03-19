# Fluid Particle Simulation

This project implements a fluid particle simulation with an emphasis on efficient spatial partitioning and parallel sorting algorithms. It leverages both CPU and GPU (CUDA) parallelism to optimize the simulation of fluid dynamics using particles.

# Project Structure

- **main.cpp**  
  Contains the main simulation logic: particle generation, spatial partitioning, sorting, and neighbor queries.

- **sorting.cpp** and **sorting.h**  
  Implement various CPU sorting algorithms (standard sort, counting sort, bitonic sort, merge sort, and bucket sort).

- **sorting_cuda.cu**  
  Implements CUDA kernels and wrapper functions for counting sort and bitonic sort on the GPU.

- **particle.h**  
  Defines the Particle structure and associated data.

- **Makefile**  
  Provides build targets for different sorting methods and CUDA-enabled variants.

# Build Instructions

Use the provided Makefile to build the project with your desired sorting method. For example:

### CPU Sort
```bash
make standard_sort
make counting_sort
make bitonic_sort
make merge_sort
make bucket_sort
```
### GPU Sort (CUDA)
```bash
make counting_sort_cuda
make bitonic_sort_cuda
```



### Clean Build Artifacts
```bash
make clean
```
### Output displays the time taken to generate and sort the spatial_lookup structure.
```bash
Bitonic Sort (CUDA) - Sorting Time taken: 143.394 ms
```
The spatial_lookup is a sorted copy of the particle array, where each particle is assigned a cell key based on its spatial position. This sorting groups particles that are spatially close together, making neighbor queries more efficient during the simulation.
