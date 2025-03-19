#include <iostream>
#include <vector>
#include <atomic>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <chrono>
#include <limits>
#include <omp.h>
#include <algorithm>
#include "particle.h"
#include "sorting.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;


/**
 * @brief Calculates a hash value for integer cell coordinates (cell_x, cell_y).
 * @param x Integer cell x-coordinate.
 * @param y Integer cell y-coordinate.
 * @return Hash of the cell coordinates.
 */
int calculateHash(int x, int y) {
    // Two different large primes to reduce collisions
    const int prime1 = 200003;
    const int prime2 = 100019;
    return x * prime1 + y * prime2;
}

/**
 * @brief Converts a continuous position (x,y) into integer cell coordinates based on cell size.
 * @param x Particle’s x position.
 * @param y Particle’s y position.
 * @param cell_size Edge length of the square cell.
 * @return (cell_x, cell_y) in integer coordinates.
 */
pair<int, int> getCellCoordinates(float x, float y, float cell_size) {
    int cell_x = static_cast<int>(floor(x / cell_size));
    int cell_y = static_cast<int>(floor(y / cell_size));
    return {cell_x, cell_y};
}

/**
 * @brief Assigns a cell key to each Particle based on its position and the cell hash.
 * @param particles Vector of Particle objects to update.
 * @param cell_size Edge length of the square cell.
 */
void assignCellKey(vector<Particle>& particles, float cell_size) {
    int num_particles = static_cast<int>(particles.size());
    for (auto& particle : particles) {
        auto [cell_x, cell_y] = getCellCoordinates(particle.x, particle.y, cell_size);
        int cell_hash = calculateHash(cell_x, cell_y);
        // Use modulo with total number of particles to keep cell_key in range [0, num_particles-1]
        particle.cell_key = cell_hash % num_particles;
    }
}

/**
 * @brief Generates random particles within the given bounding box, using OpenMP.
 * @param num_particles Number of particles to generate.
 * @param x_min Minimum x-value.
 * @param x_max Maximum x-value.
 * @param y_min Minimum y-value.
 * @param y_max Maximum y-value.
 * @return A vector of randomly generated Particle objects.
 */
vector<Particle> generateParticles(int num_particles,
                                   float x_min, float x_max,
                                   float y_min, float y_max) 
{
    vector<Particle> particles(num_particles);

    #pragma omp parallel
    {
        // Each thread gets its own random engine, seeded uniquely
        mt19937 gen(omp_get_thread_num() + static_cast<unsigned>(time(NULL)));
        uniform_real_distribution<float> x_dist(x_min, x_max);
        uniform_real_distribution<float> y_dist(y_min, y_max);

        // Parallel for loop to fill the particle data
        #pragma omp for
        for (int i = 0; i < num_particles; ++i) {
            particles[i].id = i;
            particles[i].x = x_dist(gen);
            particles[i].y = y_dist(gen);
        }
    }

    return particles;
}

/**
 * @brief Computes the “start index” for each cell key in a sorted particle array.
 *
 * After sorting by cell_key, this function records the first index in the array
 * that matches each cell_key. If a cell_key never appears, `start_indices[cell_key]`
 * will remain std::numeric_limits<int>::max().
 *
 * @param particles A vector of Particles sorted by their cell_key.
 * @return A vector of size [num_particles + 1], where each entry is
 *         the first index of that cell_key in particles.
 */
vector<int> computeStartIndices(const vector<Particle>& particles) {
    int num_particles = static_cast<int>(particles.size());
    vector<int> start_indices(num_particles + 1, numeric_limits<int>::max());

    // Record the first occurrence of each cell_key
    for (size_t i = 0; i < particles.size(); ++i) {
        int key = particles[i].cell_key;
        if (start_indices[key] == numeric_limits<int>::max()) {
            start_indices[key] = static_cast<int>(i);
        }
    }

    return start_indices;
}

/**
 * @brief Finds particles within a given radius of a specified particle (by ID).
 *
 * This checks the particle's cell and its neighboring cells to see which particles
 * lie within the specified radius (using a squared distance check).
 *
 * @param particles Main list of particles (unordered).
 * @param spatial_lookup Sorted list of particles by cell_key.
 * @param start_indices The starting index for each cell_key in spatial_lookup.
 * @param particle_id ID of the “reference” particle we’re examining.
 * @param cell_size Edge length of each cell.
 * @param x_min Minimum x-value of the bounding area.
 * @param x_max Maximum x-value of the bounding area.
 * @param y_min Minimum y-value of the bounding area.
 * @param y_max Maximum y-value of the bounding area.
 * @param radius The radius within which we search for neighbors.
 * @param particles_in_radius Output vector of IDs of neighboring particles found.
 */
void computeParticleInsideRadius(const vector<Particle>& particles,
                                 const vector<Particle>& spatial_lookup,
                                 const vector<int>&     start_indices,
                                 int                    particle_id,
                                 float                  cell_size,
                                 float                  x_min,
                                 float                  x_max,
                                 float                  y_min,
                                 float                  y_max,
                                 float                  radius,
                                 vector<int>&           particles_in_radius)
{
    // Get cell coords of target particle
    auto [cell_x, cell_y] = getCellCoordinates(particles[particle_id].x,
                                               particles[particle_id].y,
                                               cell_size);

    // Bounds for cell coordinates
    auto [cell_min_x, cell_min_y] = getCellCoordinates(x_min, y_min, cell_size);
    auto [cell_max_x, cell_max_y] = getCellCoordinates(x_max, y_max, cell_size);

    int num_particles_total = static_cast<int>(particles.size());

    // Check neighboring cells in range [-1..1] around (cell_x, cell_y)
    for (int dx = -1; dx <= 1; dx++) {
        int nx = cell_x + dx;
        if (nx < cell_min_x || nx > cell_max_x) {
            continue; 
        }
        for (int dy = -1; dy <= 1; dy++) {
            int ny = cell_y + dy;
            if (ny < cell_min_y || ny > cell_max_y) {
                continue;
            }

            // Hash and cell_key for neighbor cell
            int cell_hash = calculateHash(nx, ny);
            int cell_key = cell_hash % num_particles_total;

            // If that cell_key doesn’t exist (never assigned), skip
            if (start_indices[cell_key] == numeric_limits<int>::max()) {
                continue;
            }

            // Go through all particles in that cell
            for (int i = start_indices[cell_key]; i < num_particles_total; i++) {
                if (spatial_lookup[i].cell_key != cell_key) {
                    // Once we're out of this cell_key range, break
                    break;
                }

                // Check if current particle is within radius (excluding self)
                float dx_ = particles[particle_id].x - spatial_lookup[i].x;
                float dy_ = particles[particle_id].y - spatial_lookup[i].y;
                float distance_squared = dx_ * dx_ + dy_ * dy_;

                if (distance_squared <= (radius * radius) &&
                    spatial_lookup[i].id != particle_id)
                {
                    particles_in_radius.push_back(spatial_lookup[i].id);
                }
            }
        }
    }
}


/**
 * @brief Program entry point.
 */
int main() {
    // Number of particles
    unsigned int num_particles = 3000000;

    // Bounding box (with small offset)
    float epsilon = 1e-5f;
    float x_min   = 0.0f + epsilon;
    float y_min   = 0.0f + epsilon;
    float x_max   = 50.0f - epsilon;
    float y_max   = 50.0f - epsilon;

    // Cell configuration
    float cell_size = 1.0f;

    // Generate random particles
    vector<Particle> particles = generateParticles(num_particles,
                                                   x_min, x_max,
                                                   y_min, y_max);

    // Assign each particle a cell_key
    assignCellKey(particles, cell_size);

    // Create a copy for spatial lookup, then sort it by cell_key

    vector<Particle> spatial_lookup = particles;
    // Determine the next power of two



    auto grid_start = chrono::high_resolution_clock::now();

        // Choose sorting method at compile-time
#ifdef USE_COUNTING_SORT
    int max_key = num_particles;  // Assuming cell_key is in [0, num_particles-1]
    countingSort(spatial_lookup, max_key);
#elif defined(USE_BITONIC_SORT)
    // Bitonic sort implementation (with padding as needed)
    unsigned int padded_size = nextPowerOf2(num_particles);
    if (padded_size > num_particles) {
        Particle dummy_particle;
        dummy_particle.id = -1;
        dummy_particle.x = dummy_particle.y = std::numeric_limits<float>::infinity();
        dummy_particle.cell_key = std::numeric_limits<int>::max();
        spatial_lookup.resize(padded_size, dummy_particle);
    }
    bitonicSort(spatial_lookup, true);
#elif defined(USE_MERGE_SORT)
    mergeSort(spatial_lookup);
#elif defined(USE_BUCKET_SORT)
    bucketSort(spatial_lookup);
#elif defined(USE_COUNTING_SORT_CUDA)
    int max_key = num_particles;
    countingSortCUDA(spatial_lookup, max_key);
#elif defined(USE_BITONIC_SORT_CUDA)

    bitonicSortCUDA(spatial_lookup, true);

#else
    standardSort(spatial_lookup); // Default uses std::sort
#endif


    auto grid_end = chrono::high_resolution_clock::now();
    // After sorting, remove the dummy particles if any were added.
    if (spatial_lookup.size() > num_particles) {
        spatial_lookup.resize(num_particles);
    }

    // Compute where each cell_key begins in the sorted array
    vector<int> start_indices = computeStartIndices(spatial_lookup);

    // Example radius query around each particle
    float radius = 0.5f;

    // For demonstration, store neighbors for each particle
    vector<vector<int>> particles_in_radius_all(num_particles);

    // Parallel loop: find neighbors for every particle
    #pragma omp parallel for
    for (unsigned int i = 0; i < num_particles; ++i) {
        computeParticleInsideRadius(particles,
                                    spatial_lookup,
                                    start_indices,
                                    i,   // current particle ID
                                    cell_size,
                                    x_min, x_max,
                                    y_min, y_max,
                                    radius,
                                    particles_in_radius_all[i]);
    }

    chrono::duration<double, milli> elapsed = grid_end - grid_start;
    cout << "Sorting Time taken: " << elapsed.count() << " ms" << endl;

    return 0;
}
