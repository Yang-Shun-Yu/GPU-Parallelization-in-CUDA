#ifndef PARTICLE_H
#define PARTICLE_H

/**
 * @brief Represents a 2D particle with ID, position, and cell key.
 */
struct Particle {
    int   id;
    float x;
    float y;
    int   cell_key;
};

#endif // PARTICLE_H