/**
 * @file GammaKernels.cuh
 * @brief CUDA kernels for 3D gamma index computation
 * 
 * Implements GPU-accelerated gamma analysis for dose comparison.
 * Designed to integrate with the existing DoseCUDA infrastructure.
 */

#ifndef GAMMA_KERNELS_H
#define GAMMA_KERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Parameters for gamma computation
 */
struct GammaParams {
    float dta_mm;                    // Distance-to-agreement in mm
    float dd_percent;                // Dose difference percentage (e.g., 3.0 for 3%)
    float dose_threshold_percent;    // Threshold as % of global dose
    float global_dose;               // Reference dose for global normalization
    float max_gamma;                 // Cap gamma values at this max
    bool local_normalization;        // true = local, false = global
    
    // Grid parameters
    int nx, ny, nz;                  // Grid dimensions
    float sx, sy, sz;                // Spacing in mm
};

/**
 * @brief Results from gamma computation
 */
struct GammaStats {
    unsigned int n_evaluated;        // Number of voxels evaluated
    unsigned int n_passed;           // Voxels with gamma <= 1.0
    double sum_gamma;                // Sum for mean calculation
    double sum_gamma_sq;             // Sum of squares for variance
    
    // Histogram for percentile calculation (100 bins from 0 to max_gamma)
    unsigned int histogram[101];
};

/**
 * @brief Compute gamma index on GPU
 * 
 * @param dose_eval Evaluated dose array (flattened, z-major order)
 * @param dose_ref Reference dose array (same layout)
 * @param gamma_map Output gamma map (same layout), can be NULL
 * @param params Gamma computation parameters
 * @param stats Output statistics (host memory)
 * @param stream CUDA stream for async execution (0 for default)
 */
void gamma_3d_cuda(
    const float* dose_eval,
    const float* dose_ref,
    float* gamma_map,
    const GammaParams& params,
    GammaStats* stats,
    cudaStream_t stream = 0
);

/**
 * @brief Check if CUDA gamma is available
 */
bool gamma_cuda_available();

#endif // GAMMA_KERNELS_H
