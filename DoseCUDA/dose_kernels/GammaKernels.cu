/**
 * @file GammaKernels.cu
 * @brief CUDA implementation of 3D gamma index computation
 * 
 * High-performance gamma analysis using GPU parallelization.
 * Each thread computes gamma for one voxel, searching a local neighborhood.
 */

#include "GammaKernels.cuh"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <stdexcept>

// Maximum search radius in voxels (DTA / min_spacing)
#define MAX_SEARCH_RADIUS 32

// Block dimensions for gamma kernel
#define GAMMA_BLOCK_X 8
#define GAMMA_BLOCK_Y 8
#define GAMMA_BLOCK_Z 4


/**
 * @brief CUDA kernel for gamma computation
 * 
 * Each thread processes one voxel. Searches the neighborhood within DTA
 * to find minimum gamma value.
 */
__global__ void gamma_kernel(
    const float* __restrict__ dose_eval,
    const float* __restrict__ dose_ref,
    float* gamma_map,
    const GammaParams params,
    unsigned int* partial_n_eval,
    unsigned int* partial_n_passed,
    double* partial_sum_gamma,
    unsigned int* partial_histogram
) {
    // Global voxel indices
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Bounds check
    if (i >= params.nx || j >= params.ny || k >= params.nz) return;
    
    // Linear index
    const int idx = i + params.nx * (j + params.ny * k);
    
    // Get reference dose at this voxel
    const float D_ref_local = dose_ref[idx];
    
    // Compute threshold
    const float threshold = (params.dose_threshold_percent / 100.0f) * params.global_dose;
    
    // Skip if below threshold
    if (D_ref_local < threshold) {
        if (gamma_map) gamma_map[idx] = -1.0f;  // Mark as not evaluated
        return;
    }
    
    // Compute dose difference criterion
    const float dd_factor = params.dd_percent / 100.0f;
    const float delta_D = params.local_normalization 
        ? (dd_factor * D_ref_local) 
        : (dd_factor * params.global_dose);
    
    if (delta_D <= 0.0f) {
        if (gamma_map) gamma_map[idx] = params.max_gamma;
        return;
    }
    
    // Search radius in voxels
    const int ri = (int)ceilf(params.dta_mm / params.sx);
    const int rj = (int)ceilf(params.dta_mm / params.sy);
    const int rk = (int)ceilf(params.dta_mm / params.sz);
    
    // Precompute DTA squared
    const float dta_sq = params.dta_mm * params.dta_mm;
    const float delta_D_sq = delta_D * delta_D;
    
    // Search for minimum gamma
    float min_gamma_sq = 1e30f;
    
    // Iterate over search neighborhood
    for (int dk = -rk; dk <= rk; dk++) {
        const int kk = k + dk;
        if (kk < 0 || kk >= params.nz) continue;
        const float dz = dk * params.sz;
        const float dz_sq = dz * dz;
        
        for (int dj = -rj; dj <= rj; dj++) {
            const int jj = j + dj;
            if (jj < 0 || jj >= params.ny) continue;
            const float dy = dj * params.sy;
            const float dy_sq = dy * dy;
            
            for (int di = -ri; di <= ri; di++) {
                const int ii = i + di;
                if (ii < 0 || ii >= params.nx) continue;
                const float dx = di * params.sx;
                const float dx_sq = dx * dx;
                
                // Physical distance squared
                const float dist_sq = dx_sq + dy_sq + dz_sq;
                
                // Skip if outside DTA sphere
                if (dist_sq > dta_sq) continue;
                
                // Get evaluated dose at offset position
                const int offset_idx = ii + params.nx * (jj + params.ny * kk);
                const float D_eval_at_offset = dose_eval[offset_idx];
                
                // Compute gamma squared
                // gamma^2 = (dist/DTA)^2 + (dose_diff/delta_D)^2
                const float spatial_term = dist_sq / dta_sq;
                const float dose_diff = D_eval_at_offset - D_ref_local;
                const float dose_term = (dose_diff * dose_diff) / delta_D_sq;
                const float gamma_sq = spatial_term + dose_term;
                
                if (gamma_sq < min_gamma_sq) {
                    min_gamma_sq = gamma_sq;
                }
                
                // Early exit if gamma <= 1 (already passed)
                if (min_gamma_sq <= 1.0f) {
                    goto done_search;
                }
            }
        }
    }
    
done_search:
    // Compute final gamma value
    float gamma_val = sqrtf(min_gamma_sq);
    if (gamma_val > params.max_gamma) {
        gamma_val = params.max_gamma;
    }
    
    // Store gamma map
    if (gamma_map) {
        gamma_map[idx] = gamma_val;
    }
    
    // Update statistics using atomics
    atomicAdd(partial_n_eval, 1);
    
    if (gamma_val <= 1.0f) {
        atomicAdd(partial_n_passed, 1);
    }
    
    // Accumulate sum (use double for precision)
    atomicAdd(partial_sum_gamma, (double)gamma_val);
    
    // Update histogram (bin index from 0 to 100)
    int bin = (int)(gamma_val / params.max_gamma * 100.0f);
    if (bin > 100) bin = 100;
    if (bin < 0) bin = 0;
    atomicAdd(&partial_histogram[bin], 1);
}


/**
 * @brief Host function to launch gamma computation
 */
void gamma_3d_cuda(
    const float* dose_eval,
    const float* dose_ref,
    float* gamma_map,
    const GammaParams& params,
    GammaStats* stats,
    cudaStream_t stream
) {
    // Validate parameters
    if (!dose_eval || !dose_ref || !stats) {
        throw std::runtime_error("Null pointer passed to gamma_3d_cuda");
    }
    
    const size_t n_voxels = (size_t)params.nx * params.ny * params.nz;
    const size_t bytes = n_voxels * sizeof(float);
    
    // Allocate device memory
    float *d_dose_eval = nullptr, *d_dose_ref = nullptr, *d_gamma_map = nullptr;
    unsigned int *d_n_eval = nullptr, *d_n_passed = nullptr;
    double *d_sum_gamma = nullptr;
    unsigned int *d_histogram = nullptr;
    
    cudaError_t err;
    
    err = cudaMalloc(&d_dose_eval, bytes);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMalloc(&d_dose_ref, bytes);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    if (gamma_map) {
        err = cudaMalloc(&d_gamma_map, bytes);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Allocate counters
    err = cudaMalloc(&d_n_eval, sizeof(unsigned int));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMalloc(&d_n_passed, sizeof(unsigned int));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMalloc(&d_sum_gamma, sizeof(double));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMalloc(&d_histogram, 101 * sizeof(unsigned int));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    // Copy data to device
    err = cudaMemcpyAsync(d_dose_eval, dose_eval, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemcpyAsync(d_dose_ref, dose_ref, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    // Initialize counters to zero
    err = cudaMemsetAsync(d_n_eval, 0, sizeof(unsigned int), stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemsetAsync(d_n_passed, 0, sizeof(unsigned int), stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemsetAsync(d_sum_gamma, 0, sizeof(double), stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemsetAsync(d_histogram, 0, 101 * sizeof(unsigned int), stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    // Launch kernel
    dim3 block(GAMMA_BLOCK_X, GAMMA_BLOCK_Y, GAMMA_BLOCK_Z);
    dim3 grid(
        (params.nx + block.x - 1) / block.x,
        (params.ny + block.y - 1) / block.y,
        (params.nz + block.z - 1) / block.z
    );
    
    gamma_kernel<<<grid, block, 0, stream>>>(
        d_dose_eval, d_dose_ref, d_gamma_map, params,
        d_n_eval, d_n_passed, d_sum_gamma, d_histogram
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    // Copy results back
    err = cudaMemcpyAsync(&stats->n_evaluated, d_n_eval, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemcpyAsync(&stats->n_passed, d_n_passed, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemcpyAsync(&stats->sum_gamma, d_sum_gamma, sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemcpyAsync(stats->histogram, d_histogram, 101 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    if (gamma_map && d_gamma_map) {
        err = cudaMemcpyAsync(gamma_map, d_gamma_map, bytes, cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    // Cleanup
    cudaFree(d_dose_eval);
    cudaFree(d_dose_ref);
    if (d_gamma_map) cudaFree(d_gamma_map);
    cudaFree(d_n_eval);
    cudaFree(d_n_passed);
    cudaFree(d_sum_gamma);
    cudaFree(d_histogram);
}


/**
 * @brief Check CUDA availability
 */
bool gamma_cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}
