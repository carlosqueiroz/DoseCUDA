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
#include <cstdint>
#include <vector>

#define HIST_BINS 101

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
__device__ __forceinline__ float sample_trilinear(
    const float* __restrict__ volume,
    int nx, int ny, int nz,
    float fx, float fy, float fz
) {
    // Clamp to valid range
    fx = fminf(fmaxf(fx, 0.0f), nx - 1.0f);
    fy = fminf(fmaxf(fy, 0.0f), ny - 1.0f);
    fz = fminf(fmaxf(fz, 0.0f), nz - 1.0f);

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int z0 = (int)floorf(fz);

    int x1 = min(x0 + 1, nx - 1);
    int y1 = min(y0 + 1, ny - 1);
    int z1 = min(z0 + 1, nz - 1);

    float tx = fx - x0;
    float ty = fy - y0;
    float tz = fz - z0;

    int idx000 = x0 + nx * (y0 + ny * z0);
    int idx100 = x1 + nx * (y0 + ny * z0);
    int idx010 = x0 + nx * (y1 + ny * z0);
    int idx110 = x1 + nx * (y1 + ny * z0);
    int idx001 = x0 + nx * (y0 + ny * z1);
    int idx101 = x1 + nx * (y0 + ny * z1);
    int idx011 = x0 + nx * (y1 + ny * z1);
    int idx111 = x1 + nx * (y1 + ny * z1);

    float c000 = volume[idx000];
    float c100 = volume[idx100];
    float c010 = volume[idx010];
    float c110 = volume[idx110];
    float c001 = volume[idx001];
    float c101 = volume[idx101];
    float c011 = volume[idx011];
    float c111 = volume[idx111];

    float c00 = c000 * (1 - tx) + c100 * tx;
    float c01 = c001 * (1 - tx) + c101 * tx;
    float c10 = c010 * (1 - tx) + c110 * tx;
    float c11 = c011 * (1 - tx) + c111 * tx;

    float c0 = c00 * (1 - ty) + c10 * ty;
    float c1 = c01 * (1 - ty) + c11 * ty;

    return c0 * (1 - tz) + c1 * tz;
}

__global__ void gamma_kernel(
    const float* __restrict__ dose_eval,
    const float* __restrict__ dose_ref,
    const uint8_t* __restrict__ roi_mask,
    const OffsetEntry* __restrict__ offsets,
    int n_offsets,
    float* gamma_map,
    const GammaParams params,
    unsigned int* global_n_eval,
    unsigned int* global_n_passed,
    double* global_sum_gamma,
    unsigned int* global_histogram
) {
    // Global voxel indices
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Bounds check
    if (i >= params.nx || j >= params.ny || k >= params.nz) return;

    // Shared accumulators
    __shared__ unsigned int s_n_eval;
    __shared__ unsigned int s_n_pass;
    __shared__ double s_sum_gamma;
    __shared__ unsigned int s_hist[HIST_BINS];

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        s_n_eval = 0;
        s_n_pass = 0;
        s_sum_gamma = 0.0;
        for (int b = 0; b < HIST_BINS; ++b) s_hist[b] = 0;
    }
    __syncthreads();

    // Linear index
    const int idx = i + params.nx * (j + params.ny * k);

    // ROI mask check
    if (roi_mask && roi_mask[idx] == 0) {
        if (gamma_map) gamma_map[idx] = -1.0f;
        return;
    }

    const float D_ref_local = dose_ref[idx];

    // Compute threshold
    const float threshold = (params.dose_threshold_percent / 100.0f) * params.global_dose;
    if (D_ref_local < threshold) {
        if (gamma_map) gamma_map[idx] = -1.0f;  // Mark as not evaluated
        return;
    }

    const float dd_factor = params.dd_percent / 100.0f;
    const float delta_D = params.local_normalization
        ? (dd_factor * D_ref_local)
        : (dd_factor * params.global_dose);

    if (delta_D <= 0.0f) {
        if (gamma_map) gamma_map[idx] = params.max_gamma;
        return;
    }

    const float delta_D_sq = delta_D * delta_D;
    float min_gamma_sq = 1e30f;

    // Iterate precomputed offsets
    for (int t = 0; t < n_offsets; ++t) {
        const OffsetEntry off = offsets[t];

        float fx = (float)i + off.ox;
        float fy = (float)j + off.oy;
        float fz = (float)k + off.oz;

        // Bounds check
        if (fx < 0.f || fx > params.nx - 1 ||
            fy < 0.f || fy > params.ny - 1 ||
            fz < 0.f || fz > params.nz - 1) {
            continue;
        }

        const float D_eval_at_offset = sample_trilinear(dose_eval, params.nx, params.ny, params.nz, fx, fy, fz);

        const float dose_diff = D_eval_at_offset - D_ref_local;
        const float dose_term = (dose_diff * dose_diff) / delta_D_sq;
        const float gamma_sq = off.spatial_term + dose_term;

        if (gamma_sq < min_gamma_sq) {
            min_gamma_sq = gamma_sq;
            if (min_gamma_sq <= 1.0f) break;  // Early exit
        }
    }

    float gamma_val = sqrtf(min_gamma_sq);
    if (gamma_val > params.max_gamma) gamma_val = params.max_gamma;

    if (gamma_map) gamma_map[idx] = gamma_val;

    // Accumulate shared statistics
    atomicAdd(&s_n_eval, 1u);
    if (gamma_val <= 1.0f) atomicAdd(&s_n_pass, 1u);
    atomicAdd(&s_sum_gamma, (double)gamma_val);

    int bin = (int)(gamma_val / params.max_gamma * 100.0f);
    bin = max(0, min(bin, HIST_BINS - 1));
    atomicAdd(&s_hist[bin], 1u);

    __syncthreads();

    // Flush block aggregates to global
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        atomicAdd(global_n_eval, s_n_eval);
        atomicAdd(global_n_passed, s_n_pass);
        atomicAdd(global_sum_gamma, s_sum_gamma);
        for (int b = 0; b < HIST_BINS; ++b) {
            if (s_hist[b] > 0) atomicAdd(&global_histogram[b], s_hist[b]);
        }
    }
}


/**
 * @brief Host function to launch gamma computation
 */
void gamma_3d_cuda(
    const float* dose_eval,
    const float* dose_ref,
    const uint8_t* roi_mask,
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

    // Build search offsets (host)
    const float search_mm = params.dta_mm * params.max_gamma;
    const float samp = params.sampling > 0.f ? params.sampling : 1.0f;
    const float step_x = params.sx * samp;
    const float step_y = params.sy * samp;
    const float step_z = params.sz * samp;

    const int max_ix = (int)ceilf(search_mm / step_x);
    const int max_iy = (int)ceilf(search_mm / step_y);
    const int max_iz = (int)ceilf(search_mm / step_z);

    std::vector<OffsetEntry> offsets;
    offsets.reserve((size_t)(2 * max_ix + 1) * (2 * max_iy + 1) * (2 * max_iz + 1));

    for (int kz = -max_iz; kz <= max_iz; ++kz) {
        const float dz_mm = kz * step_z;
        for (int jy = -max_iy; jy <= max_iy; ++jy) {
            const float dy_mm = jy * step_y;
            for (int ix = -max_ix; ix <= max_ix; ++ix) {
                const float dx_mm = ix * step_x;
                const float dist_sq = dx_mm * dx_mm + dy_mm * dy_mm + dz_mm * dz_mm;
                if (dist_sq <= search_mm * search_mm + 1e-6f) {
                    OffsetEntry e;
                    e.ox = dx_mm / params.sx;
                    e.oy = dy_mm / params.sy;
                    e.oz = dz_mm / params.sz;
                    e.spatial_term = params.dta_mm > 0.f ? (dist_sq / (params.dta_mm * params.dta_mm)) : 0.f;
                    offsets.push_back(e);
                }
            }
        }
    }

    // Sort by spatial_term (equivalent to distance)
    std::sort(offsets.begin(), offsets.end(), [](const OffsetEntry& a, const OffsetEntry& b) {
        return a.spatial_term < b.spatial_term;
    });
    if (offsets.empty()) {
        OffsetEntry e{0.f, 0.f, 0.f, 0.f};
        offsets.push_back(e);
    }
    const int n_offsets = (int)offsets.size();
    
    // Allocate device memory
    float *d_dose_eval = nullptr, *d_dose_ref = nullptr, *d_gamma_map = nullptr;
    unsigned int *d_n_eval = nullptr, *d_n_passed = nullptr;
    double *d_sum_gamma = nullptr;
    unsigned int *d_histogram = nullptr;
    OffsetEntry *d_offsets = nullptr;
    uint8_t *d_roi = nullptr;
    
    cudaError_t err;
    
    err = cudaMalloc(&d_dose_eval, bytes);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMalloc(&d_dose_ref, bytes);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    err = cudaMalloc(&d_offsets, n_offsets * sizeof(OffsetEntry));
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
    
    err = cudaMalloc(&d_histogram, HIST_BINS * sizeof(unsigned int));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    // Copy data to device
    err = cudaMemcpyAsync(d_dose_eval, dose_eval, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemcpyAsync(d_dose_ref, dose_ref, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    err = cudaMemcpyAsync(d_offsets, offsets.data(), n_offsets * sizeof(OffsetEntry), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    if (roi_mask) {
        err = cudaMalloc(&d_roi, n_voxels * sizeof(uint8_t));
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        err = cudaMemcpyAsync(d_roi, roi_mask, n_voxels * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Initialize counters to zero
    err = cudaMemsetAsync(d_n_eval, 0, sizeof(unsigned int), stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemsetAsync(d_n_passed, 0, sizeof(unsigned int), stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemsetAsync(d_sum_gamma, 0, sizeof(double), stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemsetAsync(d_histogram, 0, HIST_BINS * sizeof(unsigned int), stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    // Launch kernel
    dim3 block(GAMMA_BLOCK_X, GAMMA_BLOCK_Y, GAMMA_BLOCK_Z);
    dim3 grid(
        (params.nx + block.x - 1) / block.x,
        (params.ny + block.y - 1) / block.y,
        (params.nz + block.z - 1) / block.z
    );
    
    gamma_kernel<<<grid, block, 0, stream>>>(
        d_dose_eval, d_dose_ref, d_roi, d_offsets, n_offsets, d_gamma_map, params,
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
    
    err = cudaMemcpyAsync(stats->histogram, d_histogram, HIST_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    if (gamma_map && d_gamma_map) {
        err = cudaMemcpyAsync(gamma_map, d_gamma_map, bytes, cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }
    
    // Synchronize
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    stats->sum_gamma_sq = 0.0; // not accumulated in this kernel
    
    // Cleanup
    cudaFree(d_dose_eval);
    cudaFree(d_dose_ref);
    if (d_gamma_map) cudaFree(d_gamma_map);
    cudaFree(d_n_eval);
    cudaFree(d_n_passed);
    cudaFree(d_sum_gamma);
    cudaFree(d_histogram);
    cudaFree(d_offsets);
    if (d_roi) cudaFree(d_roi);
}


/**
 * @brief Check CUDA availability
 */
bool gamma_cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}
