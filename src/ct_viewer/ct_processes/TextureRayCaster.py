"""
Texture-Accelerated Ray Casting Kernels

This module provides ray casting kernels that leverage CUDA texture memory
for hardware-accelerated trilinear interpolation. These kernels are designed
to be drop-in replacements for the Numba-based kernels in ray_cast_kernels.py.

Key advantages of texture-based ray casting:
1. Hardware trilinear interpolation (no manual 8-point sampling)
2. Texture cache optimized for spatial locality (ray marching pattern)
3. Automatic boundary handling (clamp/border modes)
4. Reduced register pressure (tex3D is a single instruction)

Usage:
    from ray_cast_kernels_texture import (
        TextureRayCaster,
        ray_cast_texture_kernel_code,
    )
    
    # Create texture-based ray caster
    caster = TextureRayCaster(volume_gpu)
    
    # Cast rays
    caster.cast_rays(
        point_coords_gpu,
        directions_gpu,
        intersection_points_gpu,
        intersection_distances_gpu,
        intersection_values_gpu,
        threshold=600.0,
        max_steps=150,
        step_sizes=[1.0, 0.1, 0.01],
        max_distances_gpu=max_distances_gpu
    )
    
    # Clean up
    caster.destroy()
"""

import cupy as cp
import numpy as np
from numba import float32, int32
from typing import Optional, List, Tuple, Union

# Import TextureObject3D from the resample module
# In production, this would be in a shared module

# =============================================================================
# RAW CUDA KERNEL CODE
# =============================================================================

# Main ray casting kernel with texture-based trilinear interpolation
ray_cast_texture_kernel_code = r'''
extern "C" __global__ void ray_cast_texture_kernel(
    cudaTextureObject_t volume_tex,
    const float* __restrict__ point_coords,     // (N, 3) flattened
    const int* __restrict__ point_mask,         // (N,) boolean mask as int
    const float* __restrict__ directions,       // (M, 3) flattened
    float* __restrict__ intersection_points,    // (N, M, 3) flattened
    float* __restrict__ intersection_distances, // (N, M) flattened
    float* __restrict__ intersection_values,    // (N, M) flattened
    const float threshold,
    const int max_steps,
    const float* __restrict__ step_sizes,       // Array of adaptive step sizes
    const int num_step_sizes,
    const float* __restrict__ max_distances,    // (N,) per-point max distances
    const int num_points,
    const int num_dirs
) {
    // 2D grid: x = points, y = directions
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dir_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (point_idx >= num_points || dir_idx >= num_dirs) return;
    if (point_mask[point_idx] == 0) return;
    
    // Load direction vector (coalesced read across threads in y-dimension)
    const int dir_offset = dir_idx * 3;
    const float dir_d = directions[dir_offset + 0];
    const float dir_h = directions[dir_offset + 1];
    const float dir_w = directions[dir_offset + 2];
    
    // Load starting point
    const int point_offset = point_idx * 3;
    float current_d = point_coords[point_offset + 0];
    float current_h = point_coords[point_offset + 1];
    float current_w = point_coords[point_offset + 2];
    
    // Check initial value - if already below threshold, skip
    // tex3D samples at (w + 0.5, h + 0.5, d + 0.5) for voxel centers
    float initial_value = tex3D<float>(volume_tex, 
                                        current_w + 0.5f, 
                                        current_h + 0.5f, 
                                        current_d + 0.5f);
    if (initial_value <= threshold) return;
    
    // Load max distance for this point
    const float max_dist = max_distances[point_idx];
    const float max_dist_sq = max_dist * max_dist;
    
    // Initialize step tracking
    int step_idx = 0;
    float current_step = step_sizes[0];
    
    float step_d = current_step * dir_d;
    float step_h = current_step * dir_h;
    float step_w = current_step * dir_w;
    
    // Accumulated shift from starting point
    float shift_d = 0.0f;
    float shift_h = 0.0f;
    float shift_w = 0.0f;
    
    // Ray marching loop
    for (int iter = 0; iter < max_steps; iter++) {
        // Advance position
        shift_d += step_d;
        shift_h += step_h;
        shift_w += step_w;
        
        current_d += step_d;
        current_h += step_h;
        current_w += step_w;
        
        // Sample volume using hardware trilinear interpolation
        // tex3D coordinate order: (x=width, y=height, z=depth)
        float value = tex3D<float>(volume_tex,
                                   current_w + 0.5f,
                                   current_h + 0.5f,
                                   current_d + 0.5f);
        
        // Compute squared distance
        float dist_sq = shift_d * shift_d + shift_h * shift_h + shift_w * shift_w;
        
        // Check termination conditions
        if (value <= threshold || dist_sq >= max_dist_sq) {
            // Try finer step size if available
            if (step_idx < num_step_sizes - 1) {
                // Backtrack
                shift_d -= step_d;
                shift_h -= step_h;
                shift_w -= step_w;
                
                current_d -= step_d;
                current_h -= step_h;
                current_w -= step_w;
                
                // Use finer step
                step_idx++;
                current_step = step_sizes[step_idx];
                step_d = current_step * dir_d;
                step_h = current_step * dir_h;
                step_w = current_step * dir_w;
                continue;
            } else {
                // No more refinement possible, ray terminates without valid intersection
                return;
            }
        }
        
        // Valid intersection found - update outputs
        // Output indexing: [point_idx, dir_idx, ...] in row-major order
        const int out_idx_2d = point_idx * num_dirs + dir_idx;
        const int out_idx_3d = out_idx_2d * 3;
        
        intersection_values[out_idx_2d] = value;
        intersection_distances[out_idx_2d] = sqrtf(dist_sq);
        intersection_points[out_idx_3d + 0] = current_d;
        intersection_points[out_idx_3d + 1] = current_h;
        intersection_points[out_idx_3d + 2] = current_w;
    }
}
'''

# Shared memory optimized version - loads directions into shared memory
ray_cast_texture_kernel_shared_code = r'''
#define TILE_SIZE_DIRS 32

extern "C" __global__ void ray_cast_texture_kernel_shared(
    cudaTextureObject_t volume_tex,
    const float* __restrict__ point_coords,
    const int* __restrict__ point_mask,
    const float* __restrict__ directions,
    float* __restrict__ intersection_points,
    float* __restrict__ intersection_distances,
    float* __restrict__ intersection_values,
    const float threshold,
    const int max_steps,
    const float* __restrict__ step_sizes,
    const int num_step_sizes,
    const float* __restrict__ max_distances,
    const int num_points,
    const int num_dirs
) {
    // Shared memory for direction vectors
    __shared__ float shared_dirs[TILE_SIZE_DIRS][3];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int point_idx = blockIdx.x * blockDim.x + tx;
    const int dir_idx = blockIdx.y * blockDim.y + ty;
    
    // Cooperative loading of directions into shared memory
    const int local_dir_idx = blockIdx.y * blockDim.y + ty;
    if (tx == 0 && local_dir_idx < num_dirs) {
        shared_dirs[ty][0] = directions[local_dir_idx * 3 + 0];
        shared_dirs[ty][1] = directions[local_dir_idx * 3 + 1];
        shared_dirs[ty][2] = directions[local_dir_idx * 3 + 2];
    }
    __syncthreads();
    
    if (point_idx >= num_points || dir_idx >= num_dirs) return;
    if (point_mask[point_idx] == 0) return;
    
    // Load direction from shared memory
    const float dir_d = shared_dirs[ty][0];
    const float dir_h = shared_dirs[ty][1];
    const float dir_w = shared_dirs[ty][2];
    
    // Load starting point
    const int point_offset = point_idx * 3;
    float current_d = point_coords[point_offset + 0];
    float current_h = point_coords[point_offset + 1];
    float current_w = point_coords[point_offset + 2];
    
    // Check initial value
    float initial_value = tex3D<float>(volume_tex,
                                        current_w + 0.5f,
                                        current_h + 0.5f,
                                        current_d + 0.5f);
    if (initial_value <= threshold) return;
    
    const float max_dist = max_distances[point_idx];
    const float max_dist_sq = max_dist * max_dist;
    
    int step_idx = 0;
    float current_step = step_sizes[0];
    
    float step_d = current_step * dir_d;
    float step_h = current_step * dir_h;
    float step_w = current_step * dir_w;
    
    float shift_d = 0.0f;
    float shift_h = 0.0f;
    float shift_w = 0.0f;
    
    for (int iter = 0; iter < max_steps; iter++) {
        shift_d += step_d;
        shift_h += step_h;
        shift_w += step_w;
        
        current_d += step_d;
        current_h += step_h;
        current_w += step_w;
        
        float value = tex3D<float>(volume_tex,
                                   current_w + 0.5f,
                                   current_h + 0.5f,
                                   current_d + 0.5f);
        
        float dist_sq = shift_d * shift_d + shift_h * shift_h + shift_w * shift_w;
        
        if (value <= threshold || dist_sq >= max_dist_sq) {
            if (step_idx < num_step_sizes - 1) {
                shift_d -= step_d;
                shift_h -= step_h;
                shift_w -= step_w;
                
                current_d -= step_d;
                current_h -= step_h;
                current_w -= step_w;
                
                step_idx++;
                current_step = step_sizes[step_idx];
                step_d = current_step * dir_d;
                step_h = current_step * dir_h;
                step_w = current_step * dir_w;
                continue;
            } else {
                return;
            }
        }
        
        const int out_idx_2d = point_idx * num_dirs + dir_idx;
        const int out_idx_3d = out_idx_2d * 3;
        
        intersection_values[out_idx_2d] = value;
        intersection_distances[out_idx_2d] = sqrtf(dist_sq);
        intersection_points[out_idx_3d + 0] = current_d;
        intersection_points[out_idx_3d + 1] = current_h;
        intersection_points[out_idx_3d + 2] = current_w;
    }
}
'''


difference_branchless_kernel_code = r'''
// ===========================================================================
// Atomic float min/max helpers
// CUDA lacks native atomicMin/atomicMax for float.
// Uses atomicCAS with int reinterpretation.
// ===========================================================================
__device__ float atomicMin_float(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_f = __int_as_float(expected);
        if (old_f <= val) break;
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (old != expected);
    return __int_as_float(old);
}

__device__ float atomicMax_float(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_f = __int_as_float(expected);
        if (old_f >= val) break;
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (old != expected);
    return __int_as_float(old);
}

// ===========================================================================
// Branchless difference profiling kernel with tex3D + warp reduction
// ===========================================================================
extern "C" __global__ void ray_cast_difference_branchless_texture(
    cudaTextureObject_t volume_tex,
    const float* __restrict__ point_coords,         // (N, 3) flattened
    const int*   __restrict__ point_mask,            // (N,) boolean as int
    const float* __restrict__ directions,            // (M, 3) flattened
    const float* __restrict__ checkpoint_distances,  // (K,) sorted
    float* __restrict__ difference_neg_output,       // (N, K) flattened
    float* __restrict__ difference_pos_output,       // (N, K) flattened
    const float checkpoint_reset_f,
    const int   num_steps_per_interval,
    const long long iter_number,
    const int   num_points,
    const int   num_dirs,
    const int   num_checkpoints
) {
    // Thread mapping: x = directions (fast dim, warp-aligned), y = points
    const int dir_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int point_idx = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int lane = threadIdx.x;
    const unsigned int FULL_MASK = 0xFFFFFFFF;

    // Active flag — inactive threads still participate in shuffles
    // but contribute identity values (0.0)
    const bool active = (point_idx < num_points &&
                         dir_idx < num_dirs &&
                         point_mask[point_idx] != 0);

    // -------------------------------------------------------------------
    // Load direction
    // -------------------------------------------------------------------
    float dir_d = 0.0f, dir_h = 0.0f, dir_w = 0.0f;
    if (active) {
        const int dir_offset = dir_idx * 3;
        dir_d = directions[dir_offset + 0];
        dir_h = directions[dir_offset + 1];
        dir_w = directions[dir_offset + 2];
    }

    // -------------------------------------------------------------------
    // Jitter — splitmix64 inline, deterministic per (point, dir, iter)
    // -------------------------------------------------------------------
    float jitter = 0.0f;

    if (active) {
        unsigned long long seed = (unsigned long long)point_idx ^
                                  ((unsigned long long)dir_idx << 32) ^
                                  (unsigned long long)iter_number;
        seed = (seed + 0x9E3779B97F4A7C15ULL) & 0xFFFFFFFFFFFFFFFFULL;
        seed = ((seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9ULL) & 0xFFFFFFFFFFFFFFFFULL;
        seed = ((seed ^ (seed >> 27)) * 0x94D049BB133111EBULL) & 0xFFFFFFFFFFFFFFFFULL;
        seed = seed ^ (seed >> 31);

        float first_step = checkpoint_distances[0] / (float)num_steps_per_interval;
        jitter = ((float)(seed & 0xFFFFFF) / (float)0xFFFFFF) * first_step;
    }

    // -------------------------------------------------------------------
    // Starting position + initial sample
    // -------------------------------------------------------------------
    float current_d = 0.0f, current_h = 0.0f, current_w = 0.0f;
    float initial_value = 0.0f;
    float source_value  = 0.0f;

    if (active) {
        const int pt_offset = point_idx * 3;
        current_d = point_coords[pt_offset + 0] + jitter * dir_d;
        current_h = point_coords[pt_offset + 1] + jitter * dir_h;
        current_w = point_coords[pt_offset + 2] + jitter * dir_w;

        initial_value = tex3D<float>(volume_tex,
                                     current_w + 0.5f,
                                     current_h + 0.5f,
                                     current_d + 0.5f);
        source_value = initial_value;
    }

    // -------------------------------------------------------------------
    // March through checkpoints
    // -------------------------------------------------------------------
    float accumulated_distance = 0.0f;
    float neg_diff = 0.0f;
    float pos_diff = 0.0f;
    const float keep_f = 1.0f - checkpoint_reset_f;

    for (int ck = 0; ck < num_checkpoints; ck++) {

        float target_distance = 0.0f;
        float step_size = 0.0f;
        if (active) {
            target_distance = checkpoint_distances[ck];
            float interval_length = target_distance - accumulated_distance;
            step_size = interval_length / (float)num_steps_per_interval;
        }

        // Uniform march
        for (int s = 0; s < num_steps_per_interval; s++) {
            if (active) {
                current_d += step_size * dir_d;
                current_h += step_size * dir_h;
                current_w += step_size * dir_w;

                source_value = tex3D<float>(volume_tex,
                                            current_w + 0.5f,
                                            current_h + 0.5f,
                                            current_d + 0.5f);

                float test_diff = initial_value - source_value;
                neg_diff = fminf(neg_diff, test_diff);
                pos_diff = fmaxf(pos_diff, test_diff);
            }
        }

        if (active) {
            accumulated_distance = target_distance;
        }

        // ---------------------------------------------------------------
        // Warp-level reduction across directions (threadIdx.x)
        // ---------------------------------------------------------------
        float warp_neg = neg_diff;
        float warp_pos = pos_diff;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_neg = fminf(warp_neg, __shfl_down_sync(FULL_MASK, warp_neg, offset));
            warp_pos = fmaxf(warp_pos, __shfl_down_sync(FULL_MASK, warp_pos, offset));
        }

        // Lane 0 writes reduced result via atomic
        if (lane == 0 && point_idx < num_points) {
            const int out_idx = point_idx * num_checkpoints + ck;
            atomicMin_float(&difference_neg_output[out_idx], warp_neg);
            atomicMax_float(&difference_pos_output[out_idx], warp_pos);
        }

        // ---------------------------------------------------------------
        // Branchless checkpoint reset
        // ---------------------------------------------------------------
        if (active) {
            initial_value = checkpoint_reset_f * source_value + keep_f * initial_value;
            neg_diff = keep_f * neg_diff;
            pos_diff = keep_f * pos_diff;
        }
    }
}
'''


gradient_branchless_kernel_code = r'''
// ===========================================================================
// Atomic float min/max helpers (same as difference kernel)
// ===========================================================================
__device__ float atomicMin_float(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_f = __int_as_float(expected);
        if (old_f <= val) break;
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (old != expected);
    return __int_as_float(old);
}

__device__ float atomicMax_float(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_f = __int_as_float(expected);
        if (old_f >= val) break;
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (old != expected);
    return __int_as_float(old);
}

// ===========================================================================
// Branchless Gaussian-weighted gradient profiling kernel
// ===========================================================================
extern "C" __global__ void ray_cast_gradient_branchless_texture(
    cudaTextureObject_t volume_tex,
    const float* __restrict__ point_coords,         // (N, 3) flattened
    const int*   __restrict__ point_mask,            // (N,) boolean as int
    const float* __restrict__ directions,            // (M, 3) flattened
    const float* __restrict__ checkpoint_distances,  // (K,) sorted
    const float* __restrict__ weight_table,          // (K * S,) precomputed weights
    float* __restrict__ gradient_neg_output,         // (N, K) flattened
    float* __restrict__ gradient_pos_output,         // (N, K) flattened
    const float checkpoint_reset_f,
    const int   num_steps_per_interval,
    const long long iter_number,
    const int   num_points,
    const int   num_dirs,
    const int   num_checkpoints
) {
    // -------------------------------------------------------------------
    // Shared memory: weight table
    // Dynamically sized via extern — launched with shared_mem_bytes
    // -------------------------------------------------------------------
    extern __shared__ float shared_weights[];

    // Thread mapping: x = directions (warp-aligned), y = points
    const int dir_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int point_idx = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int lane = threadIdx.x;
    const unsigned int FULL_MASK = 0xFFFFFFFF;

    // -------------------------------------------------------------------
    // Cooperative load of weight table into shared memory
    // -------------------------------------------------------------------
    const int table_size = num_checkpoints * num_steps_per_interval;
    const int thread_linear = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;
    for (int i = thread_linear; i < table_size; i += block_size) {
        shared_weights[i] = weight_table[i];
    }
    __syncthreads();

    // -------------------------------------------------------------------
    // Active flag
    // -------------------------------------------------------------------
    const bool active = (point_idx < num_points &&
                         dir_idx < num_dirs &&
                         point_mask[point_idx] != 0);

    // -------------------------------------------------------------------
    // Load direction
    // -------------------------------------------------------------------
    float dir_d = 0.0f, dir_h = 0.0f, dir_w = 0.0f;
    if (active) {
        const int dir_offset = dir_idx * 3;
        dir_d = directions[dir_offset + 0];
        dir_h = directions[dir_offset + 1];
        dir_w = directions[dir_offset + 2];
    }

    // -------------------------------------------------------------------
    // Jitter — splitmix64, deterministic per (point, dir, iter)
    // -------------------------------------------------------------------
    float jitter = 0.0f;
    if (active) {
        unsigned long long seed = (unsigned long long)point_idx ^
                                  ((unsigned long long)dir_idx << 32) ^
                                  (unsigned long long)iter_number;
        seed = (seed + 0x9E3779B97F4A7C15ULL) & 0xFFFFFFFFFFFFFFFFULL;
        seed = ((seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9ULL) & 0xFFFFFFFFFFFFFFFFULL;
        seed = ((seed ^ (seed >> 27)) * 0x94D049BB133111EBULL) & 0xFFFFFFFFFFFFFFFFULL;
        seed = seed ^ (seed >> 31);

        float first_step = checkpoint_distances[0] / (float)num_steps_per_interval;
        jitter = ((float)(seed & 0xFFFFFF) / (float)0xFFFFFF) * first_step;
    }

    // -------------------------------------------------------------------
    // Starting position + initial sample
    // -------------------------------------------------------------------
    float current_d = 0.0f, current_h = 0.0f, current_w = 0.0f;
    float initial_value = 0.0f;
    float source_value  = 0.0f;

    if (active) {
        const int pt_offset = point_idx * 3;
        current_d = point_coords[pt_offset + 0] + jitter * dir_d;
        current_h = point_coords[pt_offset + 1] + jitter * dir_h;
        current_w = point_coords[pt_offset + 2] + jitter * dir_w;

        initial_value = tex3D<float>(volume_tex,
                                     current_w + 0.5f,
                                     current_h + 0.5f,
                                     current_d + 0.5f);
        source_value = initial_value;
    }

    // -------------------------------------------------------------------
    // March through checkpoints
    // -------------------------------------------------------------------
    float accumulated_distance = 0.0f;
    float grad_distance = 0.0f;
    float weighted_sum = 0.0f;
    float weight_total = 0.0f;
    float neg_grad = 0.0f;
    float pos_grad = 0.0f;
    const float keep_f = 1.0f - checkpoint_reset_f;

    for (int ck = 0; ck < num_checkpoints; ck++) {

        float target_distance = 0.0f;
        float step_size = 0.0f;
        float interval_length = 0.0f;
        if (active) {
            target_distance = checkpoint_distances[ck];
            interval_length = target_distance - accumulated_distance;
            step_size = interval_length / (float)num_steps_per_interval;
            grad_distance += interval_length;
        }

        // Weight table base for this checkpoint
        const int weight_base = ck * num_steps_per_interval;

        // Uniform march with weighted accumulation
        for (int s = 0; s < num_steps_per_interval; s++) {
            if (active) {
                current_d += step_size * dir_d;
                current_h += step_size * dir_h;
                current_w += step_size * dir_w;

                source_value = tex3D<float>(volume_tex,
                                            current_w + 0.5f,
                                            current_h + 0.5f,
                                            current_d + 0.5f);

                float w = shared_weights[weight_base + s];
                weighted_sum += w * source_value;
                weight_total += w;
            }
        }

        // Compute gradient for this checkpoint
        float grad = 0.0f;
        if (active) {
            accumulated_distance = target_distance;

            // weight_total is always > 0 (weights are precomputed positive)
            // interval_length is always > 0 (checkpoints are sorted positive)
            float weighted_value = weighted_sum / weight_total;
            grad = (weighted_value - initial_value) / grad_distance;

            neg_grad = fminf(neg_grad, grad);
            pos_grad = fmaxf(pos_grad, grad);
        }

        // ---------------------------------------------------------------
        // Warp-level reduction across directions (threadIdx.x)
        // ---------------------------------------------------------------
        float warp_neg = neg_grad;
        float warp_pos = pos_grad;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_neg = fminf(warp_neg, __shfl_down_sync(FULL_MASK, warp_neg, offset));
            warp_pos = fmaxf(warp_pos, __shfl_down_sync(FULL_MASK, warp_pos, offset));
        }

        // Lane 0 writes reduced result via atomic
        if (lane == 0 && point_idx < num_points) {
            const int out_idx = point_idx * num_checkpoints + ck;
            atomicMin_float(&gradient_neg_output[out_idx], warp_neg);
            atomicMax_float(&gradient_pos_output[out_idx], warp_pos);
        }

        // ---------------------------------------------------------------
        // Branchless checkpoint reset
        //
        // reset_f = 1.0: initial_value = weighted_value, clear accumulators
        // reset_f = 0.0: keep everything cumulative
        // ---------------------------------------------------------------
        if (active) {
            float weighted_value = weighted_sum / weight_total;
            initial_value = checkpoint_reset_f * weighted_value + keep_f * initial_value;
            weighted_sum *= keep_f;
            weight_total *= keep_f;
            neg_grad *= keep_f;
            pos_grad *= keep_f;
            grad_distance *= keep_f;
        }
    }
}
'''


intensity_branchless_kernel_code = r'''
// ===========================================================================
// Branchless per-interval intensity integration kernel
// tex3D + warp-level sum reduction + atomicAdd
//
// For each (point, direction) pair, marches along the ray through K
// checkpoint intervals using uniform sub-stepping. Within each interval,
// accumulates I(r)*r² via trapezoidal rule. At the end of each interval,
// a warp-level sum reduction collapses the per-direction contributions,
// and lane 0 atomically adds the omega-weighted result to the output.
//
// Thread mapping:
//   x-dimension = directions (warp-aligned, 32 threads for __shfl_down_sync)
//   y-dimension = points
//
// Output:
//   intensity_output[point, checkpoint] = sigma_rays { omega_m * Integral_{r_{k-1}}^{r_k} I(r)*r² dr }
//
// This is a PER-INTERVAL integral. The host can cumsum if cumulative is needed.
// ===========================================================================
extern "C" __global__ void ray_cast_intensity_branchless_texture(
    cudaTextureObject_t volume_tex,
    const float* __restrict__ point_coords,         // (N, 3) flattened
    const int*   __restrict__ point_mask,            // (N,) boolean as int
    const float* __restrict__ directions,            // (M, 3) flattened
    const float* __restrict__ ray_omega,             // (M,) solid angle per direction
    const float* __restrict__ checkpoint_distances,  // (K,) sorted
    float* __restrict__ intensity_output,            // (N, K) flattened — OUTPUT
    const int   num_steps_per_interval,
    const long long iter_number,
    const int   num_points,
    const int   num_dirs,
    const int   num_checkpoints
) {
    // Thread mapping: x = directions (fast dim, warp-aligned), y = points
    const int dir_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int point_idx = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int lane = threadIdx.x;
    const unsigned int FULL_MASK = 0xFFFFFFFF;

    // Active flag — inactive threads still participate in shuffles
    // but contribute identity value (0.0 for addition)
    const bool active = (point_idx < num_points &&
                         dir_idx < num_dirs &&
                         point_mask[point_idx] != 0);

    // -------------------------------------------------------------------
    // Load direction and omega
    // -------------------------------------------------------------------
    float dir_d = 0.0f, dir_h = 0.0f, dir_w = 0.0f;
    float omega = 0.0f;
    if (active) {
        const int dir_offset = dir_idx * 3;
        dir_d = directions[dir_offset + 0];
        dir_h = directions[dir_offset + 1];
        dir_w = directions[dir_offset + 2];
        omega = ray_omega[dir_idx];
    }

    // -------------------------------------------------------------------
    // Jitter — splitmix64 inline, deterministic per (point, dir, iter)
    // -------------------------------------------------------------------
    float jitter = 0.0f;
    if (active) {
        unsigned long long seed = (unsigned long long)point_idx ^
                                  ((unsigned long long)dir_idx << 32) ^
                                  (unsigned long long)iter_number;
        seed = (seed + 0x9E3779B97F4A7C15ULL) & 0xFFFFFFFFFFFFFFFFULL;
        seed = ((seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9ULL) & 0xFFFFFFFFFFFFFFFFULL;
        seed = ((seed ^ (seed >> 27)) * 0x94D049BB133111EBULL) & 0xFFFFFFFFFFFFFFFFULL;
        seed = seed ^ (seed >> 31);

        float first_step = checkpoint_distances[0] / (float)num_steps_per_interval;
        jitter = ((float)(seed & 0xFFFFFF) / (float)0xFFFFFF) * first_step;
    }

    // -------------------------------------------------------------------
    // Starting position + initial sample
    // -------------------------------------------------------------------
    float current_d = 0.0f, current_h = 0.0f, current_w = 0.0f;
    float source_value = 0.0f;

    if (active) {
        const int pt_offset = point_idx * 3;
        current_d = point_coords[pt_offset + 0] + jitter * dir_d;
        current_h = point_coords[pt_offset + 1] + jitter * dir_h;
        current_w = point_coords[pt_offset + 2] + jitter * dir_w;

        source_value = tex3D<float>(volume_tex,
                                     current_w + 0.5f,
                                     current_h + 0.5f,
                                     current_d + 0.5f);
    }

    // -------------------------------------------------------------------
    // March through checkpoints — per-interval trapezoidal integration
    // -------------------------------------------------------------------
    float accumulated_distance = 0.0f;

    // prev_weighted = I(r) * r² at the start of the current interval
    // For the first interval, r = 0 (before any stepping), so prev_weighted = 0
    float prev_weighted = 0.0f;

    for (int ck = 0; ck < num_checkpoints; ck++) {

        float target_distance = 0.0f;
        float step_size = 0.0f;
        if (active) {
            target_distance = checkpoint_distances[ck];
            float interval_length = target_distance - accumulated_distance;
            step_size = interval_length / (float)num_steps_per_interval;
        }

        // Per-interval accumulator (reset each checkpoint)
        float interval_integral = 0.0f;

        // At the start of this interval, prev_weighted holds the value
        // from the end of the previous interval (or 0.0 for ck=0).
        // This provides continuity for trapezoidal rule across intervals.

        // Uniform march within this interval
        for (int s = 0; s < num_steps_per_interval; s++) {
            if (active) {
                current_d += step_size * dir_d;
                current_h += step_size * dir_h;
                current_w += step_size * dir_w;
                accumulated_distance += step_size;

                source_value = tex3D<float>(volume_tex,
                                            current_w + 0.5f,
                                            current_h + 0.5f,
                                            current_d + 0.5f);

                // Trapezoidal rule: 0.5 * dr * (f(r_n)*r_n² + f(r_{n+1})*r_{n+1}²)
                float next_weighted = source_value * accumulated_distance * accumulated_distance;
                interval_integral += 0.5f * step_size * (prev_weighted + next_weighted);
                prev_weighted = next_weighted;
            }
        }

        // Snap accumulated_distance to the exact checkpoint value
        // (avoids floating-point drift over many intervals)
        if (active) {
            accumulated_distance = target_distance;
        }

        // ---------------------------------------------------------------
        // Warp-level SUM reduction across directions (threadIdx.x)
        // ---------------------------------------------------------------
        float warp_sum = interval_integral * omega;  // weight by solid angle

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(FULL_MASK, warp_sum, offset);
        }

        // Lane 0 writes reduced result via native atomicAdd
        if (lane == 0 && point_idx < num_points) {
            const int out_idx = point_idx * num_checkpoints + ck;
            atomicAdd(&intensity_output[out_idx], warp_sum);
        }
    }
}
'''


meanshift_branchless_kernel_code = r'''
// ===========================================================================
// Dual-texture branchless mean-shift ray casting kernel
//
// For each (point, direction) pair, marches uniformly to max_distance,
// sampling both the CT texture (for intensity) and the difference texture
// (for boundary detection). A three-state exit latch tracks the first
// contiguous "inside" region:
//
//   BEFORE (ever_inside=0) -> INSIDE (ever_inside=1) -> EXITED (ever_exited=1)
//
// All transitions are irreversible (monotonic fmaxf). Once EXITED,
// no further accumulation or position tracking occurs, even if the ray
// encounters a second vessel.
//
// Thread mapping: Convention A
//   x-dimension = points    blockIdx.x * blockDim.x + threadIdx.x = point_idx
//   y-dimension = directions blockIdx.y * blockDim.y + threadIdx.y = dir_idx
//
// Outputs (per-ray, consumed by update_medial_axis):
//   intensity_weights[N, M]       — accumulated I(r)·r^2 inside boundary
//   intersection_points[N, M, 3]  — last valid (d, h, w) position
//   intersection_distances[N, M]  — distance to last valid position
//   intersection_values[N, M]     — CT value at last valid position
// ===========================================================================
extern "C" __global__ void ray_cast_meanshift_branchless_texture(
    cudaTextureObject_t ct_tex,
    cudaTextureObject_t diff_tex,
    const float* __restrict__ point_coords,
    const int*   __restrict__ point_mask,
    const float* __restrict__ directions,
    const float* __restrict__ max_distances,
    float* __restrict__ intensity_weights,
    float* __restrict__ intersection_points,
    float* __restrict__ intersection_distances,
    float* __restrict__ intersection_values,
    const float boundary_threshold,
    const int   num_steps,
    const long long iter_number,
    const int   num_points,
    const int   num_dirs
) {
    // Convention A: x = points, y = directions
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dir_idx   = blockIdx.y * blockDim.y + threadIdx.y;

    if (point_idx >= num_points || dir_idx >= num_dirs) return;
    if (point_mask[point_idx] == 0) return;

    // -------------------------------------------------------------------
    // Load direction vector
    // -------------------------------------------------------------------
    const int dir_offset = dir_idx * 3;
    const float dir_d = directions[dir_offset + 0];
    const float dir_h = directions[dir_offset + 1];
    const float dir_w = directions[dir_offset + 2];

    // -------------------------------------------------------------------
    // Load starting point
    // -------------------------------------------------------------------
    const int pt_offset = point_idx * 3;
    const float start_d = point_coords[pt_offset + 0];
    const float start_h = point_coords[pt_offset + 1];
    const float start_w = point_coords[pt_offset + 2];

    // -------------------------------------------------------------------
    // Compute step size from per-point max distance
    // -------------------------------------------------------------------
    const float max_dist = max_distances[point_idx];
    const float step_size = max_dist / (float)num_steps;

    // -------------------------------------------------------------------
    // Jitter — splitmix64 inline, deterministic per (point, dir, iter)
    // -------------------------------------------------------------------
    unsigned long long seed = (unsigned long long)point_idx ^
                              ((unsigned long long)dir_idx << 32) ^
                              (unsigned long long)iter_number;
    seed = (seed + 0x9E3779B97F4A7C15ULL) & 0xFFFFFFFFFFFFFFFFULL;
    seed = ((seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9ULL) & 0xFFFFFFFFFFFFFFFFULL;
    seed = ((seed ^ (seed >> 27)) * 0x94D049BB133111EBULL) & 0xFFFFFFFFFFFFFFFFULL;
    seed = seed ^ (seed >> 31);

    const float jitter = ((float)(seed & 0xFFFFFF) / (float)0xFFFFFF) * step_size;

    // -------------------------------------------------------------------
    // Initialize ray state
    // -------------------------------------------------------------------
    float current_d = start_d + jitter * dir_d;
    float current_h = start_h + jitter * dir_h;
    float current_w = start_w + jitter * dir_w;

    // Sample CT at starting position for last_valid initialization
    float ct_origin = tex3D<float>(ct_tex,
                                    current_w + 0.5f,
                                    current_h + 0.5f,
                                    current_d + 0.5f);

    // Last valid position: initialized to starting point
    float last_valid_d = start_d;
    float last_valid_h = start_h;
    float last_valid_w = start_w;
    float last_valid_dist = 0.0f;
    float last_valid_ct = ct_origin;

    // Trapezoidal integration state
    float accumulated_intensity = 0.0f;
    float accumulated_distance = 0.0f;
    float prev_weighted = 0.0f;

    // Exit latch state (three-state: BEFORE -> INSIDE -> EXITED)
    float ever_inside = 0.0f;
    float ever_exited_after_inside = 0.0f;

    // -------------------------------------------------------------------
    // Uniform march
    // -------------------------------------------------------------------
    for (int s = 0; s < num_steps; s++) {
        current_d += step_size * dir_d;
        current_h += step_size * dir_h;
        current_w += step_size * dir_w;
        accumulated_distance += step_size;

        // Sample both textures
        float ct_value = tex3D<float>(ct_tex,
                                       current_w + 0.5f,
                                       current_h + 0.5f,
                                       current_d + 0.5f);

        float diff_value = tex3D<float>(diff_tex,
                                         current_w + 0.5f,
                                         current_h + 0.5f,
                                         current_d + 0.5f);

        // ---------------------------------------------------------------
        // Exit latch logic (all branchless, all irreversible)
        // ---------------------------------------------------------------
        float raw_inside = (diff_value > boundary_threshold) ? 1.0f : 0.0f;

        // Latch: have we ever been inside?
        ever_inside = fmaxf(ever_inside, raw_inside);

        // Latch: have we exited after being inside?
        float exited_now = ever_inside * (1.0f - raw_inside);
        ever_exited_after_inside = fmaxf(ever_exited_after_inside, exited_now);

        // Final inside flag: true only if currently inside AND never exited
        float inside = raw_inside * (1.0f - ever_exited_after_inside);

        // ---------------------------------------------------------------
        // Last valid position tracking (branchless conditional update)
        // ---------------------------------------------------------------
        float outside = 1.0f - inside;
        last_valid_d    = inside * current_d            + outside * last_valid_d;
        last_valid_h    = inside * current_h            + outside * last_valid_h;
        last_valid_w    = inside * current_w            + outside * last_valid_w;
        last_valid_dist = inside * accumulated_distance + outside * last_valid_dist;
        last_valid_ct   = inside * ct_value             + outside * last_valid_ct;

        // ---------------------------------------------------------------
        // Intensity accumulation (trapezoidal rule, gated by inside)
        // ---------------------------------------------------------------
        float next_weighted = ct_value * accumulated_distance * accumulated_distance;
        float step_contribution = 0.5f * step_size * (prev_weighted + next_weighted);

        accumulated_intensity += inside * step_contribution;

        // Gate trapezoidal state: resets to 0 when outside,
        // so re-entry (which can't happen after latch) would start clean
        prev_weighted = inside * next_weighted;
    }

    // -------------------------------------------------------------------
    // Write outputs
    // -------------------------------------------------------------------
    const int out_idx_2d = point_idx * num_dirs + dir_idx;
    const int out_idx_3d = out_idx_2d * 3;

    intensity_weights[out_idx_2d] = accumulated_intensity;
    intersection_distances[out_idx_2d] = last_valid_dist;
    intersection_values[out_idx_2d] = last_valid_ct;
    intersection_points[out_idx_3d + 0] = last_valid_d;
    intersection_points[out_idx_3d + 1] = last_valid_h;
    intersection_points[out_idx_3d + 2] = last_valid_w;
}
'''


meanshift_inflection_kernel_code = r'''
// ===========================================================================
// Dual-texture branchless mean-shift kernel — inflection point boundary
//
// Same march structure as the zero-crossing variant, but the exit latch
// triggers at the inflection point of the difference profile D(r) along
// the ray. The inflection point is where |dD/dr| peaks — the steepest
// part of the vessel boundary transition.
//
// Detection logic:
//   1. Entry: diff_value > entry_threshold (same as zero-crossing variant)
//   2. While INSIDE, track max |dD/dr| seen so far via fmaxf
//   3. Exit latch triggers when |dD/dr| drops below the running max by a
//      relative fraction (gradient_decay_fraction). This indicates we've
//      passed the inflection point and the transition is flattening.
//
// The entry_threshold ensures we don't trigger on noise gradients before
// entering the vessel. The gradient_decay_fraction controls sensitivity:
//   - 0.5: latch when |dD/dr| drops to half its peak (conservative)
//   - 0.8: latch when |dD/dr| drops to 80% of peak (sensitive, earlier)
//   - 0.95: latch almost immediately after peak (very aggressive)
//
// State machine (same three states, different exit trigger):
//
//   BEFORE  -[diff > entry_threshold]->  INSIDE  -[inflection]->  EXITED
//
// Thread mapping: Convention A (x = points, y = directions)
// ===========================================================================
extern "C" __global__ void ray_cast_meanshift_inflection_texture(
    cudaTextureObject_t ct_tex,
    cudaTextureObject_t diff_tex,
    const float* __restrict__ point_coords,
    const int*   __restrict__ point_mask,
    const float* __restrict__ directions,
    const float* __restrict__ max_distances,
    float* __restrict__ intensity_weights,
    float* __restrict__ intersection_points,
    float* __restrict__ intersection_distances,
    float* __restrict__ intersection_values,
    const float entry_threshold,
    const float gradient_decay_fraction,
    const int   num_steps,
    const long long iter_number,
    const int   num_points,
    const int   num_dirs
) {
    // Convention A: x = points, y = directions
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dir_idx   = blockIdx.y * blockDim.y + threadIdx.y;

    if (point_idx >= num_points || dir_idx >= num_dirs) return;
    if (point_mask[point_idx] == 0) return;

    // -------------------------------------------------------------------
    // Load direction vector
    // -------------------------------------------------------------------
    const int dir_offset = dir_idx * 3;
    const float dir_d = directions[dir_offset + 0];
    const float dir_h = directions[dir_offset + 1];
    const float dir_w = directions[dir_offset + 2];

    // -------------------------------------------------------------------
    // Load starting point
    // -------------------------------------------------------------------
    const int pt_offset = point_idx * 3;
    const float start_d = point_coords[pt_offset + 0];
    const float start_h = point_coords[pt_offset + 1];
    const float start_w = point_coords[pt_offset + 2];

    // -------------------------------------------------------------------
    // Compute step size from per-point max distance
    // -------------------------------------------------------------------
    const float max_dist = max_distances[point_idx];
    const float step_size = max_dist / (float)num_steps;
    const float inv_step_size = (float)num_steps / max_dist;

    // -------------------------------------------------------------------
    // Jitter — splitmix64 inline
    // -------------------------------------------------------------------
    unsigned long long seed = (unsigned long long)point_idx ^
                              ((unsigned long long)dir_idx << 32) ^
                              (unsigned long long)iter_number;
    seed = (seed + 0x9E3779B97F4A7C15ULL) & 0xFFFFFFFFFFFFFFFFULL;
    seed = ((seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9ULL) & 0xFFFFFFFFFFFFFFFFULL;
    seed = ((seed ^ (seed >> 27)) * 0x94D049BB133111EBULL) & 0xFFFFFFFFFFFFFFFFULL;
    seed = seed ^ (seed >> 31);

    const float jitter = ((float)(seed & 0xFFFFFF) / (float)0xFFFFFF) * step_size;

    // -------------------------------------------------------------------
    // Initialize ray state
    // -------------------------------------------------------------------
    float current_d = start_d + jitter * dir_d;
    float current_h = start_h + jitter * dir_h;
    float current_w = start_w + jitter * dir_w;

    // Sample both textures at starting position
    float ct_origin = tex3D<float>(ct_tex,
                                    current_w + 0.5f,
                                    current_h + 0.5f,
                                    current_d + 0.5f);

    float prev_diff_value = tex3D<float>(diff_tex,
                                          current_w + 0.5f,
                                          current_h + 0.5f,
                                          current_d + 0.5f);

    // Last valid position: initialized to starting point
    float last_valid_d = start_d;
    float last_valid_h = start_h;
    float last_valid_w = start_w;
    float last_valid_dist = 0.0f;
    float last_valid_ct = ct_origin;

    // Trapezoidal integration state
    float accumulated_intensity = 0.0f;
    float accumulated_distance = 0.0f;
    float prev_weighted = 0.0f;

    // Exit latch state
    float ever_inside = 0.0f;
    float ever_exited_after_inside = 0.0f;

    // Inflection detection state
    float max_abs_gradient = 0.0f;  // running max of |dD/dr| while inside

    // -------------------------------------------------------------------
    // Uniform march
    // -------------------------------------------------------------------
    for (int s = 0; s < num_steps; s++) {
        current_d += step_size * dir_d;
        current_h += step_size * dir_h;
        current_w += step_size * dir_w;
        accumulated_distance += step_size;

        // Sample both textures
        float ct_value = tex3D<float>(ct_tex,
                                       current_w + 0.5f,
                                       current_h + 0.5f,
                                       current_d + 0.5f);

        float diff_value = tex3D<float>(diff_tex,
                                         current_w + 0.5f,
                                         current_h + 0.5f,
                                         current_d + 0.5f);

        // ---------------------------------------------------------------
        // Entry detection (same as zero-crossing variant)
        // ---------------------------------------------------------------
        float raw_inside = (diff_value > entry_threshold) ? 1.0f : 0.0f;

        // Latch: have we ever been inside?
        ever_inside = fmaxf(ever_inside, raw_inside);

        // ---------------------------------------------------------------
        // Inflection point detection (replaces zero-crossing exit)
        //
        // Compute |dD/dr| via first difference. We use the negative
        // gradient (descent from vessel to parenchyma) because that's
        // the transition direction we care about. The inflection point
        // is where this descent rate peaks.
        //
        // We track max |dD/dr| only while inside (ever_inside=1,
        // ever_exited=0), and trigger the exit latch when the current
        // |dD/dr| drops below gradient_decay_fraction * max_abs_gradient.
        // ---------------------------------------------------------------
        float gradient = (diff_value - prev_diff_value) * inv_step_size;
        float abs_gradient = fabsf(gradient);

        // Only update the running max while inside and not yet exited
        float inside_and_not_exited = ever_inside * (1.0f - ever_exited_after_inside);
        max_abs_gradient = fmaxf(max_abs_gradient,
                                  inside_and_not_exited * abs_gradient);

        // Inflection exit condition: gradient has decayed below the
        // fraction of the peak. Only trigger when:
        //   1. We've been inside (ever_inside = 1)
        //   2. We've seen a meaningful gradient (max_abs_gradient > 0)
        //   3. Current |dD/dr| has dropped below the decay fraction
        //
        // The (max_abs_gradient > 0) check prevents triggering on the
        // first step inside before any gradient has been observed.
        float gradient_decayed = (abs_gradient < gradient_decay_fraction * max_abs_gradient
                                  && max_abs_gradient > 0.0f) ? 1.0f : 0.0f;

        float inflection_exit = ever_inside * gradient_decayed;

        // Combine: exit on inflection OR on diff crossing below threshold
        // (the threshold exit serves as a safety net for cases where the
        // gradient never clearly peaks, e.g., very gradual transitions)
        float threshold_exit = ever_inside * (1.0f - raw_inside);
        float exited_now = fmaxf(inflection_exit, threshold_exit);

        // Permanent exit latch
        ever_exited_after_inside = fmaxf(ever_exited_after_inside, exited_now);

        // Final inside flag
        float inside = raw_inside * (1.0f - ever_exited_after_inside);

        // Save for next step's gradient computation
        prev_diff_value = diff_value;

        // ---------------------------------------------------------------
        // Last valid position tracking (branchless conditional update)
        // ---------------------------------------------------------------
        float outside = 1.0f - inside;
        last_valid_d    = inside * current_d            + outside * last_valid_d;
        last_valid_h    = inside * current_h            + outside * last_valid_h;
        last_valid_w    = inside * current_w            + outside * last_valid_w;
        last_valid_dist = inside * accumulated_distance + outside * last_valid_dist;
        last_valid_ct   = inside * ct_value             + outside * last_valid_ct;

        // ---------------------------------------------------------------
        // Intensity accumulation (trapezoidal rule, gated by inside)
        // ---------------------------------------------------------------
        float next_weighted = ct_value * accumulated_distance * accumulated_distance;
        float step_contribution = 0.5f * step_size * (prev_weighted + next_weighted);

        accumulated_intensity += inside * step_contribution;
        prev_weighted = inside * next_weighted;
    }

    // -------------------------------------------------------------------
    // Write outputs
    // -------------------------------------------------------------------
    const int out_idx_2d = point_idx * num_dirs + dir_idx;
    const int out_idx_3d = out_idx_2d * 3;

    intensity_weights[out_idx_2d] = accumulated_intensity;
    intersection_distances[out_idx_2d] = last_valid_dist;
    intersection_values[out_idx_2d] = last_valid_ct;
    intersection_points[out_idx_3d + 0] = last_valid_d;
    intersection_points[out_idx_3d + 1] = last_valid_h;
    intersection_points[out_idx_3d + 2] = last_valid_w;
}
'''


meanshift_grace_kernel_code = r'''
// ===========================================================================
// Dual-texture branchless mean-shift kernel - grace distance variant
//
// Four-state exit latch with grace period for bridging grid aliasing gaps:
//
//
// All state encoded in float registers, all transitions via fmaxf/fminf
// and conditional multiplies. No control flow branches.
//
// Thread mapping: Convention A (x = points, y = directions)
// ===========================================================================
extern "C" __global__ void ray_cast_meanshift_grace_texture(
    cudaTextureObject_t ct_tex,
    cudaTextureObject_t diff_tex,
    const float* __restrict__ point_coords,
    const int*   __restrict__ point_mask,
    const float* __restrict__ directions,
    const float* __restrict__ max_distances,
    float* __restrict__ intensity_weights,
    float* __restrict__ intersection_points,
    float* __restrict__ intersection_distances,
    float* __restrict__ intersection_values,
    const float boundary_threshold,
    const float grace_distance,
    const int   num_steps,
    const long long iter_number,
    const int   num_points,
    const int   num_dirs
) {
    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dir_idx   = blockIdx.y * blockDim.y + threadIdx.y;

    if (point_idx >= num_points || dir_idx >= num_dirs) return;
    if (point_mask[point_idx] == 0) return;

    // -------------------------------------------------------------------
    // Load direction vector
    // -------------------------------------------------------------------
    const int dir_offset = dir_idx * 3;
    const float dir_d = directions[dir_offset + 0];
    const float dir_h = directions[dir_offset + 1];
    const float dir_w = directions[dir_offset + 2];

    // -------------------------------------------------------------------
    // Load starting point
    // -------------------------------------------------------------------
    const int pt_offset = point_idx * 3;
    const float start_d = point_coords[pt_offset + 0];
    const float start_h = point_coords[pt_offset + 1];
    const float start_w = point_coords[pt_offset + 2];

    // -------------------------------------------------------------------
    // Compute step size
    // -------------------------------------------------------------------
    const float max_dist = max_distances[point_idx];
    const float step_size = max_dist / (float)num_steps;

    // -------------------------------------------------------------------
    // Jitter
    // -------------------------------------------------------------------
    unsigned long long seed = (unsigned long long)point_idx ^
                              ((unsigned long long)dir_idx << 32) ^
                              (unsigned long long)iter_number;
    seed = (seed + 0x9E3779B97F4A7C15ULL) & 0xFFFFFFFFFFFFFFFFULL;
    seed = ((seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9ULL) & 0xFFFFFFFFFFFFFFFFULL;
    seed = ((seed ^ (seed >> 27)) * 0x94D049BB133111EBULL) & 0xFFFFFFFFFFFFFFFFULL;
    seed = seed ^ (seed >> 31);
    const float jitter = ((float)(seed & 0xFFFFFF) / (float)0xFFFFFF) * step_size;

    // -------------------------------------------------------------------
    // Initialize ray state
    // -------------------------------------------------------------------
    float current_d = start_d + jitter * dir_d;
    float current_h = start_h + jitter * dir_h;
    float current_w = start_w + jitter * dir_w;

    float ct_origin = tex3D<float>(ct_tex,
                                    current_w + 0.5f,
                                    current_h + 0.5f,
                                    current_d + 0.5f);

    // Last valid position (updated while INSIDE, frozen during GRACE/EXITED)
    float last_valid_d = start_d;
    float last_valid_h = start_h;
    float last_valid_w = start_w;
    float last_valid_dist = 0.0f;
    float last_valid_ct = ct_origin;

    // Exit position: captured at the moment of first exit from INSIDE.
    // If grace period expires, this becomes the final intersection point.
    float exit_pos_d = start_d;
    float exit_pos_h = start_h;
    float exit_pos_w = start_w;
    float exit_pos_dist = 0.0f;
    float exit_pos_ct = ct_origin;

    // Trapezoidal integration state
    float accumulated_intensity = 0.0f;
    float accumulated_distance = 0.0f;
    float prev_weighted = 0.0f;

    // State machine registers
    float ever_inside = 0.0f;                // latches to 1.0 on first entry
    float permanently_exited = 0.0f;         // latches to 1.0 when grace expires
    float in_grace_period = 0.0f;            // 1.0 while in grace, 0.0 otherwise
    float grace_distance_traveled = 0.0f;    // distance since entering grace

    // -------------------------------------------------------------------
    // Uniform march
    // -------------------------------------------------------------------
    for (int s = 0; s < num_steps; s++) {
        current_d += step_size * dir_d;
        current_h += step_size * dir_h;
        current_w += step_size * dir_w;
        accumulated_distance += step_size;

        float ct_value = tex3D<float>(ct_tex,
                                       current_w + 0.5f,
                                       current_h + 0.5f,
                                       current_d + 0.5f);

        float diff_value = tex3D<float>(diff_tex,
                                         current_w + 0.5f,
                                         current_h + 0.5f,
                                         current_d + 0.5f);

        // ---------------------------------------------------------------
        // Classify current sample
        // ---------------------------------------------------------------
        float raw_inside = (diff_value > boundary_threshold) ? 1.0f : 0.0f;

        // Have we ever entered the vessel?
        ever_inside = fmaxf(ever_inside, raw_inside);

        // Are we currently outside after having been inside?
        float currently_outside = ever_inside * (1.0f - raw_inside);

        // ---------------------------------------------------------------
        // Grace period logic
        //
        // Entering grace: transition from inside to outside (and not
        // already permanently exited). Capture exit position.
        //
        // During grace: accumulate distance. If we re-enter (raw_inside=1),
        // reset grace state. If distance exceeds grace_distance, latch.
        // ---------------------------------------------------------------

        // Detect entering grace period (was inside, now outside, not yet
        // permanently exited, not already in grace)
        float entering_grace = currently_outside
                             * (1.0f - permanently_exited)
                             * (1.0f - in_grace_period);

        // Capture exit position at the moment we first leave.
        // Only overwrite if entering_grace is 1.0; otherwise keep previous.
        float keep_exit = 1.0f - entering_grace;
        exit_pos_d    = entering_grace * last_valid_d    + keep_exit * exit_pos_d;
        exit_pos_h    = entering_grace * last_valid_h    + keep_exit * exit_pos_h;
        exit_pos_w    = entering_grace * last_valid_w    + keep_exit * exit_pos_w;
        exit_pos_dist = entering_grace * last_valid_dist + keep_exit * exit_pos_dist;
        exit_pos_ct   = entering_grace * last_valid_ct   + keep_exit * exit_pos_ct;

        // Update grace state
        // Enter grace if just started, OR stay in grace if already there
        float was_in_grace = in_grace_period;
        in_grace_period = fmaxf(in_grace_period, entering_grace);

        // Accumulate grace distance while in grace and outside
        grace_distance_traveled += in_grace_period * currently_outside * step_size;

        // Re-entry: if we're in grace and raw_inside=1, reset grace
        float re_entered = in_grace_period * raw_inside * (1.0f - permanently_exited);
        // Reset grace state on re-entry
        in_grace_period    = in_grace_period * (1.0f - re_entered);
        grace_distance_traveled = grace_distance_traveled * (1.0f - re_entered);

        // Grace expired: grace distance exceeded threshold
        float grace_expired = (grace_distance_traveled > grace_distance) ? 1.0f : 0.0f;
        grace_expired = grace_expired * in_grace_period;

        // Permanent exit latch
        permanently_exited = fmaxf(permanently_exited, grace_expired);

        // ---------------------------------------------------------------
        // Compute effective inside flag
        // inside = 1 only if: currently positive AND ever been inside
        //          AND not permanently exited AND not in grace period
        // ---------------------------------------------------------------
        float inside = raw_inside
                     * ever_inside
                     * (1.0f - permanently_exited)
                     * (1.0f - in_grace_period);

        // ---------------------------------------------------------------
        // Last valid position tracking
        // ---------------------------------------------------------------
        float outside = 1.0f - inside;
        last_valid_d    = inside * current_d            + outside * last_valid_d;
        last_valid_h    = inside * current_h            + outside * last_valid_h;
        last_valid_w    = inside * current_w            + outside * last_valid_w;
        last_valid_dist = inside * accumulated_distance + outside * last_valid_dist;
        last_valid_ct   = inside * ct_value             + outside * last_valid_ct;

        // ---------------------------------------------------------------
        // Intensity accumulation (only while INSIDE, not during GRACE)
        // ---------------------------------------------------------------
        float next_weighted = ct_value * accumulated_distance * accumulated_distance;
        float step_contribution = 0.5f * step_size * (prev_weighted + next_weighted);

        accumulated_intensity += inside * step_contribution;
        prev_weighted = inside * next_weighted;
    }

    // -------------------------------------------------------------------
    // Resolve final position
    //
    // If permanently_exited: use exit_pos (where the ray first left)
    // If still inside or in grace at end of march: use last_valid
    // If never entered: use start position (last_valid default)
    // -------------------------------------------------------------------
    float use_exit = permanently_exited;
    float use_last = 1.0f - use_exit;

    float final_d    = use_exit * exit_pos_d    + use_last * last_valid_d;
    float final_h    = use_exit * exit_pos_h    + use_last * last_valid_h;
    float final_w    = use_exit * exit_pos_w    + use_last * last_valid_w;
    float final_dist = use_exit * exit_pos_dist + use_last * last_valid_dist;
    float final_ct   = use_exit * exit_pos_ct   + use_last * last_valid_ct;

    // -------------------------------------------------------------------
    // Write outputs
    // -------------------------------------------------------------------
    const int out_idx_2d = point_idx * num_dirs + dir_idx;
    const int out_idx_3d = out_idx_2d * 3;

    intensity_weights[out_idx_2d] = accumulated_intensity;
    intersection_distances[out_idx_2d] = final_dist;
    intersection_values[out_idx_2d] = final_ct;
    intersection_points[out_idx_3d + 0] = final_d;
    intersection_points[out_idx_3d + 1] = final_h;
    intersection_points[out_idx_3d + 2] = final_w;
}
'''


# Kernel cache
_ray_cast_kernel_cache = {}


# =============================================================================
# TEXTURE RAY CASTER CLASS
# =============================================================================


class TextureObject3D:
    """
    RAII wrapper for CUDA 3D texture objects.
    
    Manages the lifecycle of CUDA arrays and texture objects, ensuring proper
    cleanup when the object goes out of scope or is explicitly deleted.
    
    Usage:
        tex = TextureObject3D(volume_gpu)
        # Use tex.texture_object in kernel calls
        del tex  # Explicit cleanup, or let garbage collector handle it
    
    For batch processing, cache the TextureObject3D to avoid repeated allocation:
        tex = TextureObject3D(volume_gpu)
        for scale in scales:
            resample_with_cached_texture(tex, output, scale)
        del tex
    """
    
    def __init__(
        self, 
        volume_gpu: cp.ndarray,
        address_mode: str = 'clamp',
        filter_mode: str = 'linear',
        normalized_coords: bool = False
    ):
        """
        Create a CUDA texture object for hardware-accelerated 3D sampling.
        
        Parameters
        ----------
        volume_gpu : cp.ndarray
            3D input volume on GPU. Must be float32 and C-contiguous.
        address_mode : str
            Boundary handling: 'clamp', 'border' (zero), 'wrap', or 'mirror'
        filter_mode : str
            Interpolation: 'linear' (trilinear) or 'point' (nearest neighbor)
        normalized_coords : bool
            If True, coordinates are in [0, 1). If False, in [0, dim).
        """
        from cupy.cuda import texture, runtime
        
        # Validate input
        if volume_gpu.ndim != 3:
            raise ValueError(f"Expected 3D array, got {volume_gpu.ndim}D")
        if not volume_gpu.flags.c_contiguous:
            volume_gpu = cp.ascontiguousarray(volume_gpu)
        if volume_gpu.dtype != cp.float32:
            volume_gpu = volume_gpu.astype(cp.float32)
        
        self._texture_module = texture
        self.shape = volume_gpu.shape  # (D, H, W)
        self.device = volume_gpu.device.id
        
        # Store references to prevent garbage collection
        self._cuda_array = None
        self._texture_object = None
        
        # Address mode mapping
        address_mode_map = {
            'clamp': runtime.cudaAddressModeClamp,
            'border': runtime.cudaAddressModeBorder,
            'wrap': runtime.cudaAddressModeWrap,
            'mirror': runtime.cudaAddressModeMirror,
        }
        if address_mode not in address_mode_map:
            raise ValueError(f"Unknown address_mode: {address_mode}. "
                           f"Use one of {list(address_mode_map.keys())}")
        
        # Filter mode mapping
        filter_mode_map = {
            'linear': runtime.cudaFilterModeLinear,
            'point': runtime.cudaFilterModePoint,
        }
        if filter_mode not in filter_mode_map:
            raise ValueError(f"Unknown filter_mode: {filter_mode}. "
                           f"Use one of {list(filter_mode_map.keys())}")
        
        with cp.cuda.Device(self.device):
            # Channel format: single-channel float32
            channel_desc = texture.ChannelFormatDescriptor(
                32, 0, 0, 0,  # 32 bits for X channel, 0 for Y/Z/W
                runtime.cudaChannelFormatKindFloat
            )
            
            # Allocate CUDA array with texture-optimized memory layout
            # Note: CUDAarray constructor takes (W, H, D) order, not (D, H, W)
            D, H, W = volume_gpu.shape
            self._cuda_array = texture.CUDAarray(channel_desc, W, H, D)
            
            # Copy data from linear GPU memory to CUDA array
            # This reformats the data into the texture-optimized layout
            self._cuda_array.copy_from(volume_gpu)
            
            # Resource descriptor: tells texture where data lives
            res_desc = texture.ResourceDescriptor(
                runtime.cudaResourceTypeArray,
                cuArr=self._cuda_array
            )
            
            # Texture descriptor: tells texture how to sample
            tex_desc = texture.TextureDescriptor(
                (address_mode_map[address_mode],) * 3,  # Same mode for all axes
                filter_mode_map[filter_mode],
                runtime.cudaReadModeElementType,  # Return raw float, not normalized
                normalizedCoords=int(normalized_coords)
            )
            
            # Create the texture object
            self._texture_object = texture.TextureObject(res_desc, tex_desc)
    
    @property
    def ptr(self) -> int:
        """Raw pointer (cudaTextureObject_t) for passing to kernels."""
        if self._texture_object is None:
            raise RuntimeError("Texture object has been destroyed")
        return self._texture_object.ptr
    
    @property
    def texture_object(self):
        """The underlying CuPy TextureObject."""
        return self._texture_object
    
    def __del__(self):
        """Clean up CUDA resources."""
        self.destroy()
    
    def destroy(self):
        """Explicitly release CUDA resources."""
        self._texture_object = None
        self._cuda_array = None

class TextureRayCaster:
    """
    High-performance ray caster using CUDA texture memory.
    
    This class wraps a 3D volume in a texture object and provides methods
    for ray casting with hardware-accelerated trilinear interpolation.
    
    The texture object is created once and reused for multiple ray casting
    calls, amortizing the texture creation overhead.
    
    Parameters
    ----------
    volume_gpu : cp.ndarray
        3D volume on GPU (D, H, W), will be converted to float32 if needed
    address_mode : str
        Boundary handling: 'clamp', 'border', 'wrap', or 'mirror'
    device : int, optional
        GPU device ID. If None, uses the volume's device.
        
    Examples
    --------
    >>> volume = cp.array(my_volume, dtype=cp.float32)
    >>> caster = TextureRayCaster(volume)
    >>> 
    >>> # Prepare outputs
    >>> intersection_points = cp.zeros((N, M, 3), dtype=cp.float32)
    >>> intersection_distances = cp.zeros((N, M), dtype=cp.float32)
    >>> intersection_values = cp.zeros((N, M), dtype=cp.float32)
    >>> 
    >>> # Cast rays
    >>> caster.cast_rays(
    ...     point_coords, directions,
    ...     intersection_points, intersection_distances, intersection_values,
    ...     threshold=600.0, max_steps=150,
    ...     step_sizes=[1.0, 0.1, 0.01],
    ...     max_distances=max_dist_array
    ... )
    >>> 
    >>> caster.destroy()
    """
    
    def __init__(
        self,
        volume_gpu: cp.ndarray,
        address_mode: str = 'clamp',
        device: Optional[int] = None,
        boundary_volume_gpu=None,
        boundary_address_mode: str = 'border'
    ):
        if device is None:
            device = volume_gpu.device.id
        
        self.device = device
        self.volume_shape = volume_gpu.shape
        
        with cp.cuda.Device(device):
            # Create texture object
            self._texture = TextureObject3D(
                volume_gpu,
                address_mode=address_mode,
                filter_mode='linear',  # Enable hardware trilinear interpolation
                normalized_coords=False
            )
            self._boundary_texture = None

            if boundary_volume_gpu is not None:
                self._boundary_texture = TextureObject3D(
                    boundary_volume_gpu,
                    address_mode=boundary_address_mode,
                    filter_mode='linear',
                    normalized_coords=False
                )
            # Compile kernels
            self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile and cache the ray casting kernels."""
        global _ray_cast_kernel_cache
        
        if 'ray_cast_texture_kernel' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_texture_kernel'] = cp.RawKernel(
                ray_cast_texture_kernel_code,
                'ray_cast_texture_kernel'
            )
        
        if 'ray_cast_texture_kernel_shared' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_texture_kernel_shared'] = cp.RawKernel(
                ray_cast_texture_kernel_shared_code,
                'ray_cast_texture_kernel_shared'
            )
        
        if 'ray_cast_difference_branchless_texture' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_difference_branchless_texture'] = cp.RawKernel(
                difference_branchless_kernel_code,
                'ray_cast_difference_branchless_texture'
            )
        
        if 'ray_cast_gradient_branchless_texture' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_gradient_branchless_texture'] = cp.RawKernel(
                gradient_branchless_kernel_code,
                'ray_cast_gradient_branchless_texture'
            )

        if 'ray_cast_intensity_branchless_texture' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_intensity_branchless_texture'] = cp.RawKernel(
                intensity_branchless_kernel_code,
                'ray_cast_intensity_branchless_texture'
            )

        if 'ray_cast_meanshift_branchless_texture' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_meanshift_branchless_texture'] = cp.RawKernel(
                meanshift_branchless_kernel_code,
                'ray_cast_meanshift_branchless_texture'
            )

        if 'ray_cast_meanshift_inflection_texture' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_meanshift_inflection_texture'] = cp.RawKernel(
                meanshift_inflection_kernel_code,
                'ray_cast_meanshift_inflection_texture'
            )

        if 'ray_cast_meanshift_grace_texture' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_meanshift_grace_texture'] = cp.RawKernel(
                meanshift_grace_kernel_code,
                'ray_cast_meanshift_grace_texture'
            )


        self._kernel = _ray_cast_kernel_cache['ray_cast_texture_kernel']
        self._kernel_shared = _ray_cast_kernel_cache['ray_cast_texture_kernel_shared']
        self._kernel_branchless_difference = _ray_cast_kernel_cache['ray_cast_difference_branchless_texture']
        self._kernel_branchless_gradient = _ray_cast_kernel_cache['ray_cast_gradient_branchless_texture']
        self._kernel_branchless_intensity = _ray_cast_kernel_cache['ray_cast_intensity_branchless_texture']
        self._kernel_meanshift_branchless = _ray_cast_kernel_cache['ray_cast_meanshift_branchless_texture']
        self._kernel_meanshift_inflection = _ray_cast_kernel_cache['ray_cast_meanshift_inflection_texture']
        self._kernel_meanshift_grace = _ray_cast_kernel_cache['ray_cast_meanshift_grace_texture']

    @property
    def texture(self) -> TextureObject3D:
        """The underlying texture object."""
        return self._texture
    

    def cast_rays(
        self,
        point_coords: cp.ndarray,
        directions: cp.ndarray,
        intersection_points: cp.ndarray,
        intersection_distances: cp.ndarray,
        intersection_values: cp.ndarray,
        threshold: float,
        max_steps: int,
        step_sizes: Union[List[float], cp.ndarray],
        max_distances: cp.ndarray,
        point_mask: Optional[cp.ndarray] = None,
        use_shared_memory: bool = True,
        stream: Optional[cp.cuda.Stream] = None
    ) -> None:
        """
        Cast rays from points in specified directions.
        
        Parameters
        ----------
        point_coords : cp.ndarray
            Starting points, shape (N, 3) with coordinates (d, h, w)
        directions : cp.ndarray
            Unit direction vectors, shape (M, 3)
        intersection_points : cp.ndarray
            Output array for intersection coordinates, shape (N, M, 3)
        intersection_distances : cp.ndarray
            Output array for distances to intersections, shape (N, M)
        intersection_values : cp.ndarray
            Output array for interpolated values at intersections, shape (N, M)
        threshold : float
            Value threshold for surface detection
        max_steps : int
            Maximum ray marching iterations
        step_sizes : list or cp.ndarray
            Adaptive step sizes (e.g., [1.0, 0.1, 0.01])
        max_distances : cp.ndarray
            Maximum ray distance per point, shape (N,)
        point_mask : cp.ndarray, optional
            Boolean mask for active points, shape (N,). If None, all points are active.
        use_shared_memory : bool
            Use shared memory kernel for directions (default True)
        stream : cp.cuda.Stream, optional
            CUDA stream for async execution
        """
        num_points = point_coords.shape[0]
        num_dirs = directions.shape[0]
        
        with cp.cuda.Device(self.device):
            if not point_coords.flags.c_contiguous:
                point_coords = cp.ascontiguousarray(point_coords)
            if not directions.flags.c_contiguous:
                directions = cp.ascontiguousarray(directions)
            
            if point_mask is None:
                point_mask = cp.ones(num_points, dtype=cp.int32)
            elif point_mask.dtype != cp.int32:
                point_mask = point_mask.astype(cp.int32)
            
            if isinstance(step_sizes, list):
                step_sizes_gpu = cp.array(step_sizes, dtype=cp.float32)
            else:
                step_sizes_gpu = step_sizes.astype(cp.float32) if step_sizes.dtype != cp.float32 else step_sizes
            
            num_step_sizes = len(step_sizes_gpu)
            
            if max_distances.dtype != cp.float32:
                max_distances = max_distances.astype(cp.float32)
            
            kernel = self._kernel_shared if use_shared_memory else self._kernel
            threads_per_block = (32, 16)
            blocks_per_grid_x = (num_points + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (num_dirs + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            args = (
                self._texture.texture_object,    # cudaTextureObject_t
                point_coords,                    # float* point_coords
                point_mask,                      # int* point_mask
                directions,                      # float* directions
                intersection_points,             # float* intersection_points
                intersection_distances,          # float* intersection_distances
                intersection_values,             # float* intersection_values
                float32(threshold),              # float threshold
                int32(max_steps),                # int max_steps
                step_sizes_gpu,                  # float* step_sizes
                int32(num_step_sizes),           # int num_step_sizes
                max_distances,                   # float* max_distances
                int32(num_points),               # int num_points
                int32(num_dirs),                 # int num_dirs
            )
            
            if stream is not None:
                kernel(blocks_per_grid, threads_per_block, args, stream=stream)
            else:
                kernel(blocks_per_grid, threads_per_block, args)
                cp.cuda.runtime.deviceSynchronize()
    

    def cast_difference_branchless(
        self,  # TextureRayCaster instance
        point_coords: cp.ndarray,
        directions: cp.ndarray,
        point_mask: cp.ndarray,
        checkpoint_distances: cp.ndarray,
        difference_neg_output: cp.ndarray,
        difference_pos_output: cp.ndarray,
        checkpoint_reset: bool = True,
        num_steps_per_interval: int = 4,
        iter_number: int = 0,
        threads_x: int = 32,
        threads_y: int = 4,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> None:
        """
        Branchless difference profiling using texture memory and warp reduction.

        Parameters
        ----------
        point_coords : cp.ndarray, shape (N, 3)
            Point coordinates (d, h, w), float32, C-contiguous.
        directions : cp.ndarray, shape (M, 3)
            Unit direction vectors, float32, C-contiguous.
        point_mask : cp.ndarray, shape (N,)
            Boolean mask as int32 (1 = active, 0 = skip).
        checkpoint_distances : cp.ndarray, shape (K,)
            Sorted distance checkpoints, float32.
        difference_neg_output : cp.ndarray, shape (N, K)
            Output: most negative difference at each checkpoint. Must be
            zero-initialized before launch.
        difference_pos_output : cp.ndarray, shape (N, K)
            Output: most positive difference at each checkpoint. Must be
            zero-initialized before launch.
        checkpoint_reset : bool
            If True, reset reference value at each checkpoint boundary.
        num_steps_per_interval : int
            Uniform steps per checkpoint interval.
        iter_number : int
            Iteration seed for deterministic jitter.
        threads_x : int
            Threads in x-dimension (directions). Should be 32 for warp reduction.
            Can be tuned, but must be a multiple of 32 (warp size).
        threads_y : int
            Threads in y-dimension (points). Tunable per GPU.
        stream : cp.cuda.Stream, optional
            CUDA stream for async execution.
        """
        num_points = point_coords.shape[0]
        num_dirs = directions.shape[0]
        num_checkpoints = checkpoint_distances.shape[0]

        with cp.cuda.Device(self.device):
            # Ensure contiguous float32 / int32
            if not point_coords.flags.c_contiguous:
                point_coords = cp.ascontiguousarray(point_coords)
            if not directions.flags.c_contiguous:
                directions = cp.ascontiguousarray(directions)
            if point_mask.dtype != cp.int32:
                point_mask = point_mask.astype(cp.int32)

            kernel = self._kernel_branchless_difference

            checkpoint_reset_f = np.float32(1.0 if checkpoint_reset else 0.0)

            threads_per_block = (threads_x, threads_y)
            blocks_x = (num_dirs + threads_x - 1) // threads_x      # directions
            blocks_y = (num_points + threads_y - 1) // threads_y     # points
            blocks_per_grid = (blocks_x, blocks_y)

            args = (
                self._texture.texture_object,   # cudaTextureObject_t
                point_coords,                   # float*
                point_mask,                     # int*
                directions,                     # float*
                checkpoint_distances,           # float*
                difference_neg_output,          # float*
                difference_pos_output,          # float*
                checkpoint_reset_f,             # float
                int32(num_steps_per_interval),  # int
                np.int64(iter_number),          # long long
                int32(num_points),              # int
                int32(num_dirs),                # int
                int32(num_checkpoints),         # int
            )

            if stream is not None:
                kernel(blocks_per_grid, threads_per_block, args, stream=stream)
            else:
                kernel(blocks_per_grid, threads_per_block, args)
                cp.cuda.runtime.deviceSynchronize()


    def cast_gradient_branchless(
        self,  # TextureRayCaster instance
        point_coords: cp.ndarray,
        directions: cp.ndarray,
        point_mask: cp.ndarray,
        checkpoint_distances: cp.ndarray,
        weight_table: cp.ndarray,
        gradient_neg_output: cp.ndarray,
        gradient_pos_output: cp.ndarray,
        checkpoint_reset: bool = True,
        num_steps_per_interval: int = 4,
        iter_number: int = 0,
        threads_x: int = 32,
        threads_y: int = 4,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> None:
        """
        Branchless Gaussian-weighted gradient profiling using texture memory
        and warp reduction.

        Parameters
        ----------
        point_coords : cp.ndarray, shape (N, 3)
            Point coordinates (d, h, w), float32, C-contiguous.
        directions : cp.ndarray, shape (M, 3)
            Unit direction vectors, float32, C-contiguous.
        point_mask : cp.ndarray, shape (N,)
            Boolean mask as int32 (1 = active, 0 = skip).
        checkpoint_distances : cp.ndarray, shape (K,)
            Sorted distance checkpoints, float32.
        weight_table : cp.ndarray, shape (K * num_steps_per_interval,)
            Precomputed Gaussian weights, float32. Layout:
            [ck0_step0, ck0_step1, ..., ck1_step0, ck1_step1, ...]
        gradient_neg_output : cp.ndarray, shape (N, K)
            Output: most negative gradient at each checkpoint. Must be
            zero-initialized before launch.
        gradient_pos_output : cp.ndarray, shape (N, K)
            Output: most positive gradient at each checkpoint. Must be
            zero-initialized before launch.
        checkpoint_reset : bool
            If True, reset reference value and accumulators at each checkpoint.
        num_steps_per_interval : int
            Uniform steps per checkpoint interval.
        iter_number : int
            Iteration seed for deterministic jitter.
        threads_x : int
            Threads in x-dimension (directions). Should be 32 for warp reduction.
        threads_y : int
            Threads in y-dimension (points). Tunable per GPU.
        stream : cp.cuda.Stream, optional
            CUDA stream for async execution.
        """
        num_points = point_coords.shape[0]
        num_dirs = directions.shape[0]
        num_checkpoints = checkpoint_distances.shape[0]

        with cp.cuda.Device(self.device):
            if not point_coords.flags.c_contiguous:
                point_coords = cp.ascontiguousarray(point_coords)
            if not directions.flags.c_contiguous:
                directions = cp.ascontiguousarray(directions)
            if point_mask.dtype != cp.int32:
                point_mask = point_mask.astype(cp.int32)
            if not weight_table.flags.c_contiguous:
                weight_table = cp.ascontiguousarray(weight_table)

            kernel = self._kernel_branchless_gradient

            checkpoint_reset_f = np.float32(1.0 if checkpoint_reset else 0.0)

            threads_per_block = (threads_x, threads_y)
            blocks_x = (num_dirs + threads_x - 1) // threads_x
            blocks_y = (num_points + threads_y - 1) // threads_y
            blocks_per_grid = (blocks_x, blocks_y)

            # Dynamic shared memory for weight table
            shared_mem_bytes = num_checkpoints * num_steps_per_interval * 4  # float32

            args = (
                self._texture.texture_object,
                point_coords,
                point_mask,
                directions,
                checkpoint_distances,
                weight_table,
                gradient_neg_output,
                gradient_pos_output,
                checkpoint_reset_f,
                int32(num_steps_per_interval),
                np.int64(iter_number),
                int32(num_points),
                int32(num_dirs),
                int32(num_checkpoints),
            )

            if stream is not None:
                kernel(blocks_per_grid, threads_per_block, args,
                    shared_mem=shared_mem_bytes, stream=stream)
            else:
                kernel(blocks_per_grid, threads_per_block, args,
                    shared_mem=shared_mem_bytes)
                cp.cuda.runtime.deviceSynchronize()


    def cast_intensity_branchless(
        self,  # TextureRayCaster instance
        point_coords: cp.ndarray,
        directions: cp.ndarray,
        point_mask: cp.ndarray,
        ray_omega: cp.ndarray,
        checkpoint_distances: cp.ndarray,
        intensity_output: cp.ndarray,
        num_steps_per_interval: int = 4,
        iter_number: int = 0,
        threads_x: int = 32,
        threads_y: int = 4,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> None:
        """
        Branchless per-interval intensity integration using texture memory
        and warp-level sum reduction.

        Parameters
        ----------
        point_coords : cp.ndarray, shape (N, 3)
            Point coordinates (d, h, w), float32, C-contiguous.
        directions : cp.ndarray, shape (M, 3)
            Unit direction vectors, float32, C-contiguous.
        point_mask : cp.ndarray, shape (N,)
            Boolean mask as int32 (1 = active, 0 = skip).
        ray_omega : cp.ndarray, shape (M,)
            Solid angle per direction, float32.
        checkpoint_distances : cp.ndarray, shape (K,)
            Sorted distance checkpoints, float32.
        intensity_output : cp.ndarray, shape (N, K)
            Output: omega-weighted per-interval integrated intensity at each
            checkpoint. Must be zero-initialized before launch. Each element
            receives the sum across all directions of:
                omega_m * ∫_{r_{k-1}}^{r_k} I(r)*r² dr
        num_steps_per_interval : int
            Uniform steps per checkpoint interval.
        iter_number : int
            Iteration seed for deterministic jitter.
        threads_x : int
            Threads in x-dimension (directions). Should be 32 for warp reduction.
        threads_y : int
            Threads in y-dimension (points). Tunable per GPU.
        stream : cp.cuda.Stream, optional
            CUDA stream for async execution.
        """
        num_points = point_coords.shape[0]
        num_dirs = directions.shape[0]
        num_checkpoints = checkpoint_distances.shape[0]

        with cp.cuda.Device(self.device):
            # Ensure contiguous float32 / int32
            if not point_coords.flags.c_contiguous:
                point_coords = cp.ascontiguousarray(point_coords)
            if not directions.flags.c_contiguous:
                directions = cp.ascontiguousarray(directions)
            if point_mask.dtype != cp.int32:
                point_mask = point_mask.astype(cp.int32)
            if not ray_omega.flags.c_contiguous:
                ray_omega = cp.ascontiguousarray(ray_omega)

            kernel = self._kernel_branchless_intensity

            threads_per_block = (threads_x, threads_y)
            blocks_x = (num_dirs + threads_x - 1) // threads_x      # directions
            blocks_y = (num_points + threads_y - 1) // threads_y     # points
            blocks_per_grid = (blocks_x, blocks_y)

            args = (
                self._texture.texture_object,   # cudaTextureObject_t
                point_coords,                   # float*
                point_mask,                     # int*
                directions,                     # float*
                ray_omega,                      # float*
                checkpoint_distances,           # float*
                intensity_output,               # float*
                int32(num_steps_per_interval),  # int
                np.int64(iter_number),          # long long
                int32(num_points),              # int
                int32(num_dirs),                # int
                int32(num_checkpoints),         # int
            )

            if stream is not None:
                kernel(blocks_per_grid, threads_per_block, args, stream=stream)
            else:
                kernel(blocks_per_grid, threads_per_block, args)
                cp.cuda.runtime.deviceSynchronize()


    def cast_meanshift_branchless(
        self,  # TextureRayCaster instance
        point_coords: cp.ndarray,
        directions: cp.ndarray,
        point_mask: cp.ndarray,
        max_distances: cp.ndarray,
        intensity_weights: cp.ndarray,
        intersection_points: cp.ndarray,
        intersection_distances: cp.ndarray,
        intersection_values: cp.ndarray,
        boundary_threshold: float = 0.0,
        num_steps: int = 48,
        iter_number: int = 0,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> None:
        """
        Dual-texture branchless mean-shift ray casting with exit latch.

        Marches each ray uniformly to max_distances[point], sampling the CT
        texture for intensity integration and the boundary texture for
        inside/outside determination. The exit latch ensures each ray only
        accumulates within its first contiguous vessel region.

        Parameters
        ----------
        point_coords : cp.ndarray, shape (N, 3)
            Current point positions (d, h, w), float32, C-contiguous.
        directions : cp.ndarray, shape (M, 3)
            Unit direction vectors, float32, C-contiguous.
        point_mask : cp.ndarray, shape (N,)
            Boolean mask as int32 (1 = active, 0 = skip).
        max_distances : cp.ndarray, shape (N,)
            Per-point maximum ray distance, float32.
        intensity_weights : cp.ndarray, shape (N, M)
            OUTPUT: per-ray accumulated I(r)·r² inside boundary.
            Must be pre-initialized (kernel overwrites for active points).
        intersection_points : cp.ndarray, shape (N, M, 3)
            OUTPUT: last valid position on each ray.
        intersection_distances : cp.ndarray, shape (N, M)
            OUTPUT: distance to last valid position.
        intersection_values : cp.ndarray, shape (N, M)
            OUTPUT: CT value at last valid position.
        boundary_threshold : float
            Threshold on the difference texture. diff_value > threshold
            means "inside vessel." Default 0.0 (sign change boundary).
        num_steps : int
            Number of uniform steps per ray. Step size =
            max_distances[point] / num_steps. Default 48.
        iter_number : int
            Iteration seed for deterministic jitter.
        stream : cp.cuda.Stream, optional
            CUDA stream for async execution.

        Notes
        -----
        Requires self._boundary_texture to be set (a TextureObject3D wrapping
        the std-normalized difference volume). If not set, raises RuntimeError.
        """
        if self._boundary_texture is None:
            raise RuntimeError(
                "cast_meanshift_branchless requires a boundary texture. "
                "Create TextureRayCaster with boundary_volume_gpu parameter."
            )

        num_points = point_coords.shape[0]
        num_dirs = directions.shape[0]

        with cp.cuda.Device(self.device):
            # Ensure contiguity and correct dtypes
            if not point_coords.flags.c_contiguous:
                point_coords = cp.ascontiguousarray(point_coords)
            if not directions.flags.c_contiguous:
                directions = cp.ascontiguousarray(directions)
            if point_mask.dtype != cp.int32:
                point_mask = point_mask.astype(cp.int32)
            if max_distances.dtype != cp.float32:
                max_distances = max_distances.astype(cp.float32)

            kernel = self._kernel_meanshift_branchless

            # Convention A: x = points, y = directions
            threads_per_block = (32, 16)
            blocks_x = (num_points + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_y = (num_dirs + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_x, blocks_y)

            args = (
                self._texture.texture_object,            # cudaTextureObject_t ct_tex
                self._boundary_texture.texture_object,   # cudaTextureObject_t diff_tex
                point_coords,                            # float*
                point_mask,                              # int*
                directions,                              # float*
                max_distances,                           # float*
                intensity_weights,                       # float* OUTPUT
                intersection_points,                     # float* OUTPUT
                intersection_distances,                  # float* OUTPUT
                intersection_values,                     # float* OUTPUT
                np.float32(boundary_threshold),          # float
                int32(num_steps),                        # int
                np.int64(iter_number),                   # long long
                int32(num_points),                       # int
                int32(num_dirs),                         # int
            )

            if stream is not None:
                kernel(blocks_per_grid, threads_per_block, args, stream=stream)
            else:
                kernel(blocks_per_grid, threads_per_block, args)
                cp.cuda.runtime.deviceSynchronize()

    
    def cast_meanshift_inflection(
        self,  # TextureRayCaster instance
        point_coords: cp.ndarray,
        directions: cp.ndarray,
        point_mask: cp.ndarray,
        max_distances: cp.ndarray,
        intensity_weights: cp.ndarray,
        intersection_points: cp.ndarray,
        intersection_distances: cp.ndarray,
        intersection_values: cp.ndarray,
        entry_threshold: float = 0.0,
        gradient_decay_fraction: float = 0.5,
        num_steps: int = 48,
        iter_number: int = 0,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> None:
        """
        Dual-texture branchless mean-shift with inflection point boundary.

        Same interface as cast_meanshift_branchless, but the exit latch triggers
        at the inflection point of the difference profile D(r) along the ray —
        where |dD/dr| peaks and starts declining — rather than at the
        zero-crossing of D(r).

        Parameters
        ----------
        point_coords : cp.ndarray, shape (N, 3)
            Current point positions (d, h, w), float32, C-contiguous.
        directions : cp.ndarray, shape (M, 3)
            Unit direction vectors, float32, C-contiguous.
        point_mask : cp.ndarray, shape (N,)
            Boolean mask as int32 (1 = active, 0 = skip).
        max_distances : cp.ndarray, shape (N,)
            Per-point maximum ray distance, float32.
        intensity_weights : cp.ndarray, shape (N, M)
            OUTPUT: per-ray accumulated I(r)·r² inside boundary.
        intersection_points : cp.ndarray, shape (N, M, 3)
            OUTPUT: last valid position on each ray.
        intersection_distances : cp.ndarray, shape (N, M)
            OUTPUT: distance to last valid position.
        intersection_values : cp.ndarray, shape (N, M)
            OUTPUT: CT value at last valid position.
        entry_threshold : float
            Threshold on the difference texture for detecting vessel entry.
            diff_value > entry_threshold means "inside vessel."
            Default 0.0 (positive difference = inside).
        gradient_decay_fraction : float
            Fraction of peak |dD/dr| below which the exit latch triggers.
            Controls boundary placement:
                0.5:  latch when gradient drops to half its peak (conservative,
                    boundary at the half-width of the transition)
                0.8:  latch when gradient drops to 80% of peak (sensitive,
                    boundary closer to the peak gradient location)
                0.95: latch almost immediately after peak (aggressive, boundary
                    at the steepest point of the transition)
            Default 0.5 (half-maximum of the gradient, analogous to FWHM).
        num_steps : int
            Number of uniform steps per ray. Default 48.
        iter_number : int
            Iteration seed for deterministic jitter.
        stream : cp.cuda.Stream, optional
            CUDA stream for async execution.

        Notes
        -----
        The inflection point boundary is placed ~0.5–1.0 voxels inside the
        zero-crossing boundary for typical lung CT PSF widths. This corresponds
        to the true geometric edge of the vessel (center of the PSF blur).

        A safety-net threshold exit is also active: if the difference value
        drops below entry_threshold before the gradient decay triggers, the
        ray exits via the threshold path. This handles cases where the
        gradient profile is noisy or doesn't have a clear peak.
        """
        if self._boundary_texture is None:
            raise RuntimeError(
                "cast_meanshift_inflection requires a boundary texture. "
                "Create TextureRayCaster with boundary_volume_gpu parameter."
            )

        num_points = point_coords.shape[0]
        num_dirs = directions.shape[0]

        with cp.cuda.Device(self.device):
            if not point_coords.flags.c_contiguous:
                point_coords = cp.ascontiguousarray(point_coords)
            if not directions.flags.c_contiguous:
                directions = cp.ascontiguousarray(directions)
            if point_mask.dtype != cp.int32:
                point_mask = point_mask.astype(cp.int32)
            if max_distances.dtype != cp.float32:
                max_distances = max_distances.astype(cp.float32)

            kernel = self._kernel_meanshift_inflection

            # Convention A: x = points, y = directions
            threads_per_block = (32, 16)
            blocks_x = (num_points + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_y = (num_dirs + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_x, blocks_y)

            args = (
                self._texture.texture_object,            # cudaTextureObject_t ct_tex
                self._boundary_texture.texture_object,   # cudaTextureObject_t diff_tex
                point_coords,                            # float*
                point_mask,                              # int*
                directions,                              # float*
                max_distances,                           # float*
                intensity_weights,                       # float* OUTPUT
                intersection_points,                     # float* OUTPUT
                intersection_distances,                  # float* OUTPUT
                intersection_values,                     # float* OUTPUT
                np.float32(entry_threshold),             # float
                np.float32(gradient_decay_fraction),     # float
                int32(num_steps),                        # int
                np.int64(iter_number),                   # long long
                int32(num_points),                       # int
                int32(num_dirs),                         # int
            )

            if stream is not None:
                kernel(blocks_per_grid, threads_per_block, args, stream=stream)
            else:
                kernel(blocks_per_grid, threads_per_block, args)
                cp.cuda.runtime.deviceSynchronize()


    def cast_meanshift_grace(
        self,  # TextureRayCaster instance
        point_coords: cp.ndarray,
        directions: cp.ndarray,
        point_mask: cp.ndarray,
        max_distances: cp.ndarray,
        intensity_weights: cp.ndarray,
        intersection_points: cp.ndarray,
        intersection_distances: cp.ndarray,
        intersection_values: cp.ndarray,
        boundary_threshold: float = 0.0,
        grace_distance: float = 1.5,
        num_steps: int = 48,
        iter_number: int = 0,
        stream: Optional[cp.cuda.Stream] = None,
    ) -> None:
        """
        Dual-texture branchless mean-shift with grace distance boundary.

        Same as cast_meanshift_branchless, but allows rays to re-enter the
        vessel after brief exits caused by grid aliasing in the boundary map.
        If a ray stays outside for longer than grace_distance, it latches
        permanently at the position where it first exited.

        Parameters
        ----------
        point_coords : cp.ndarray, shape (N, 3)
        directions : cp.ndarray, shape (M, 3)
        point_mask : cp.ndarray, shape (N,)
        max_distances : cp.ndarray, shape (N,)
        intensity_weights : cp.ndarray, shape (N, M) — OUTPUT
        intersection_points : cp.ndarray, shape (N, M, 3) — OUTPUT
        intersection_distances : cp.ndarray, shape (N, M) — OUTPUT
        intersection_values : cp.ndarray, shape (N, M) — OUTPUT
        boundary_threshold : float
            Threshold on difference texture. Default 0.0.
        grace_distance : float
            Maximum distance (in voxels) a ray can travel outside the
            boundary before permanently latching. Should be ~1-2 voxels
            to bridge single-voxel grid aliasing gaps without leaking
            between separate vessels. Default 1.5.
        num_steps : int
            Uniform steps per ray. Default 48.
        iter_number : int
            Jitter seed.
        stream : cp.cuda.Stream, optional
        """
        if self._boundary_texture is None:
            raise RuntimeError(
                "cast_meanshift_grace requires a boundary texture. "
                "Create TextureRayCaster with boundary_volume_gpu parameter."
            )

        num_points = point_coords.shape[0]
        num_dirs = directions.shape[0]

        with cp.cuda.Device(self.device):
            if not point_coords.flags.c_contiguous:
                point_coords = cp.ascontiguousarray(point_coords)
            if not directions.flags.c_contiguous:
                directions = cp.ascontiguousarray(directions)
            if point_mask.dtype != cp.int32:
                point_mask = point_mask.astype(cp.int32)
            if max_distances.dtype != cp.float32:
                max_distances = max_distances.astype(cp.float32)

            kernel = self._kernel_meanshift_grace

            threads_per_block = (32, 16)
            blocks_x = (num_points + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_y = (num_dirs + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_x, blocks_y)

            args = (
                self._texture.texture_object,
                self._boundary_texture.texture_object,
                point_coords,
                point_mask,
                directions,
                max_distances,
                intensity_weights,
                intersection_points,
                intersection_distances,
                intersection_values,
                np.float32(boundary_threshold),
                np.float32(grace_distance),
                int32(num_steps),
                np.int64(iter_number),
                int32(num_points),
                int32(num_dirs),
            )

            if stream is not None:
                kernel(blocks_per_grid, threads_per_block, args, stream=stream)
            else:
                kernel(blocks_per_grid, threads_per_block, args)
                cp.cuda.runtime.deviceSynchronize()


    def destroy(self):
        """Release texture resources."""
        if self._texture is not None:
            self._texture.destroy()
            self._texture = None

        if self._boundary_texture is not None:
            self._boundary_texture.destroy()
            self._boundary_texture = None


    def __del__(self):
        self.destroy()


# =============================================================================
# PREWARM FUNCTION
# =============================================================================

def prewarm_kernels(device: int = 0) -> None:
    """
    Pre-compile ray casting kernels to avoid JIT overhead on first use.
    """
    global _ray_cast_kernel_cache
    
    with cp.cuda.Device(device):
        # Compile kernels by creating them
        if 'ray_cast_texture_kernel' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_texture_kernel'] = cp.RawKernel(
                ray_cast_texture_kernel_code,
                'ray_cast_texture_kernel'
            )
        
        if 'ray_cast_texture_kernel_shared' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_texture_kernel_shared'] = cp.RawKernel(
                ray_cast_texture_kernel_shared_code,
                'ray_cast_texture_kernel_shared'
            )

        if 'ray_cast_difference_branchless_texture' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_difference_branchless_texture'] = cp.RawKernel(
                difference_branchless_kernel_code,
                'ray_cast_difference_branchless_texture'
            )

        if 'ray_cast_gradient_branchless_texture' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_gradient_branchless_texture'] = cp.RawKernel(
                gradient_branchless_kernel_code,
                'ray_cast_gradient_branchless_texture'
            )

        if 'ray_cast_intensity_branchless_texture' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_intensity_branchless_texture'] = cp.RawKernel(
                intensity_branchless_kernel_code,
                'ray_cast_intensity_branchless_texture'
            )

        if 'ray_cast_meanshift_branchless_texture' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_meanshift_branchless_texture'] = cp.RawKernel(
                meanshift_branchless_kernel_code,
                'ray_cast_meanshift_branchless_texture'
            )

        if 'ray_cast_meanshift_inflection_texture' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_meanshift_inflection_texture'] = cp.RawKernel(
                meanshift_inflection_kernel_code,
                'ray_cast_meanshift_inflection_texture'
            )

        if 'ray_cast_meanshift_grace_texture' not in _ray_cast_kernel_cache:
            _ray_cast_kernel_cache['ray_cast_meanshift_grace_texture'] = cp.RawKernel(
                meanshift_grace_kernel_code,
                'ray_cast_meanshift_grace_texture'
            )
            
        print(f"Texture ray cast kernels prewarmed on device {device}")