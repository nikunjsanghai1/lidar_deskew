#include "cuda_deskew.h"
#include "cuda_math.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// Thread block size
#define BLOCK_SIZE 256

// Global state
static int g_initialized = 0;
static char g_last_error[256] = {0};

// Device memory pointers
static uint8_t* d_points_in = NULL;      // Raw point cloud data
static float* d_times = NULL;            // Per-point relative timestamps
static Pose3D_GPU* d_poses = NULL;       // Per-point interpolated poses
static float* d_xyz_out = NULL;          // Output XYZ coordinates

// Pose buffer on device
static PoseSample_GPU* d_pose_buffer = NULL;
static size_t d_pose_buffer_size = 0;

// Static transforms on device
static Pose3D_GPU* d_T_BL = NULL;        // base_link -> livox_frame
static Pose3D_GPU* d_T_LB = NULL;        // livox_frame -> base_link

// Reference pose (T_WB at t_n)
static Pose3D_GPU* d_T_WB_n = NULL;
static Pose3D_GPU* d_T_WB_n_inv = NULL;

// Host-side copy of reference pose
static Pose3D_GPU h_T_WB_n;

// CUDA stream
static cudaStream_t g_stream = 0;

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        snprintf(g_last_error, sizeof(g_last_error), \
            "CUDA error at %s:%d: %s", __FILE__, __LINE__, \
            cudaGetErrorString(err)); \
        return CUDA_DESKEW_ERROR_KERNEL; \
    } \
} while(0)

#define CUDA_CHECK_INIT(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        snprintf(g_last_error, sizeof(g_last_error), \
            "CUDA error at %s:%d: %s", __FILE__, __LINE__, \
            cudaGetErrorString(err)); \
        cuda_deskew_cleanup(); \
        return CUDA_DESKEW_ERROR_INIT; \
    } \
} while(0)

//=============================================================================
// CUDA Kernels
//=============================================================================

/**
 * Kernel 1: Extract per-point timestamps
 *
 * Each thread processes one point, extracting its relative timestamp
 */
__global__ void extractTimesKernel(
    const uint8_t* __restrict__ points_in,
    uint32_t point_step,
    uint32_t time_offset,
    TimeFieldType time_type,
    size_t num_points,
    float* __restrict__ times_out)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    const uint8_t* point_ptr = points_in + (idx * point_step);
    const uint8_t* time_ptr = point_ptr + time_offset;

    float time_value = 0.0f;

    switch (time_type) {
        case TIME_FIELD_OFFSET_NS: {
            uint32_t offset_ns;
            memcpy(&offset_ns, time_ptr, sizeof(uint32_t));
            time_value = offset_ns * 1e-9f;
            break;
        }
        case TIME_FIELD_FLOAT_SEC: {
            memcpy(&time_value, time_ptr, sizeof(float));
            break;
        }
        case TIME_FIELD_TIMESTAMP: {
            double timestamp;
            memcpy(&timestamp, time_ptr, sizeof(double));
            // Will be converted to relative offset by host
            time_value = (float)timestamp;
            break;
        }
    }

    times_out[idx] = time_value;
}

/**
 * Kernel 2: Batch pose interpolation
 *
 * Each thread interpolates the pose at one point's timestamp using
 * binary search + SLERP
 */
__global__ void interpolatePosesKernel(
    const float* __restrict__ query_times,
    size_t num_points,
    const PoseSample_GPU* __restrict__ buffer,
    size_t buffer_size,
    double header_stamp,
    Pose3D_GPU* __restrict__ poses_out)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // Query time in ROS time domain
    double query_t = header_stamp + (double)query_times[idx];

    // Binary search for bracketing samples
    int lower = 0;
    int upper = (int)buffer_size - 1;

    // Check bounds
    if (query_t <= buffer[0].timestamp) {
        poses_out[idx] = buffer[0].pose;
        return;
    }
    if (query_t >= buffer[buffer_size - 1].timestamp) {
        poses_out[idx] = buffer[buffer_size - 1].pose;
        return;
    }

    // Binary search
    while (upper - lower > 1) {
        int mid = (lower + upper) / 2;
        if (buffer[mid].timestamp <= query_t) {
            lower = mid;
        } else {
            upper = mid;
        }
    }

    // Interpolate between buffer[lower] and buffer[upper]
    double dt_total = buffer[upper].timestamp - buffer[lower].timestamp;
    double dt_query = query_t - buffer[lower].timestamp;

    float alpha = 0.0f;
    if (dt_total > 1e-9) {
        alpha = (float)(dt_query / dt_total);
        alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
    }

    pose_interpolate(&buffer[lower].pose, &buffer[upper].pose, alpha, &poses_out[idx]);
}

/**
 * Kernel 3: Deskew transformation
 *
 * Each thread transforms one point using the deskewing formula:
 * 1. T_Bn_Bi = inv(T_WB_n) * T_WB_i
 * 2. p_Bi = T_LB * p_L
 * 3. p_Bn = T_Bn_Bi * p_Bi
 * 4. p_Ln = T_BL * p_Bn
 */
__global__ void deskewKernel(
    const uint8_t* __restrict__ points_in,
    uint32_t point_step,
    uint32_t x_offset,
    uint32_t y_offset,
    uint32_t z_offset,
    const Pose3D_GPU* __restrict__ poses_i,
    const Pose3D_GPU* __restrict__ T_WB_n_inv,
    const Pose3D_GPU* __restrict__ T_BL,
    const Pose3D_GPU* __restrict__ T_LB,
    size_t num_points,
    float* __restrict__ xyz_out)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    const uint8_t* point_ptr = points_in + (idx * point_step);

    // Read original point in livox_frame
    float p_L_x, p_L_y, p_L_z;
    memcpy(&p_L_x, point_ptr + x_offset, sizeof(float));
    memcpy(&p_L_y, point_ptr + y_offset, sizeof(float));
    memcpy(&p_L_z, point_ptr + z_offset, sizeof(float));

    // Step 1: Compute relative motion T_Bn_Bi = inv(T_WB_n) * T_WB_i
    Pose3D_GPU T_Bn_Bi;
    pose_compose(T_WB_n_inv, &poses_i[idx], &T_Bn_Bi);

    // Step 2: Transform point to base coordinates at t_i: p_Bi = T_LB * p_L
    float p_Bi_x, p_Bi_y, p_Bi_z;
    transform_point(T_LB, p_L_x, p_L_y, p_L_z, &p_Bi_x, &p_Bi_y, &p_Bi_z);

    // Step 3: Warp point into base coordinates at t_n: p_Bn = T_Bn_Bi * p_Bi
    float p_Bn_x, p_Bn_y, p_Bn_z;
    transform_point(&T_Bn_Bi, p_Bi_x, p_Bi_y, p_Bi_z, &p_Bn_x, &p_Bn_y, &p_Bn_z);

    // Step 4: Transform back to livox_frame: p_Ln = T_BL * p_Bn
    float p_Ln_x, p_Ln_y, p_Ln_z;
    transform_point(T_BL, p_Bn_x, p_Bn_y, p_Bn_z, &p_Ln_x, &p_Ln_y, &p_Ln_z);

    // Write output
    xyz_out[idx * 3 + 0] = p_Ln_x;
    xyz_out[idx * 3 + 1] = p_Ln_y;
    xyz_out[idx * 3 + 2] = p_Ln_z;
}

/**
 * Kernel to compute reference pose (T_WB at t_n = header_stamp + scan_duration)
 * Single thread kernel
 */
__global__ void computeReferencePoseKernel(
    const PoseSample_GPU* __restrict__ buffer,
    size_t buffer_size,
    double t_n,
    Pose3D_GPU* __restrict__ T_WB_n_out,
    Pose3D_GPU* __restrict__ T_WB_n_inv_out)
{
    // Single thread
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Binary search for t_n
    int lower = 0;
    int upper = (int)buffer_size - 1;

    if (t_n <= buffer[0].timestamp) {
        *T_WB_n_out = buffer[0].pose;
        pose_inverse(T_WB_n_out, T_WB_n_inv_out);
        return;
    }
    if (t_n >= buffer[buffer_size - 1].timestamp) {
        *T_WB_n_out = buffer[buffer_size - 1].pose;
        pose_inverse(T_WB_n_out, T_WB_n_inv_out);
        return;
    }

    while (upper - lower > 1) {
        int mid = (lower + upper) / 2;
        if (buffer[mid].timestamp <= t_n) {
            lower = mid;
        } else {
            upper = mid;
        }
    }

    double dt_total = buffer[upper].timestamp - buffer[lower].timestamp;
    double dt_query = t_n - buffer[lower].timestamp;

    float alpha = 0.0f;
    if (dt_total > 1e-9) {
        alpha = (float)(dt_query / dt_total);
        alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
    }

    pose_interpolate(&buffer[lower].pose, &buffer[upper].pose, alpha, T_WB_n_out);
    pose_inverse(T_WB_n_out, T_WB_n_inv_out);
}

//=============================================================================
// Public API Implementation
//=============================================================================

CudaDeskewResult cuda_deskew_init(void)
{
    if (g_initialized) {
        return CUDA_DESKEW_SUCCESS;
    }

    // Create stream
    CUDA_CHECK_INIT(cudaStreamCreate(&g_stream));

    // Allocate device memory
    size_t points_bytes = MAX_POINTS * 48;  // Conservative point size
    size_t times_bytes = MAX_POINTS * sizeof(float);
    size_t poses_bytes = MAX_POINTS * sizeof(Pose3D_GPU);
    size_t xyz_bytes = MAX_POINTS * 3 * sizeof(float);
    size_t pose_buffer_bytes = MAX_POSE_SAMPLES * sizeof(PoseSample_GPU);

    CUDA_CHECK_INIT(cudaMalloc(&d_points_in, points_bytes));
    CUDA_CHECK_INIT(cudaMalloc(&d_times, times_bytes));
    CUDA_CHECK_INIT(cudaMalloc(&d_poses, poses_bytes));
    CUDA_CHECK_INIT(cudaMalloc(&d_xyz_out, xyz_bytes));
    CUDA_CHECK_INIT(cudaMalloc(&d_pose_buffer, pose_buffer_bytes));
    CUDA_CHECK_INIT(cudaMalloc(&d_T_BL, sizeof(Pose3D_GPU)));
    CUDA_CHECK_INIT(cudaMalloc(&d_T_LB, sizeof(Pose3D_GPU)));
    CUDA_CHECK_INIT(cudaMalloc(&d_T_WB_n, sizeof(Pose3D_GPU)));
    CUDA_CHECK_INIT(cudaMalloc(&d_T_WB_n_inv, sizeof(Pose3D_GPU)));

    g_initialized = 1;
    return CUDA_DESKEW_SUCCESS;
}

void cuda_deskew_cleanup(void)
{
    if (g_stream) {
        cudaStreamDestroy(g_stream);
        g_stream = 0;
    }

    if (d_points_in) { cudaFree(d_points_in); d_points_in = NULL; }
    if (d_times) { cudaFree(d_times); d_times = NULL; }
    if (d_poses) { cudaFree(d_poses); d_poses = NULL; }
    if (d_xyz_out) { cudaFree(d_xyz_out); d_xyz_out = NULL; }
    if (d_pose_buffer) { cudaFree(d_pose_buffer); d_pose_buffer = NULL; }
    if (d_T_BL) { cudaFree(d_T_BL); d_T_BL = NULL; }
    if (d_T_LB) { cudaFree(d_T_LB); d_T_LB = NULL; }
    if (d_T_WB_n) { cudaFree(d_T_WB_n); d_T_WB_n = NULL; }
    if (d_T_WB_n_inv) { cudaFree(d_T_WB_n_inv); d_T_WB_n_inv = NULL; }

    d_pose_buffer_size = 0;
    g_initialized = 0;
}

int cuda_deskew_is_initialized(void)
{
    return g_initialized;
}

CudaDeskewResult cuda_upload_pose_buffer(
    const PoseSample_GPU* samples,
    size_t num_samples)
{
    if (!g_initialized) {
        snprintf(g_last_error, sizeof(g_last_error), "CUDA not initialized");
        return CUDA_DESKEW_ERROR_INIT;
    }

    if (num_samples > MAX_POSE_SAMPLES) {
        num_samples = MAX_POSE_SAMPLES;
    }

    CUDA_CHECK(cudaMemcpyAsync(d_pose_buffer, samples,
        num_samples * sizeof(PoseSample_GPU),
        cudaMemcpyHostToDevice, g_stream));

    d_pose_buffer_size = num_samples;
    return CUDA_DESKEW_SUCCESS;
}

CudaDeskewResult cuda_upload_static_transforms(
    const Pose3D_GPU* T_BL,
    const Pose3D_GPU* T_LB)
{
    if (!g_initialized) {
        snprintf(g_last_error, sizeof(g_last_error), "CUDA not initialized");
        return CUDA_DESKEW_ERROR_INIT;
    }

    CUDA_CHECK(cudaMemcpyAsync(d_T_BL, T_BL, sizeof(Pose3D_GPU),
        cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_T_LB, T_LB, sizeof(Pose3D_GPU),
        cudaMemcpyHostToDevice, g_stream));

    return CUDA_DESKEW_SUCCESS;
}

CudaDeskewResult cuda_deskew_cloud(
    const uint8_t* points_in,
    const float* times_relative,
    const DeskewConfig* config,
    float* xyz_out)
{
    if (!g_initialized) {
        snprintf(g_last_error, sizeof(g_last_error), "CUDA not initialized");
        return CUDA_DESKEW_ERROR_INIT;
    }

    if (!points_in || !times_relative || !config || !xyz_out) {
        snprintf(g_last_error, sizeof(g_last_error), "Invalid input pointers");
        return CUDA_DESKEW_ERROR_INVALID_INPUT;
    }

    if (config->num_points == 0 || config->num_points > MAX_POINTS) {
        snprintf(g_last_error, sizeof(g_last_error),
            "Invalid point count: %u (max %d)", config->num_points, MAX_POINTS);
        return CUDA_DESKEW_ERROR_INVALID_INPUT;
    }

    if (d_pose_buffer_size < 2) {
        snprintf(g_last_error, sizeof(g_last_error),
            "Pose buffer too small: %zu samples", d_pose_buffer_size);
        return CUDA_DESKEW_ERROR_NO_POSES;
    }

    size_t num_points = config->num_points;
    size_t input_bytes = num_points * config->point_step;

    // Calculate grid dimensions
    dim3 block(BLOCK_SIZE);
    dim3 grid((num_points + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Step 1: Copy input points to device
    CUDA_CHECK(cudaMemcpyAsync(d_points_in, points_in, input_bytes,
        cudaMemcpyHostToDevice, g_stream));

    // Step 2: Copy pre-computed relative timestamps to device
    CUDA_CHECK(cudaMemcpyAsync(d_times, times_relative,
        num_points * sizeof(float),
        cudaMemcpyHostToDevice, g_stream));

    // Step 3: Compute reference pose at t_n
    double t_n = config->header_stamp + config->scan_duration;
    computeReferencePoseKernel<<<1, 1, 0, g_stream>>>(
        d_pose_buffer,
        d_pose_buffer_size,
        t_n,
        d_T_WB_n,
        d_T_WB_n_inv
    );

    // Step 4: Interpolate poses for all points
    interpolatePosesKernel<<<grid, block, 0, g_stream>>>(
        d_times,
        num_points,
        d_pose_buffer,
        d_pose_buffer_size,
        config->header_stamp,
        d_poses
    );

    // Step 5: Deskew all points
    deskewKernel<<<grid, block, 0, g_stream>>>(
        d_points_in,
        config->point_step,
        config->x_offset,
        config->y_offset,
        config->z_offset,
        d_poses,
        d_T_WB_n_inv,
        d_T_BL,
        d_T_LB,
        num_points,
        d_xyz_out
    );

    // Step 6: Copy results back to host
    CUDA_CHECK(cudaMemcpyAsync(xyz_out, d_xyz_out,
        num_points * 3 * sizeof(float),
        cudaMemcpyDeviceToHost, g_stream));

    // Copy reference pose for output timestamp
    CUDA_CHECK(cudaMemcpyAsync(&h_T_WB_n, d_T_WB_n, sizeof(Pose3D_GPU),
        cudaMemcpyDeviceToHost, g_stream));

    // Synchronize
    CUDA_CHECK(cudaStreamSynchronize(g_stream));

    return CUDA_DESKEW_SUCCESS;
}

CudaDeskewResult cuda_get_reference_pose(Pose3D_GPU* pose_out)
{
    if (!pose_out) {
        return CUDA_DESKEW_ERROR_INVALID_INPUT;
    }
    *pose_out = h_T_WB_n;
    return CUDA_DESKEW_SUCCESS;
}

void cuda_deskew_sync(void)
{
    if (g_stream) {
        cudaStreamSynchronize(g_stream);
    }
}

const char* cuda_deskew_get_last_error(void)
{
    return g_last_error[0] ? g_last_error : NULL;
}
