#ifndef LIVOX_DESKEW_CUDA__CUDA_DESKEW_H_
#define LIVOX_DESKEW_CUDA__CUDA_DESKEW_H_

#include "cuda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize CUDA resources for deskewing
 *
 * Pre-allocates GPU memory for:
 * - Input point cloud data
 * - Output deskewed coordinates
 * - Per-point timestamps
 * - Per-point interpolated poses
 * - Pose buffer copy
 * - Static transforms
 *
 * @return CUDA_DESKEW_SUCCESS on success, error code otherwise
 */
CudaDeskewResult cuda_deskew_init(void);

/**
 * @brief Cleanup all CUDA resources
 */
void cuda_deskew_cleanup(void);

/**
 * @brief Check if CUDA deskew is initialized
 * @return 1 if initialized, 0 otherwise
 */
int cuda_deskew_is_initialized(void);

/**
 * @brief Upload pose buffer samples to GPU
 *
 * @param samples Array of pose samples (CPU memory)
 * @param num_samples Number of samples in array
 * @return CUDA_DESKEW_SUCCESS on success, error code otherwise
 */
CudaDeskewResult cuda_upload_pose_buffer(
    const PoseSample_GPU* samples,
    size_t num_samples
);

/**
 * @brief Upload static transforms to GPU
 *
 * @param T_BL Transform from base_link to livox_frame
 * @param T_LB Transform from livox_frame to base_link (inverse of T_BL)
 * @return CUDA_DESKEW_SUCCESS on success, error code otherwise
 */
CudaDeskewResult cuda_upload_static_transforms(
    const Pose3D_GPU* T_BL,
    const Pose3D_GPU* T_LB
);

/**
 * @brief Main deskewing function - processes entire point cloud on GPU
 *
 * Pipeline:
 * 1. Copy input points and pre-computed times to GPU (async)
 * 2. Interpolate poses kernel
 * 3. Deskew transform kernel
 * 4. Copy results back to CPU
 *
 * @param points_in Raw PointCloud2 data buffer (CPU pinned memory preferred)
 * @param times_relative Pre-computed relative timestamps (0.0 to scan_duration)
 *                       Must be pre-allocated with num_points floats
 * @param config Deskew configuration (point count, offsets, etc.)
 * @param xyz_out Output buffer for deskewed XYZ coordinates [x0,y0,z0,x1,y1,z1,...]
 *                Must be pre-allocated with 3 * num_points floats
 * @return CUDA_DESKEW_SUCCESS on success, error code otherwise
 */
CudaDeskewResult cuda_deskew_cloud(
    const uint8_t* points_in,
    const float* times_relative,
    const DeskewConfig* config,
    float* xyz_out
);

/**
 * @brief Get the reference pose (T_WB at t_n) after deskewing
 *
 * Useful for publishing the correct output timestamp
 *
 * @param pose_out Output pose structure
 * @return CUDA_DESKEW_SUCCESS on success, error code otherwise
 */
CudaDeskewResult cuda_get_reference_pose(Pose3D_GPU* pose_out);

/**
 * @brief Synchronize CUDA stream (wait for completion)
 */
void cuda_deskew_sync(void);

/**
 * @brief Get last CUDA error message
 * @return String describing last error, or NULL if no error
 */
const char* cuda_deskew_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif  // LIVOX_DESKEW_CUDA__CUDA_DESKEW_H_
