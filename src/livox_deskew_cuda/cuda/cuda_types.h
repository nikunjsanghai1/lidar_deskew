#ifndef LIVOX_DESKEW_CUDA__CUDA_TYPES_H_
#define LIVOX_DESKEW_CUDA__CUDA_TYPES_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum points per scan (pre-allocated)
#define MAX_POINTS 25000

// Maximum pose samples in buffer
#define MAX_POSE_SAMPLES 500

// Time field types
typedef enum {
    TIME_FIELD_OFFSET_NS = 0,    // uint32 nanoseconds offset from header.stamp
    TIME_FIELD_FLOAT_SEC = 1,    // float32 seconds offset from header.stamp
    TIME_FIELD_TIMESTAMP = 2     // float64 absolute timestamp (sensor time)
} TimeFieldType;

// GPU-compatible pose structure (no Eigen, plain C)
// Represents SE(3) transformation
typedef struct {
    float tx, ty, tz;           // Translation (12 bytes)
    float qw, qx, qy, qz;       // Quaternion wxyz (16 bytes)
} Pose3D_GPU;                   // Total: 28 bytes

// GPU pose sample with timestamp
typedef struct {
    double timestamp;           // Time in seconds (8 bytes)
    Pose3D_GPU pose;            // SE(3) pose (28 bytes)
} PoseSample_GPU;               // Total: 36 bytes (+ 4 padding = 40)

// Deskew configuration passed to GPU
typedef struct {
    uint32_t num_points;
    uint32_t point_step;        // Bytes per point in PointCloud2
    uint32_t time_field_offset; // Byte offset to time field
    TimeFieldType time_field_type;
    double header_stamp;        // ROS header timestamp (seconds)
    double scan_duration;       // t_max - t_min (seconds)

    // Field offsets for x, y, z
    uint32_t x_offset;
    uint32_t y_offset;
    uint32_t z_offset;
} DeskewConfig;

// Result codes
typedef enum {
    CUDA_DESKEW_SUCCESS = 0,
    CUDA_DESKEW_ERROR_INIT = -1,
    CUDA_DESKEW_ERROR_MEMORY = -2,
    CUDA_DESKEW_ERROR_KERNEL = -3,
    CUDA_DESKEW_ERROR_INVALID_INPUT = -4,
    CUDA_DESKEW_ERROR_NO_POSES = -5
} CudaDeskewResult;

#ifdef __cplusplus
}
#endif

#endif  // LIVOX_DESKEW_CUDA__CUDA_TYPES_H_
