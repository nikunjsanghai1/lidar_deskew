# Livox Deskew CUDA

CUDA-accelerated ROS2 package for deskewing Livox LiDAR point clouds using interpolated ego motion on NVIDIA AGX Orin.

## Overview

This package provides GPU-accelerated motion deskewing for Livox LiDAR scans. During the ~100ms scan duration, robot motion causes each point to be measured at a different pose. This package transforms all points to a common reference time using CUDA parallel processing.

## Performance

| Metric | CPU Version | CUDA Version |
|--------|-------------|--------------|
| Deskew rate | 1.7 Hz | 10+ Hz |
| Per-scan latency | 600 ms | < 50 ms |
| Points/second | 35k | 210k+ |
| CPU utilization | 100% | < 20% |

## Architecture

```
┌─────────────────────────────────────────┐
│         ROS2 C++ Layer (CPU)            │
│  - Subscribe /livox/lidar, /tf          │
│  - Manage pose buffer                   │
│  - Publish /livox/lidar_deskew          │
└─────────────────────┬───────────────────┘
                      │ H2D Transfer
                      ▼
┌─────────────────────────────────────────┐
│          CUDA C Layer (GPU)             │
│  - Extract timestamps (parallel)        │
│  - Interpolate poses (parallel)         │
│  - Deskew transforms (parallel)         │
└─────────────────────┬───────────────────┘
                      │ D2H Transfer
                      ▼
┌─────────────────────────────────────────┐
│         ROS2 C++ Layer (CPU)            │
│  - Construct PointCloud2                │
│  - Publish deskewed cloud               │
└─────────────────────────────────────────┘
```

## CUDA Kernels

1. **extractTimesKernel**: Extract per-point timestamps in parallel
2. **interpolatePosesKernel**: Batch pose interpolation with parallel binary search + SLERP
3. **deskewKernel**: Parallel point transformation using interpolated poses

## Deskewing Math

Same algorithm as CPU version:

```
Frames:
  W = odom, B(t) = base_link at time t, L = livox_frame

For each point i with timestamp t_i:
  1. T_Bn_Bi = inverse(T_WB_n) * T_WB_i  (relative motion)
  2. p_Bi = T_LB * p_L                    (to base at t_i)
  3. p_Bn = T_Bn_Bi * p_Bi                (warp to t_n)
  4. p_Ln = T_BL * p_Bn                   (back to livox)
```

## Requirements

- NVIDIA AGX Orin (SM_87)
- CUDA Toolkit 11.4+ (JetPack 5.x)
- ROS2 Humble
- Same ROS dependencies as CPU version

## Build

```bash
cd ~/preprocess_ws
colcon build --packages-select livox_deskew_cuda
source install/setup.bash
```

## Usage

### Launch with defaults
```bash
ros2 launch livox_deskew_cuda deskew_cuda.launch.py
```

### Launch with custom parameters
```bash
ros2 launch livox_deskew_cuda deskew_cuda.launch.py \
    input_topic:=/livox/lidar \
    output_topic:=/livox/lidar_deskew \
    odom_frame:=odom \
    base_frame:=base_link \
    lidar_frame:=livox_frame
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_topic` | string | `/livox/lidar` | Input point cloud topic |
| `output_topic` | string | `/livox/lidar_deskew` | Output deskewed cloud topic |
| `odom_frame` | string | `odom` | Odometry frame name |
| `base_frame` | string | `base_link` | Robot base frame name |
| `lidar_frame` | string | `livox_frame` | LiDAR sensor frame name |
| `buffer_seconds` | double | `2.0` | Pose buffer duration |
| `max_missing_ratio` | double | `0.02` | Max failed interpolation ratio |
| `use_tf` | bool | `true` | Use TF for pose data |
| `use_odom_fallback` | bool | `true` | Use /odometry as fallback |

## GPU Memory Usage

Pre-allocated buffers (for 25,000 points):
- Input points: 1.2 MB
- Output XYZ: 300 KB
- Timestamps: 100 KB
- Poses: 700 KB
- Pose buffer: 20 KB
- **Total: ~2.3 MB**

## Topics

### Subscribed
- `/livox/lidar` (sensor_msgs/PointCloud2)
- `/tf`, `/tf_static` (tf2_msgs/TFMessage)
- `/odometry` (nav_msgs/Odometry)

### Published
- `/livox/lidar_deskew` (sensor_msgs/PointCloud2)

## Profiling

```bash
# Profile CUDA kernels
nsys profile ros2 run livox_deskew_cuda livox_deskew_cuda_node

# View kernel timing
nvprof ros2 run livox_deskew_cuda livox_deskew_cuda_node
```

## License

MIT
