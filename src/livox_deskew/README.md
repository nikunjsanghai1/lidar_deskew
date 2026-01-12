# Livox Deskew

ROS2 package for deskewing Livox LiDAR point clouds using interpolated ego motion.

## Overview

This package corrects motion distortion in Livox LiDAR scans. During the ~100ms scan duration, robot motion causes each point to be measured at a different pose. This package transforms all points to a common reference time using interpolated pose data.

## Critical: Interpolation Requirement

**TF and /odometry are published at 100 Hz (discrete samples). LiDAR points have per-point timestamps. Direct TF lookup at each point time would cause jitter due to the discrete nature of TF.**

**Solution**: Build a pose buffer from TF samples and interpolate to each point's timestamp:
- Translation: Linear interpolation
- Rotation: Spherical linear interpolation (SLERP)

This ensures smooth, jitter-free deskewing.

## Input Point Format

Livox PointXYZRTLT fields:
- x, y, z (float32)
- reflectivity
- tag
- line
- **time field** (one of the following, detected automatically):
  - `timestamp` (uint64): Absolute timestamp in nanoseconds
  - `offset_time` (uint32): Offset in nanoseconds from header.stamp
  - `time` (float32): Offset in seconds from header.stamp

## Deskewing Math

### Frames

- W = odom (world/odometry frame)
- B(t) = base_link at time t
- L = livox_frame

### Static Extrinsic

TF provides T_B_L (base_link -> livox_frame). Cache both:
- T_BL = base -> livox
- T_LB = inverse(T_BL)

### Dynamic Pose

Pose buffer provides T_WB(t) (odom -> base_link) for any t via interpolation.

### Deskew Formulas

Given point i:
- raw point p_i_L in livox_frame
- timestamp t_i

Reference time:
- t_n = max(t_i) in scan

Query interpolated robot poses:
- T_WB_n = T_WB(t_n)
- T_WB_i = T_WB(t_i)

Relative motion from base at t_i to base at t_n:
```
T_Bn_Bi = inverse(T_WB_n) * T_WB_i
```

Convert point to base coordinates at its measurement time:
```
p_i_Bi = T_LB * p_i_L
```

Warp point into base coordinates at t_n:
```
p_i_Bn = T_Bn_Bi * p_i_Bi
```

Convert deskewed point back to livox_frame at t_n:
```
p_i_Ln = T_BL * p_i_Bn
```

Publish p_i_Ln in livox_frame, stamped at t_n.

## Dependencies

- ROS2 Humble
- rclcpp
- sensor_msgs
- nav_msgs
- geometry_msgs
- tf2
- tf2_ros
- tf2_msgs
- tf2_geometry_msgs
- builtin_interfaces
- Eigen3

## Build

```bash
cd ~/preprocess_ws
colcon build --packages-select livox_deskew
source install/setup.bash
```

## Usage

### Launch with defaults

```bash
ros2 launch livox_deskew deskew.launch.py
```

### Launch with custom parameters

```bash
ros2 launch livox_deskew deskew.launch.py \
    input_topic:=/livox/lidar \
    output_topic:=/livox/lidar_deskew \
    odom_frame:=odom \
    base_frame:=base_link \
    lidar_frame:=livox_frame
```

### Run node directly

```bash
ros2 run livox_deskew livox_deskew_node --ros-args \
    -p input_topic:=/livox/lidar \
    -p output_topic:=/livox/lidar_deskew
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_topic` | string | `/livox/lidar` | Input point cloud topic |
| `output_topic` | string | `/livox/lidar_deskew` | Output deskewed point cloud topic |
| `odom_frame` | string | `odom` | Odometry frame name |
| `base_frame` | string | `base_link` | Robot base frame name |
| `lidar_frame` | string | `livox_frame` | LiDAR sensor frame name |
| `buffer_seconds` | double | `2.0` | Pose buffer duration in seconds |
| `max_missing_ratio` | double | `0.02` | Max ratio of points that can fail interpolation before dropping scan |
| `use_tf` | bool | `true` | Use TF for pose data |
| `use_odom_fallback` | bool | `true` | Use /odometry as fallback when TF unavailable |

## Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/livox/lidar` | sensor_msgs/PointCloud2 | Input Livox point cloud |
| `/tf` | tf2_msgs/TFMessage | Dynamic transforms (odom -> base_link) |
| `/tf_static` | tf2_msgs/TFMessage | Static transforms (base_link -> livox_frame) |
| `/odometry` | nav_msgs/Odometry | Fallback pose source |

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `/livox/lidar_deskew` | sensor_msgs/PointCloud2 | Deskewed point cloud |

## Output

- `header.stamp` = t_n (max point timestamp = end of scan)
- `header.frame_id` = "livox_frame"
- All point fields (reflectivity, tag, line) preserved

## Failure Handling

- If pose interpolation fails at t_n: drop scan with warning
- If interpolation fails for individual points: copy unchanged
- If > max_missing_ratio of points fail: drop scan

## Testing

```bash
colcon test --packages-select livox_deskew
colcon test-result --verbose
```

## License

MIT
