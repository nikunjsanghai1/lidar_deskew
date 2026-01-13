#ifndef LIVOX_DESKEW_CUDA__POSE_BUFFER_HPP_
#define LIVOX_DESKEW_CUDA__POSE_BUFFER_HPP_

#include <deque>
#include <mutex>
#include <optional>
#include <vector>

#include <rclcpp/time.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "cuda_types.h"

namespace livox_deskew_cuda
{

/**
 * @brief SE(3) pose representation using Eigen types (CPU-side)
 */
struct Pose3D
{
  Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond rotation{Eigen::Quaterniond::Identity()};

  Pose3D() = default;

  Pose3D(const Eigen::Vector3d & t, const Eigen::Quaterniond & q)
  : translation(t), rotation(q.normalized())
  {
  }

  /**
   * @brief Convert to GPU-compatible format
   */
  Pose3D_GPU toGPU() const
  {
    Pose3D_GPU gpu_pose;
    gpu_pose.tx = static_cast<float>(translation.x());
    gpu_pose.ty = static_cast<float>(translation.y());
    gpu_pose.tz = static_cast<float>(translation.z());
    gpu_pose.qw = static_cast<float>(rotation.w());
    gpu_pose.qx = static_cast<float>(rotation.x());
    gpu_pose.qy = static_cast<float>(rotation.y());
    gpu_pose.qz = static_cast<float>(rotation.z());
    return gpu_pose;
  }

  /**
   * @brief Compute inverse of this pose
   */
  Pose3D inverse() const
  {
    Eigen::Quaterniond q_inv = rotation.inverse();
    Eigen::Vector3d t_inv = -(q_inv * translation);
    return Pose3D(t_inv, q_inv);
  }
};

/**
 * @brief Timestamped pose sample
 */
struct PoseSample
{
  rclcpp::Time timestamp;
  Pose3D pose;

  PoseSample() = default;

  PoseSample(const rclcpp::Time & t, const Pose3D & p)
  : timestamp(t), pose(p)
  {
  }

  /**
   * @brief Convert to GPU-compatible format
   */
  PoseSample_GPU toGPU() const
  {
    PoseSample_GPU gpu_sample;
    gpu_sample.timestamp = timestamp.seconds();
    gpu_sample.pose = pose.toGPU();
    return gpu_sample;
  }
};

/**
 * @brief Thread-safe buffer for storing poses (CPU-side)
 *
 * Provides methods to export to GPU-compatible format for CUDA processing.
 */
class PoseBuffer
{
public:
  explicit PoseBuffer(double buffer_duration = 2.0);

  void addSample(const rclcpp::Time & timestamp, const Pose3D & pose);
  size_t size() const;
  bool hasEnoughSamples() const;
  std::optional<std::pair<rclcpp::Time, rclcpp::Time>> getTimeRange() const;
  void clear();
  void setBufferDuration(double duration);

  /**
   * @brief Export buffer to GPU-compatible format
   *
   * Thread-safe: locks mutex during export
   * @return Vector of GPU-compatible pose samples
   */
  std::vector<PoseSample_GPU> exportToGPU() const;

  /**
   * @brief Get number of samples for GPU export
   */
  size_t getExportSize() const;

private:
  void prune(const rclcpp::Time & current_time);

  std::deque<PoseSample> samples_;
  double buffer_duration_;
  mutable std::mutex mutex_;
};

}  // namespace livox_deskew_cuda

#endif  // LIVOX_DESKEW_CUDA__POSE_BUFFER_HPP_
