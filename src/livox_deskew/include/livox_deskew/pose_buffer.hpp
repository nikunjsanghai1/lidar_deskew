#ifndef LIVOX_DESKEW__POSE_BUFFER_HPP_
#define LIVOX_DESKEW__POSE_BUFFER_HPP_

#include <deque>
#include <mutex>
#include <optional>

#include <rclcpp/time.hpp>

#include "livox_deskew/math_types.hpp"

namespace livox_deskew
{

/**
 * @brief Thread-safe buffer for storing and interpolating poses
 *
 * This class stores pose samples at discrete timestamps (e.g., 100 Hz from TF or odometry)
 * and provides interpolation to arbitrary query times using:
 * - Linear interpolation for translation
 * - Spherical linear interpolation (SLERP) for rotation
 *
 * CRITICAL: This interpolation is required because TF and odometry are published at
 * discrete rates (100 Hz), but LiDAR points have per-point timestamps. Direct TF
 * lookups at per-point times would cause jitter due to the discrete nature of TF.
 */
class PoseBuffer
{
public:
  /**
   * @brief Constructor
   * @param buffer_duration Maximum age of samples to keep (seconds)
   */
  explicit PoseBuffer(double buffer_duration = 2.0);

  /**
   * @brief Add a new pose sample to the buffer
   * @param timestamp Time of the pose sample
   * @param pose The SE(3) pose at this timestamp
   *
   * Automatically prunes samples older than buffer_duration.
   */
  void addSample(const rclcpp::Time & timestamp, const Pose3D & pose);

  /**
   * @brief Interpolate pose at arbitrary query time
   * @param query_time Time at which to interpolate the pose
   * @param out_pose Output pose (only valid if return is true)
   * @return true if interpolation successful, false if query_time outside buffer range
   *
   * Interpolation method:
   * - Find bracketing samples t_k <= query_time <= t_{k+1}
   * - Compute alpha = (query_time - t_k) / (t_{k+1} - t_k)
   * - Translation: p(t) = (1 - alpha) * p_k + alpha * p_{k+1}
   * - Rotation: q(t) = slerp(q_k, q_{k+1}, alpha)
   */
  bool interpolatePose(const rclcpp::Time & query_time, Pose3D & out_pose) const;

  /**
   * @brief Get the number of samples currently in the buffer
   */
  size_t size() const;

  /**
   * @brief Check if buffer has enough samples for interpolation
   */
  bool hasEnoughSamples() const;

  /**
   * @brief Get the time range covered by the buffer
   * @return Optional pair of (oldest_time, newest_time), empty if buffer has < 2 samples
   */
  std::optional<std::pair<rclcpp::Time, rclcpp::Time>> getTimeRange() const;

  /**
   * @brief Clear all samples from the buffer
   */
  void clear();

  /**
   * @brief Set buffer duration
   */
  void setBufferDuration(double duration);

private:
  /**
   * @brief Remove samples older than buffer_duration from the latest sample
   * @param current_time Reference time for pruning
   */
  void prune(const rclcpp::Time & current_time);

  /**
   * @brief Find bracketing samples for interpolation (internal, assumes mutex held)
   * @param query_time Time to find brackets for
   * @param lower_idx Output index of lower bracket sample
   * @param upper_idx Output index of upper bracket sample
   * @return true if brackets found
   */
  bool findBrackets(
    const rclcpp::Time & query_time,
    size_t & lower_idx, size_t & upper_idx) const;

  std::deque<PoseSample> samples_;
  double buffer_duration_;
  mutable std::mutex mutex_;
};

}  // namespace livox_deskew

#endif  // LIVOX_DESKEW__POSE_BUFFER_HPP_
