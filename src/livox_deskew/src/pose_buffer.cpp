#include "livox_deskew/pose_buffer.hpp"

#include <algorithm>

namespace livox_deskew
{

PoseBuffer::PoseBuffer(double buffer_duration)
: buffer_duration_(buffer_duration)
{
}

void PoseBuffer::addSample(const rclcpp::Time & timestamp, const Pose3D & pose)
{
  std::lock_guard<std::mutex> lock(mutex_);

  // Insert sample maintaining sorted order by timestamp
  PoseSample sample(timestamp, pose);

  if (samples_.empty() || timestamp >= samples_.back().timestamp) {
    samples_.push_back(sample);
  } else {
    // Find insertion point (rare case: out-of-order arrival)
    auto it = std::lower_bound(
      samples_.begin(), samples_.end(), sample,
      [](const PoseSample & a, const PoseSample & b) {
        return a.timestamp < b.timestamp;
      });
    samples_.insert(it, sample);
  }

  // Prune old samples
  prune(timestamp);
}

void PoseBuffer::prune(const rclcpp::Time & current_time)
{
  // Remove samples older than buffer_duration from current_time
  rclcpp::Duration max_age = rclcpp::Duration::from_seconds(buffer_duration_);
  rclcpp::Time oldest_allowed = current_time - max_age;

  while (!samples_.empty() && samples_.front().timestamp < oldest_allowed) {
    samples_.pop_front();
  }
}

bool PoseBuffer::findBrackets(
  const rclcpp::Time & query_time,
  size_t & lower_idx, size_t & upper_idx) const
{
  if (samples_.size() < 2) {
    return false;
  }

  // Check if query_time is outside buffer range
  if (query_time < samples_.front().timestamp || query_time > samples_.back().timestamp) {
    return false;
  }

  // Binary search for upper bound
  auto upper_it = std::upper_bound(
    samples_.begin(), samples_.end(), query_time,
    [](const rclcpp::Time & t, const PoseSample & sample) {
      return t < sample.timestamp;
    });

  if (upper_it == samples_.begin()) {
    // query_time is exactly at or before first sample
    if (query_time == samples_.front().timestamp) {
      lower_idx = 0;
      upper_idx = 0;
      return true;
    }
    return false;
  }

  if (upper_it == samples_.end()) {
    // query_time is at or after last sample
    if (query_time == samples_.back().timestamp) {
      lower_idx = samples_.size() - 1;
      upper_idx = samples_.size() - 1;
      return true;
    }
    return false;
  }

  upper_idx = std::distance(samples_.begin(), upper_it);
  lower_idx = upper_idx - 1;

  return true;
}

bool PoseBuffer::interpolatePose(const rclcpp::Time & query_time, Pose3D & out_pose) const
{
  std::lock_guard<std::mutex> lock(mutex_);

  size_t lower_idx, upper_idx;
  if (!findBrackets(query_time, lower_idx, upper_idx)) {
    return false;
  }

  const PoseSample & lower = samples_[lower_idx];
  const PoseSample & upper = samples_[upper_idx];

  // Handle exact match or same sample case
  if (lower_idx == upper_idx || lower.timestamp == upper.timestamp) {
    out_pose = lower.pose;
    return true;
  }

  // Compute interpolation factor alpha
  double dt_total = (upper.timestamp - lower.timestamp).seconds();
  double dt_query = (query_time - lower.timestamp).seconds();
  double alpha = dt_query / dt_total;

  // Clamp alpha to [0, 1] for numerical safety
  alpha = std::clamp(alpha, 0.0, 1.0);

  // Interpolate: linear for translation, SLERP for rotation
  out_pose = Pose3D::interpolate(lower.pose, upper.pose, alpha);

  return true;
}

size_t PoseBuffer::size() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return samples_.size();
}

bool PoseBuffer::hasEnoughSamples() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return samples_.size() >= 2;
}

std::optional<std::pair<rclcpp::Time, rclcpp::Time>> PoseBuffer::getTimeRange() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (samples_.size() < 2) {
    return std::nullopt;
  }
  return std::make_pair(samples_.front().timestamp, samples_.back().timestamp);
}

void PoseBuffer::clear()
{
  std::lock_guard<std::mutex> lock(mutex_);
  samples_.clear();
}

void PoseBuffer::setBufferDuration(double duration)
{
  std::lock_guard<std::mutex> lock(mutex_);
  buffer_duration_ = duration;
}

}  // namespace livox_deskew
