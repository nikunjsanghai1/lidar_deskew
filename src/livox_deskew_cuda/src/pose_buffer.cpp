#include "livox_deskew_cuda/pose_buffer.hpp"

#include <algorithm>

namespace livox_deskew_cuda
{

PoseBuffer::PoseBuffer(double buffer_duration)
: buffer_duration_(buffer_duration)
{
}

void PoseBuffer::addSample(const rclcpp::Time & timestamp, const Pose3D & pose)
{
  std::lock_guard<std::mutex> lock(mutex_);

  PoseSample sample(timestamp, pose);

  if (samples_.empty() || timestamp >= samples_.back().timestamp) {
    samples_.push_back(sample);
  } else {
    auto it = std::lower_bound(
      samples_.begin(), samples_.end(), sample,
      [](const PoseSample & a, const PoseSample & b) {
        return a.timestamp < b.timestamp;
      });
    samples_.insert(it, sample);
  }

  prune(timestamp);
}

void PoseBuffer::prune(const rclcpp::Time & current_time)
{
  rclcpp::Duration max_age = rclcpp::Duration::from_seconds(buffer_duration_);
  rclcpp::Time oldest_allowed = current_time - max_age;

  while (!samples_.empty() && samples_.front().timestamp < oldest_allowed) {
    samples_.pop_front();
  }
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

std::vector<PoseSample_GPU> PoseBuffer::exportToGPU() const
{
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<PoseSample_GPU> gpu_samples;
  gpu_samples.reserve(samples_.size());

  for (const auto & sample : samples_) {
    gpu_samples.push_back(sample.toGPU());
  }

  return gpu_samples;
}

size_t PoseBuffer::getExportSize() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return samples_.size();
}

}  // namespace livox_deskew_cuda
