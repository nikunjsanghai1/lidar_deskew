#include "livox_deskew/point_time_extractor.hpp"

#include <stdexcept>
#include <limits>
#include <cstring>

namespace livox_deskew
{

std::optional<sensor_msgs::msg::PointField> PointTimeExtractor::findField(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const std::string & name) const
{
  for (const auto & field : cloud.fields) {
    if (field.name == name) {
      return field;
    }
  }
  return std::nullopt;
}

bool PointTimeExtractor::configure(const sensor_msgs::msg::PointCloud2 & cloud)
{
  configured_ = false;
  time_field_type_ = TimeFieldType::NONE;
  time_field_name_.clear();
  time_field_offset_ = 0;
  point_step_ = cloud.point_step;

  // Priority 1: Check for absolute timestamp field
  // Note: ROS2 PointField doesn't have UINT64/INT64 types
  // Livox drivers may use FLOAT64 for absolute timestamps (as double seconds)
  // or store as two UINT32 fields, or use a custom encoding
  if (auto field = findField(cloud, "timestamp")) {
    if (field->datatype == sensor_msgs::msg::PointField::FLOAT64) {
      // timestamp as double (seconds)
      time_field_type_ = TimeFieldType::TIMESTAMP_NS;  // We'll convert from double seconds
      time_field_name_ = "timestamp";
      time_field_offset_ = field->offset;
      configured_ = true;
      return true;
    } else if (field->datatype == sensor_msgs::msg::PointField::UINT32) {
      // Some drivers use UINT32 with count=2 to store uint64
      // Or it could be a relative timestamp in ns
      time_field_type_ = TimeFieldType::OFFSET_TIME_NS;
      time_field_name_ = "timestamp";
      time_field_offset_ = field->offset;
      configured_ = true;
      return true;
    }
  }

  // Priority 2: Check for offset_time field (uint32 nanoseconds offset)
  if (auto field = findField(cloud, "offset_time")) {
    if (field->datatype == sensor_msgs::msg::PointField::UINT32 ||
      field->datatype == sensor_msgs::msg::PointField::INT32)
    {
      time_field_type_ = TimeFieldType::OFFSET_TIME_NS;
      time_field_name_ = "offset_time";
      time_field_offset_ = field->offset;
      configured_ = true;
      return true;
    }
  }

  // Priority 3: Check for time field (float32 seconds offset)
  if (auto field = findField(cloud, "time")) {
    if (field->datatype == sensor_msgs::msg::PointField::FLOAT32) {
      time_field_type_ = TimeFieldType::TIME_FLOAT_SEC;
      time_field_name_ = "time";
      time_field_offset_ = field->offset;
      configured_ = true;
      return true;
    }
  }

  // Priority 4: Check for t field (alternative name for time)
  if (auto field = findField(cloud, "t")) {
    if (field->datatype == sensor_msgs::msg::PointField::FLOAT32) {
      time_field_type_ = TimeFieldType::TIME_FLOAT_SEC;
      time_field_name_ = "t";
      time_field_offset_ = field->offset;
      configured_ = true;
      return true;
    } else if (field->datatype == sensor_msgs::msg::PointField::UINT32 ||
      field->datatype == sensor_msgs::msg::PointField::INT32)
    {
      time_field_type_ = TimeFieldType::OFFSET_TIME_NS;
      time_field_name_ = "t";
      time_field_offset_ = field->offset;
      configured_ = true;
      return true;
    }
  }

  // No valid time field found - fail loudly as per specification
  return false;
}

rclcpp::Time PointTimeExtractor::getPointTime(
  const uint8_t * point_ptr,
  const rclcpp::Time & header_stamp) const
{
  if (!configured_) {
    throw std::runtime_error("PointTimeExtractor not configured. Call configure() first.");
  }

  const uint8_t * time_ptr = point_ptr + time_field_offset_;

  switch (time_field_type_) {
    case TimeFieldType::TIMESTAMP_NS:
      {
        // Stored as FLOAT64 (double seconds) - may be absolute sensor time
        // Return raw value; deskew_node will convert to ROS time via relative offset
        double timestamp_sec;
        std::memcpy(&timestamp_sec, time_ptr, sizeof(double));
        int32_t sec = static_cast<int32_t>(timestamp_sec);
        uint32_t nsec = static_cast<uint32_t>((timestamp_sec - sec) * 1e9);
        return rclcpp::Time(sec, nsec, header_stamp.get_clock_type());
      }

    case TimeFieldType::TIMESTAMP_US:
      {
        // Same as TIMESTAMP_NS
        double timestamp_sec;
        std::memcpy(&timestamp_sec, time_ptr, sizeof(double));
        int32_t sec = static_cast<int32_t>(timestamp_sec);
        uint32_t nsec = static_cast<uint32_t>((timestamp_sec - sec) * 1e9);
        return rclcpp::Time(sec, nsec, header_stamp.get_clock_type());
      }

    case TimeFieldType::OFFSET_TIME_NS:
      {
        uint32_t offset_ns;
        std::memcpy(&offset_ns, time_ptr, sizeof(uint32_t));
        return header_stamp + rclcpp::Duration(0, offset_ns);
      }

    case TimeFieldType::TIME_FLOAT_SEC:
      {
        float offset_sec;
        std::memcpy(&offset_sec, time_ptr, sizeof(float));
        int64_t offset_ns = static_cast<int64_t>(offset_sec * 1e9);
        return header_stamp + rclcpp::Duration::from_nanoseconds(offset_ns);
      }

    case TimeFieldType::NONE:
    default:
      throw std::runtime_error("Invalid time field type");
  }
}

std::vector<rclcpp::Time> PointTimeExtractor::extractAllTimes(
  const sensor_msgs::msg::PointCloud2 & cloud) const
{
  if (!configured_) {
    throw std::runtime_error("PointTimeExtractor not configured. Call configure() first.");
  }

  size_t num_points = cloud.width * cloud.height;
  std::vector<rclcpp::Time> times;
  times.reserve(num_points);

  rclcpp::Time header_stamp(cloud.header.stamp);

  for (size_t i = 0; i < num_points; ++i) {
    const uint8_t * point_ptr = cloud.data.data() + (i * point_step_);
    times.push_back(getPointTime(point_ptr, header_stamp));
  }

  return times;
}

std::pair<rclcpp::Time, rclcpp::Time> PointTimeExtractor::getTimeRange(
  const sensor_msgs::msg::PointCloud2 & cloud) const
{
  if (!configured_) {
    throw std::runtime_error("PointTimeExtractor not configured. Call configure() first.");
  }

  size_t num_points = cloud.width * cloud.height;
  if (num_points == 0) {
    rclcpp::Time header_stamp(cloud.header.stamp);
    return {header_stamp, header_stamp};
  }

  rclcpp::Time header_stamp(cloud.header.stamp);

  // Initialize with first point
  const uint8_t * first_ptr = cloud.data.data();
  rclcpp::Time min_time = getPointTime(first_ptr, header_stamp);
  rclcpp::Time max_time = min_time;

  for (size_t i = 1; i < num_points; ++i) {
    const uint8_t * point_ptr = cloud.data.data() + (i * point_step_);
    rclcpp::Time t = getPointTime(point_ptr, header_stamp);
    if (t < min_time) {
      min_time = t;
    }
    if (t > max_time) {
      max_time = t;
    }
  }

  return {min_time, max_time};
}

}  // namespace livox_deskew
