#ifndef LIVOX_DESKEW__POINT_TIME_EXTRACTOR_HPP_
#define LIVOX_DESKEW__POINT_TIME_EXTRACTOR_HPP_

#include <string>
#include <vector>
#include <optional>

#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace livox_deskew
{

/**
 * @brief Extracts per-point timestamps from Livox PointXYZRTLT point clouds
 *
 * Livox drivers may use different field names and types for per-point time:
 * 1. timestamp (uint64, absolute nanoseconds or microseconds) -> use directly as t_i
 * 2. offset_time (uint32, nanoseconds offset from header.stamp) -> t_i = header.stamp + offset_time
 * 3. time (float32, seconds offset from header.stamp) -> t_i = header.stamp + time
 *
 * This class auto-detects the time field format and extracts timestamps accordingly.
 */
class PointTimeExtractor
{
public:
  /**
   * @brief Time field types supported by Livox drivers
   */
  enum class TimeFieldType
  {
    NONE,             // No time field found
    TIMESTAMP_NS,     // uint64 absolute timestamp in nanoseconds
    TIMESTAMP_US,     // uint64 absolute timestamp in microseconds
    OFFSET_TIME_NS,   // uint32 offset in nanoseconds from header.stamp
    TIME_FLOAT_SEC    // float32 offset in seconds from header.stamp
  };

  PointTimeExtractor() = default;

  /**
   * @brief Configure the extractor based on point cloud field layout
   * @param cloud The point cloud to analyze
   * @return true if a valid time field was found, false otherwise
   *
   * This should be called once when the first cloud arrives, or when
   * the cloud format changes. It detects which time field is present
   * and caches the offset and type for efficient extraction.
   */
  bool configure(const sensor_msgs::msg::PointCloud2 & cloud);

  /**
   * @brief Check if the extractor is configured
   */
  bool isConfigured() const { return configured_; }

  /**
   * @brief Get the detected time field type
   */
  TimeFieldType getTimeFieldType() const { return time_field_type_; }

  /**
   * @brief Get the name of the detected time field
   */
  std::string getTimeFieldName() const { return time_field_name_; }

  /**
   * @brief Extract timestamp for a single point
   * @param point_ptr Pointer to the start of the point data
   * @param header_stamp The cloud header timestamp (used for offset-based fields)
   * @return The absolute timestamp for this point
   *
   * @pre configure() must have been called successfully
   */
  rclcpp::Time getPointTime(
    const uint8_t * point_ptr,
    const rclcpp::Time & header_stamp) const;

  /**
   * @brief Extract all point timestamps from a cloud
   * @param cloud The input point cloud
   * @return Vector of timestamps, one per point
   *
   * @pre configure() must have been called successfully
   */
  std::vector<rclcpp::Time> extractAllTimes(
    const sensor_msgs::msg::PointCloud2 & cloud) const;

  /**
   * @brief Find min and max timestamps in the cloud
   * @param cloud The input point cloud
   * @return Pair of (min_time, max_time)
   *
   * @pre configure() must have been called successfully
   */
  std::pair<rclcpp::Time, rclcpp::Time> getTimeRange(
    const sensor_msgs::msg::PointCloud2 & cloud) const;

private:
  /**
   * @brief Find a field by name in the point cloud
   */
  std::optional<sensor_msgs::msg::PointField> findField(
    const sensor_msgs::msg::PointCloud2 & cloud,
    const std::string & name) const;

  bool configured_ = false;
  TimeFieldType time_field_type_ = TimeFieldType::NONE;
  std::string time_field_name_;
  uint32_t time_field_offset_ = 0;
  uint32_t point_step_ = 0;
};

}  // namespace livox_deskew

#endif  // LIVOX_DESKEW__POINT_TIME_EXTRACTOR_HPP_
