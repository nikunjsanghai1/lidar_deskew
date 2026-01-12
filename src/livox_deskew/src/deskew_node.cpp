#include "livox_deskew/deskew_node.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace livox_deskew
{

DeskewNode::DeskewNode(const rclcpp::NodeOptions & options)
: Node("livox_deskew_node", options),
  pose_buffer_(2.0)
{
  // Declare parameters
  input_topic_ = this->declare_parameter<std::string>("input_topic", "/livox/lidar");
  output_topic_ = this->declare_parameter<std::string>("output_topic", "/livox/lidar_deskew");
  odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
  base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
  lidar_frame_ = this->declare_parameter<std::string>("lidar_frame", "livox_frame");
  buffer_seconds_ = this->declare_parameter<double>("buffer_seconds", 2.0);
  max_missing_ratio_ = this->declare_parameter<double>("max_missing_ratio", 0.02);
  use_tf_ = this->declare_parameter<bool>("use_tf", true);
  use_odom_fallback_ = this->declare_parameter<bool>("use_odom_fallback", true);

  // Configure pose buffer
  pose_buffer_.setBufferDuration(buffer_seconds_);

  // Initialize TF2 for static transform lookup
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

  // Publisher
  cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);

  // Subscribers
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    input_topic_, rclcpp::SensorDataQoS(),
    std::bind(&DeskewNode::cloudCallback, this, std::placeholders::_1));

  if (use_tf_) {
    // Subscribe to /tf directly to capture all transforms as they arrive
    tf_sub_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf", 100,
      std::bind(&DeskewNode::tfCallback, this, std::placeholders::_1));

    // Subscribe to /tf_static for static transforms
    tf_static_sub_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf_static", rclcpp::QoS(100).transient_local(),
      std::bind(&DeskewNode::tfCallback, this, std::placeholders::_1));
  }

  if (use_odom_fallback_) {
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odometry", 100,
      std::bind(&DeskewNode::odomCallback, this, std::placeholders::_1));
  }

  RCLCPP_INFO(this->get_logger(), "Livox deskew node initialized");
  RCLCPP_INFO(this->get_logger(), "  Input: %s -> Output: %s", input_topic_.c_str(), output_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "  Frames: %s -> %s -> %s", odom_frame_.c_str(), base_frame_.c_str(), lidar_frame_.c_str());
  RCLCPP_INFO(this->get_logger(), "  Buffer duration: %.2f s, Max missing ratio: %.2f", buffer_seconds_, max_missing_ratio_);
}

Pose3D DeskewNode::geometryPoseToPose3D(const geometry_msgs::msg::Pose & pose) const
{
  Eigen::Vector3d translation(pose.position.x, pose.position.y, pose.position.z);
  Eigen::Quaterniond rotation(
    pose.orientation.w,
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z);
  return Pose3D(translation, rotation);
}

Pose3D DeskewNode::transformStampedToPose3D(const geometry_msgs::msg::TransformStamped & tf) const
{
  Eigen::Vector3d translation(
    tf.transform.translation.x,
    tf.transform.translation.y,
    tf.transform.translation.z);
  Eigen::Quaterniond rotation(
    tf.transform.rotation.w,
    tf.transform.rotation.x,
    tf.transform.rotation.y,
    tf.transform.rotation.z);
  return Pose3D(translation, rotation);
}

bool DeskewNode::initializeStaticTransforms()
{
  if (static_tf_initialized_) {
    return true;
  }

  try {
    // Get T_BL: base_link -> livox_frame
    auto tf_msg = tf_buffer_->lookupTransform(
      base_frame_, lidar_frame_, tf2::TimePointZero);
    T_BL_ = transformStampedToPose3D(tf_msg);
    T_LB_ = T_BL_.inverse();
    static_tf_initialized_ = true;
    RCLCPP_INFO(this->get_logger(), "Static transform %s -> %s initialized",
      base_frame_.c_str(), lidar_frame_.c_str());
    return true;
  } catch (const tf2::TransformException & ex) {
    // Static transform not yet available
    return false;
  }
}

void DeskewNode::tfCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg)
{
  for (const auto & transform : msg->transforms) {
    // Extract odom -> base_link transforms for the pose buffer
    if (transform.header.frame_id == odom_frame_ &&
      transform.child_frame_id == base_frame_)
    {
      Pose3D pose = transformStampedToPose3D(transform);
      rclcpp::Time timestamp(transform.header.stamp);
      pose_buffer_.addSample(timestamp, pose);
    }
  }
}

void DeskewNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  // Only use odometry if TF is not available or as fallback
  // Check if pose buffer already has recent samples from TF
  auto time_range = pose_buffer_.getTimeRange();
  rclcpp::Time odom_time(msg->header.stamp);

  // If we have TF data within 100ms, skip odometry
  if (time_range.has_value()) {
    double age = (odom_time - time_range->second).seconds();
    if (age < 0.1 && use_tf_) {
      return;  // TF data is fresh, skip odometry
    }
  }

  Pose3D pose = geometryPoseToPose3D(msg->pose.pose);
  pose_buffer_.addSample(odom_time, pose);
}

void DeskewNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  // Initialize static transforms if not done
  if (!initializeStaticTransforms()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "Waiting for static transform %s -> %s", base_frame_.c_str(), lidar_frame_.c_str());
    return;
  }

  // Configure time extractor on first cloud
  if (!time_extractor_configured_) {
    if (!time_extractor_.configure(*msg)) {
      RCLCPP_ERROR(this->get_logger(),
        "Failed to configure point time extractor. No valid time field found in point cloud. "
        "Expected one of: timestamp (uint64), offset_time (uint32), time (float32)");
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Point time extractor configured with field: %s",
      time_extractor_.getTimeFieldName().c_str());
    time_extractor_configured_ = true;
  }

  // Check pose buffer has data
  if (!pose_buffer_.hasEnoughSamples()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "Pose buffer has insufficient samples for interpolation. Waiting for TF/odometry data.");
    return;
  }

  // Deskew the cloud
  sensor_msgs::msg::PointCloud2 output;
  if (deskewCloud(*msg, output)) {
    cloud_pub_->publish(output);
  }
}

bool DeskewNode::deskewCloud(
  const sensor_msgs::msg::PointCloud2 & input,
  sensor_msgs::msg::PointCloud2 & output)
{
  size_t num_points = input.width * input.height;
  if (num_points == 0) {
    return false;
  }

  // Get raw time range from the cloud (may be in sensor time domain)
  auto [raw_t_min, raw_t_max] = time_extractor_.getTimeRange(input);

  // Compute scan duration (relative offset) - this is time-domain independent
  double scan_duration = (raw_t_max - raw_t_min).seconds();

  // Use header.stamp as base time (ROS time domain)
  // Reference time t_n = header.stamp + scan_duration (end of scan in ROS time)
  rclcpp::Time header_stamp(input.header.stamp);
  rclcpp::Time t_n = header_stamp + rclcpp::Duration::from_seconds(scan_duration);

  // Store raw_t_min for computing per-point relative offsets
  double raw_t_min_sec = raw_t_min.seconds();

  // Get interpolated pose at reference time t_n
  Pose3D T_WB_n;
  if (!pose_buffer_.interpolatePose(t_n, T_WB_n)) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "Cannot interpolate pose at reference time t_n=%.3f (header=%.3f, duration=%.6f). Dropping scan.",
      t_n.seconds(), header_stamp.seconds(), scan_duration);
    return false;
  }

  // Precompute inverse of T_WB_n
  Pose3D T_WB_n_inv = T_WB_n.inverse();

  // Initialize output cloud (copy structure from input)
  output = input;
  output.header.stamp = t_n;
  output.header.frame_id = lidar_frame_;

  // Find x, y, z field offsets
  uint32_t x_offset = 0, y_offset = 0, z_offset = 0;
  bool found_xyz = false;
  for (const auto & field : input.fields) {
    if (field.name == "x") {
      x_offset = field.offset;
    } else if (field.name == "y") {
      y_offset = field.offset;
    } else if (field.name == "z") {
      z_offset = field.offset;
    }
  }
  for (const auto & field : input.fields) {
    if (field.name == "x" || field.name == "y" || field.name == "z") {
      found_xyz = true;
      break;
    }
  }
  if (!found_xyz) {
    RCLCPP_ERROR(this->get_logger(), "Point cloud missing x, y, z fields");
    return false;
  }

  // Process each point
  size_t missing_count = 0;
  uint32_t point_step = input.point_step;

  for (size_t i = 0; i < num_points; ++i) {
    const uint8_t * in_ptr = input.data.data() + (i * point_step);
    uint8_t * out_ptr = output.data.data() + (i * point_step);

    // Get raw point timestamp (may be in sensor time domain)
    rclcpp::Time raw_t_i = time_extractor_.getPointTime(in_ptr, header_stamp);

    // Convert to ROS time: compute relative offset from scan start, add to header.stamp
    double relative_offset = raw_t_i.seconds() - raw_t_min_sec;
    rclcpp::Time t_i = header_stamp + rclcpp::Duration::from_seconds(relative_offset);

    // Get interpolated pose at t_i (now in ROS time domain)
    Pose3D T_WB_i;
    if (!pose_buffer_.interpolatePose(t_i, T_WB_i)) {
      // Interpolation failed for this point - copy unchanged (per user preference)
      missing_count++;
      continue;  // Point data already copied via output = input
    }

    // Read original point in livox_frame
    float x_L, y_L, z_L;
    std::memcpy(&x_L, in_ptr + x_offset, sizeof(float));
    std::memcpy(&y_L, in_ptr + y_offset, sizeof(float));
    std::memcpy(&z_L, in_ptr + z_offset, sizeof(float));
    Eigen::Vector3d p_i_L(x_L, y_L, z_L);

    // Deskewing math:
    // T_Bn_Bi = inverse(T_WB_n) * T_WB_i
    // p_i_Bi = T_LB * p_i_L          (point in base at t_i)
    // p_i_Bn = T_Bn_Bi * p_i_Bi      (warped to base at t_n)
    // p_i_Ln = T_BL * p_i_Bn         (back to livox_frame at t_n)

    Pose3D T_Bn_Bi = T_WB_n_inv * T_WB_i;
    Eigen::Vector3d p_i_Bi = T_LB_.transformPoint(p_i_L);
    Eigen::Vector3d p_i_Bn = T_Bn_Bi.transformPoint(p_i_Bi);
    Eigen::Vector3d p_i_Ln = T_BL_.transformPoint(p_i_Bn);

    // Write deskewed point
    float x_out = static_cast<float>(p_i_Ln.x());
    float y_out = static_cast<float>(p_i_Ln.y());
    float z_out = static_cast<float>(p_i_Ln.z());
    std::memcpy(out_ptr + x_offset, &x_out, sizeof(float));
    std::memcpy(out_ptr + y_offset, &y_out, sizeof(float));
    std::memcpy(out_ptr + z_offset, &z_out, sizeof(float));

    // Note: reflectivity, tag, line fields are unchanged (already copied)
  }

  // Check missing ratio
  double missing_ratio = static_cast<double>(missing_count) / num_points;
  if (missing_ratio > max_missing_ratio_) {
    RCLCPP_WARN(this->get_logger(),
      "Missing ratio %.3f exceeds threshold %.3f. Dropping scan. "
      "(%zu/%zu points failed interpolation)",
      missing_ratio, max_missing_ratio_, missing_count, num_points);
    return false;
  }

  if (missing_count > 0) {
    RCLCPP_DEBUG(this->get_logger(),
      "Deskewed cloud with %zu/%zu points copied unchanged (interpolation failed)",
      missing_count, num_points);
  }

  return true;
}

}  // namespace livox_deskew
