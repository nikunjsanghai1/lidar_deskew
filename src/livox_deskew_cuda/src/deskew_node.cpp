#include "livox_deskew_cuda/deskew_node.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cstring>
#include <limits>

namespace livox_deskew_cuda
{

DeskewNode::DeskewNode(const rclcpp::NodeOptions & options)
: Node("livox_deskew_cuda_node", options),
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

  pose_buffer_.setBufferDuration(buffer_seconds_);

  // Pre-allocate output buffer
  xyz_output_buffer_.resize(MAX_POINTS * 3);

  // Initialize TF2
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

  // Initialize CUDA
  if (!initializeCuda()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize CUDA. Node will not process clouds.");
  }

  // Publisher
  cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 10);

  // Subscribers
  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    input_topic_, rclcpp::SensorDataQoS(),
    std::bind(&DeskewNode::cloudCallback, this, std::placeholders::_1));

  if (use_tf_) {
    tf_sub_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf", 100,
      std::bind(&DeskewNode::tfCallback, this, std::placeholders::_1));

    tf_static_sub_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf_static", rclcpp::QoS(100).transient_local(),
      std::bind(&DeskewNode::tfCallback, this, std::placeholders::_1));
  }

  if (use_odom_fallback_) {
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odometry", 100,
      std::bind(&DeskewNode::odomCallback, this, std::placeholders::_1));
  }

  RCLCPP_INFO(this->get_logger(), "CUDA Livox deskew node initialized");
  RCLCPP_INFO(this->get_logger(), "  Input: %s -> Output: %s", input_topic_.c_str(), output_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "  Frames: %s -> %s -> %s",
    odom_frame_.c_str(), base_frame_.c_str(), lidar_frame_.c_str());
  RCLCPP_INFO(this->get_logger(), "  CUDA initialized: %s", cuda_initialized_ ? "yes" : "no");
}

DeskewNode::~DeskewNode()
{
  if (cuda_initialized_) {
    cuda_deskew_cleanup();
  }
}

bool DeskewNode::initializeCuda()
{
  CudaDeskewResult result = cuda_deskew_init();
  if (result != CUDA_DESKEW_SUCCESS) {
    const char* error = cuda_deskew_get_last_error();
    RCLCPP_ERROR(this->get_logger(), "CUDA init failed: %s", error ? error : "unknown error");
    return false;
  }
  cuda_initialized_ = true;
  return true;
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
    auto tf_msg = tf_buffer_->lookupTransform(
      base_frame_, lidar_frame_, tf2::TimePointZero);
    T_BL_ = transformStampedToPose3D(tf_msg);
    T_LB_ = T_BL_.inverse();

    // Convert to GPU format
    T_BL_gpu_ = T_BL_.toGPU();
    T_LB_gpu_ = T_LB_.toGPU();

    // Upload to GPU
    CudaDeskewResult result = cuda_upload_static_transforms(&T_BL_gpu_, &T_LB_gpu_);
    if (result != CUDA_DESKEW_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Failed to upload static transforms to GPU");
      return false;
    }

    static_tf_initialized_ = true;
    RCLCPP_INFO(this->get_logger(), "Static transform %s -> %s initialized and uploaded to GPU",
      base_frame_.c_str(), lidar_frame_.c_str());
    return true;
  } catch (const tf2::TransformException & ex) {
    return false;
  }
}

void DeskewNode::tfCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg)
{
  for (const auto & transform : msg->transforms) {
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
  auto time_range = pose_buffer_.getTimeRange();
  rclcpp::Time odom_time(msg->header.stamp);

  if (time_range.has_value()) {
    double age = (odom_time - time_range->second).seconds();
    if (age < 0.1 && use_tf_) {
      return;
    }
  }

  Pose3D pose = geometryPoseToPose3D(msg->pose.pose);
  pose_buffer_.addSample(odom_time, pose);
}

bool DeskewNode::detectTimeField(const sensor_msgs::msg::PointCloud2 & cloud)
{
  time_field_configured_ = false;

  // Find x, y, z offsets
  for (const auto & field : cloud.fields) {
    if (field.name == "x") x_offset_ = field.offset;
    else if (field.name == "y") y_offset_ = field.offset;
    else if (field.name == "z") z_offset_ = field.offset;
  }

  // Detect time field
  for (const auto & field : cloud.fields) {
    if (field.name == "timestamp" && field.datatype == sensor_msgs::msg::PointField::FLOAT64) {
      time_field_type_ = TIME_FIELD_TIMESTAMP;
      time_field_offset_ = field.offset;
      time_field_configured_ = true;
      RCLCPP_INFO(this->get_logger(), "Time field: timestamp (float64) at offset %u", time_field_offset_);
      return true;
    }
  }

  for (const auto & field : cloud.fields) {
    if (field.name == "offset_time" &&
      (field.datatype == sensor_msgs::msg::PointField::UINT32 ||
       field.datatype == sensor_msgs::msg::PointField::INT32))
    {
      time_field_type_ = TIME_FIELD_OFFSET_NS;
      time_field_offset_ = field.offset;
      time_field_configured_ = true;
      RCLCPP_INFO(this->get_logger(), "Time field: offset_time (uint32 ns) at offset %u", time_field_offset_);
      return true;
    }
  }

  for (const auto & field : cloud.fields) {
    if (field.name == "time" && field.datatype == sensor_msgs::msg::PointField::FLOAT32) {
      time_field_type_ = TIME_FIELD_FLOAT_SEC;
      time_field_offset_ = field.offset;
      time_field_configured_ = true;
      RCLCPP_INFO(this->get_logger(), "Time field: time (float32 sec) at offset %u", time_field_offset_);
      return true;
    }
  }

  for (const auto & field : cloud.fields) {
    if (field.name == "t") {
      if (field.datatype == sensor_msgs::msg::PointField::FLOAT32) {
        time_field_type_ = TIME_FIELD_FLOAT_SEC;
        time_field_offset_ = field.offset;
        time_field_configured_ = true;
        RCLCPP_INFO(this->get_logger(), "Time field: t (float32 sec) at offset %u", time_field_offset_);
        return true;
      } else if (field.datatype == sensor_msgs::msg::PointField::UINT32) {
        time_field_type_ = TIME_FIELD_OFFSET_NS;
        time_field_offset_ = field.offset;
        time_field_configured_ = true;
        RCLCPP_INFO(this->get_logger(), "Time field: t (uint32 ns) at offset %u", time_field_offset_);
        return true;
      }
    }
  }

  RCLCPP_ERROR(this->get_logger(), "No valid time field found in point cloud");
  return false;
}

void DeskewNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  if (!cuda_initialized_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "CUDA not initialized, dropping cloud");
    return;
  }

  if (!initializeStaticTransforms()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "Waiting for static transform %s -> %s", base_frame_.c_str(), lidar_frame_.c_str());
    return;
  }

  if (!time_field_configured_) {
    if (!detectTimeField(*msg)) {
      return;
    }
  }

  if (!pose_buffer_.hasEnoughSamples()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "Pose buffer has insufficient samples (%zu). Waiting for TF/odometry data.",
      pose_buffer_.size());
    return;
  }

  // Debug: Log pose buffer status
  auto time_range = pose_buffer_.getTimeRange();
  if (time_range.has_value()) {
    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "Processing cloud: %zu points, pose buffer: %zu samples [%.3f - %.3f]",
      static_cast<size_t>(msg->width * msg->height),
      pose_buffer_.size(),
      time_range->first.seconds(),
      time_range->second.seconds());
  }

  sensor_msgs::msg::PointCloud2 output;
  if (deskewCloud(*msg, output)) {
    cloud_pub_->publish(output);
    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "Published deskewed cloud");
  }
}

bool DeskewNode::deskewCloud(
  const sensor_msgs::msg::PointCloud2 & input,
  sensor_msgs::msg::PointCloud2 & output)
{
  size_t num_points = input.width * input.height;
  if (num_points == 0 || num_points > MAX_POINTS) {
    RCLCPP_WARN(this->get_logger(), "Invalid point count: %zu", num_points);
    return false;
  }

  // Export pose buffer to GPU format and upload
  std::vector<PoseSample_GPU> gpu_poses = pose_buffer_.exportToGPU();
  if (gpu_poses.size() < 2) {
    RCLCPP_WARN(this->get_logger(), "Not enough pose samples for interpolation");
    return false;
  }

  CudaDeskewResult result = cuda_upload_pose_buffer(gpu_poses.data(), gpu_poses.size());
  if (result != CUDA_DESKEW_SUCCESS) {
    RCLCPP_ERROR(this->get_logger(), "Failed to upload pose buffer: %s",
      cuda_deskew_get_last_error());
    return false;
  }

  // Compute relative timestamps on CPU to avoid float32 precision loss
  std::vector<float> relative_times(num_points);
  double t_min = std::numeric_limits<double>::max();
  double t_max = std::numeric_limits<double>::lowest();

  // First pass: read all timestamps and find min/max
  // Note: TIME_FIELD_TIMESTAMP stores nanoseconds as float64, must convert to seconds
  for (size_t i = 0; i < num_points; ++i) {
    const uint8_t* pt = input.data.data() + (i * input.point_step);
    double t = 0.0;

    if (time_field_type_ == TIME_FIELD_TIMESTAMP) {
      double t_ns;
      std::memcpy(&t_ns, pt + time_field_offset_, sizeof(double));
      t = t_ns * 1e-9;  // Convert nanoseconds to seconds
    } else if (time_field_type_ == TIME_FIELD_OFFSET_NS) {
      uint32_t t_ns;
      std::memcpy(&t_ns, pt + time_field_offset_, sizeof(uint32_t));
      t = static_cast<double>(t_ns) * 1e-9;
    } else {  // TIME_FIELD_FLOAT_SEC
      float t_f;
      std::memcpy(&t_f, pt + time_field_offset_, sizeof(float));
      t = static_cast<double>(t_f);
    }

    if (t < t_min) t_min = t;
    if (t > t_max) t_max = t;
  }

  // Second pass: compute relative offsets from t_min
  // Use double arithmetic to preserve precision before final float conversion
  for (size_t i = 0; i < num_points; ++i) {
    const uint8_t* pt = input.data.data() + (i * input.point_step);
    double t = 0.0;

    if (time_field_type_ == TIME_FIELD_TIMESTAMP) {
      double t_ns;
      std::memcpy(&t_ns, pt + time_field_offset_, sizeof(double));
      t = t_ns * 1e-9;  // Convert nanoseconds to seconds
    } else if (time_field_type_ == TIME_FIELD_OFFSET_NS) {
      uint32_t t_ns;
      std::memcpy(&t_ns, pt + time_field_offset_, sizeof(uint32_t));
      t = static_cast<double>(t_ns) * 1e-9;
    } else {
      float t_f;
      std::memcpy(&t_f, pt + time_field_offset_, sizeof(float));
      t = static_cast<double>(t_f);
    }

    // Relative offset from scan start (0.0 to scan_duration)
    relative_times[i] = static_cast<float>(t - t_min);
  }

  // Scan duration = t_max - t_min
  double scan_duration = t_max - t_min;
  if (scan_duration <= 0.0) {
    scan_duration = 0.1;  // Fallback to 100ms if all timestamps are the same
  }

  // Prepare config
  DeskewConfig config;
  config.num_points = static_cast<uint32_t>(num_points);
  config.point_step = input.point_step;
  config.time_field_offset = time_field_offset_;
  config.time_field_type = time_field_type_;
  config.header_stamp = rclcpp::Time(input.header.stamp).seconds();
  config.scan_duration = scan_duration;
  config.x_offset = x_offset_;
  config.y_offset = y_offset_;
  config.z_offset = z_offset_;

  // Run CUDA deskewing with pre-computed relative times
  result = cuda_deskew_cloud(input.data.data(), relative_times.data(), &config, xyz_output_buffer_.data());
  if (result != CUDA_DESKEW_SUCCESS) {
    RCLCPP_WARN(this->get_logger(), "CUDA deskew failed: %s", cuda_deskew_get_last_error());
    return false;
  }

  // Construct output message
  output = input;  // Copy all fields, metadata
  // Use input stamp + scan duration for output timestamp
  // Reference time is end of scan (t_n = t_min + scan_duration in ROS time)
  rclcpp::Time input_stamp(input.header.stamp);
  output.header.stamp = input_stamp + rclcpp::Duration::from_seconds(scan_duration);
  output.header.frame_id = lidar_frame_;

  // Write deskewed XYZ back to output
  for (size_t i = 0; i < num_points; ++i) {
    uint8_t* point_ptr = output.data.data() + (i * input.point_step);
    float x = xyz_output_buffer_[i * 3 + 0];
    float y = xyz_output_buffer_[i * 3 + 1];
    float z = xyz_output_buffer_[i * 3 + 2];
    std::memcpy(point_ptr + x_offset_, &x, sizeof(float));
    std::memcpy(point_ptr + y_offset_, &y, sizeof(float));
    std::memcpy(point_ptr + z_offset_, &z, sizeof(float));
  }

  return true;
}

}  // namespace livox_deskew_cuda
