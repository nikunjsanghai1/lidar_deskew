#ifndef LIVOX_DESKEW_CUDA__DESKEW_NODE_HPP_
#define LIVOX_DESKEW_CUDA__DESKEW_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "livox_deskew_cuda/pose_buffer.hpp"
#include "cuda_types.h"
#include "cuda_deskew.h"

namespace livox_deskew_cuda
{

/**
 * @brief ROS2 node for CUDA-accelerated LiDAR deskewing
 *
 * This node subscribes to raw Livox point clouds, deskews them using
 * GPU-accelerated interpolation and transformation, and publishes
 * the corrected point clouds.
 *
 * DESKEWING MATH (same as CPU version):
 * - T_Bn_Bi = inverse(T_WB_n) * T_WB_i
 * - p_i_Bi = T_LB * p_i_L
 * - p_i_Bn = T_Bn_Bi * p_i_Bi
 * - p_i_Ln = T_BL * p_i_Bn
 */
class DeskewNode : public rclcpp::Node
{
public:
  explicit DeskewNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~DeskewNode();

private:
  // Callbacks
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void tfCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

  // Initialization
  bool initializeStaticTransforms();
  bool initializeCuda();

  // Helpers
  Pose3D geometryPoseToPose3D(const geometry_msgs::msg::Pose & pose) const;
  Pose3D transformStampedToPose3D(const geometry_msgs::msg::TransformStamped & tf) const;
  bool detectTimeField(const sensor_msgs::msg::PointCloud2 & cloud);

  // Deskewing
  bool deskewCloud(
    const sensor_msgs::msg::PointCloud2 & input,
    sensor_msgs::msg::PointCloud2 & output);

  // Parameters
  std::string input_topic_;
  std::string output_topic_;
  std::string odom_frame_;
  std::string base_frame_;
  std::string lidar_frame_;
  double buffer_seconds_;
  double max_missing_ratio_;
  bool use_tf_;
  bool use_odom_fallback_;
  bool rosbag_mode_;  // true: measure processing time, false: measure wall clock latency

  // Static transforms (cached)
  bool static_tf_initialized_ = false;
  Pose3D T_BL_;  // base_link -> livox_frame
  Pose3D T_LB_;  // livox_frame -> base_link
  Pose3D_GPU T_BL_gpu_;
  Pose3D_GPU T_LB_gpu_;

  // Pose buffer (CPU-side)
  PoseBuffer pose_buffer_;

  // Time field detection
  bool time_field_configured_ = false;
  TimeFieldType time_field_type_;
  uint32_t time_field_offset_ = 0;
  uint32_t x_offset_ = 0;
  uint32_t y_offset_ = 0;
  uint32_t z_offset_ = 0;

  // CUDA state
  bool cuda_initialized_ = false;

  // Output buffer (pre-allocated)
  std::vector<float> xyz_output_buffer_;

  // TF2 for static transform lookup
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_static_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  // Publisher
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
};

}  // namespace livox_deskew_cuda

#endif  // LIVOX_DESKEW_CUDA__DESKEW_NODE_HPP_
