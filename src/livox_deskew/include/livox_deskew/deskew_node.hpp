#ifndef LIVOX_DESKEW__DESKEW_NODE_HPP_
#define LIVOX_DESKEW__DESKEW_NODE_HPP_

#include <memory>
#include <string>
#include <optional>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "livox_deskew/pose_buffer.hpp"
#include "livox_deskew/point_time_extractor.hpp"
#include "livox_deskew/math_types.hpp"

namespace livox_deskew
{

/**
 * @brief ROS2 node for deskewing Livox LiDAR point clouds using interpolated ego motion
 *
 * DESKEWING MATH:
 * ==============
 *
 * Frames:
 *   W = odom (world/odometry frame)
 *   B(t) = base_link at time t
 *   L = livox_frame
 *
 * Static extrinsic:
 *   T_BL = transform from base_link to livox_frame (cached from TF static)
 *   T_LB = inverse(T_BL)
 *
 * Dynamic pose:
 *   T_WB(t) = transform from odom to base_link at time t (from pose buffer via interpolation)
 *
 * Deskew formulas:
 *   Given point i:
 *     - raw point p_i_L in livox_frame
 *     - timestamp t_i
 *
 *   Reference time:
 *     t_n = max(t_i) in scan
 *
 *   Query interpolated robot poses:
 *     T_WB_n = T_WB(t_n)
 *     T_WB_i = T_WB(t_i)
 *
 *   Relative motion from base at t_i to base at t_n:
 *     T_Bn_Bi = inverse(T_WB_n) * T_WB_i
 *
 *   Convert point to base coordinates at its measurement time:
 *     p_i_Bi = T_LB * p_i_L
 *
 *   Warp point into base coordinates at t_n:
 *     p_i_Bn = T_Bn_Bi * p_i_Bi
 *
 *   Convert deskewed point back to livox_frame at t_n:
 *     p_i_Ln = T_BL * p_i_Bn
 *
 *   Publish p_i_Ln in livox_frame, stamped at t_n.
 *
 * INTERPOLATION REQUIREMENT:
 * =========================
 * TF and odometry are published at discrete rates (100 Hz). Per-point timestamps
 * can be at arbitrary times. Direct TF lookup per point would cause jitter.
 * Solution: Build a PoseBuffer from TF samples and interpolate to each point time.
 * - Translation: linear interpolation
 * - Rotation: SLERP (spherical linear interpolation)
 */
class DeskewNode : public rclcpp::Node
{
public:
  explicit DeskewNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // Callbacks
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void tfCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

  // Initialization
  bool initializeStaticTransforms();
  Pose3D geometryPoseToPose3D(const geometry_msgs::msg::Pose & pose) const;
  Pose3D transformStampedToPose3D(const geometry_msgs::msg::TransformStamped & tf) const;

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

  // Static transforms (cached)
  bool static_tf_initialized_ = false;
  Pose3D T_BL_;  // base_link -> livox_frame
  Pose3D T_LB_;  // livox_frame -> base_link

  // Pose buffer for interpolation
  PoseBuffer pose_buffer_;

  // Point time extraction
  PointTimeExtractor time_extractor_;
  bool time_extractor_configured_ = false;

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

}  // namespace livox_deskew

#endif  // LIVOX_DESKEW__DESKEW_NODE_HPP_
