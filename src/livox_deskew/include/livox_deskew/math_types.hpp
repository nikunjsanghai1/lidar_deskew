#ifndef LIVOX_DESKEW__MATH_TYPES_HPP_
#define LIVOX_DESKEW__MATH_TYPES_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <rclcpp/time.hpp>

namespace livox_deskew
{

/**
 * @brief SE(3) pose representation using Eigen types
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
   * @brief Get the 4x4 homogeneous transformation matrix
   */
  Eigen::Affine3d toAffine() const
  {
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.translation() = translation;
    transform.linear() = rotation.toRotationMatrix();
    return transform;
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

  /**
   * @brief Compose two poses: this * other
   * Returns T_this * T_other
   */
  Pose3D operator*(const Pose3D & other) const
  {
    Eigen::Quaterniond q_result = rotation * other.rotation;
    Eigen::Vector3d t_result = rotation * other.translation + translation;
    return Pose3D(t_result, q_result);
  }

  /**
   * @brief Transform a point by this pose
   */
  Eigen::Vector3d transformPoint(const Eigen::Vector3d & point) const
  {
    return rotation * point + translation;
  }

  /**
   * @brief Linear interpolation of translation, SLERP for rotation
   * @param other The target pose
   * @param alpha Interpolation factor in [0, 1]
   */
  static Pose3D interpolate(const Pose3D & p0, const Pose3D & p1, double alpha)
  {
    Eigen::Vector3d t_interp = (1.0 - alpha) * p0.translation + alpha * p1.translation;
    Eigen::Quaterniond q_interp = p0.rotation.slerp(alpha, p1.rotation);
    return Pose3D(t_interp, q_interp);
  }
};

/**
 * @brief Timestamped pose sample for buffering
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
};

}  // namespace livox_deskew

#endif  // LIVOX_DESKEW__MATH_TYPES_HPP_
