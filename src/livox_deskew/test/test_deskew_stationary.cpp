#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

#include <rclcpp/rclcpp.hpp>

#include "livox_deskew/math_types.hpp"
#include "livox_deskew/pose_buffer.hpp"

using namespace livox_deskew;

class DeskewMathTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    buffer_ = std::make_unique<PoseBuffer>(10.0);
  }

  std::unique_ptr<PoseBuffer> buffer_;
};

// Test that stationary robot produces identity deskew
TEST_F(DeskewMathTest, StationaryRobotNoDeskew)
{
  // When T_WB(t) is constant for all t, deskewing should produce p_i_Ln == p_i_L

  // Setup: Robot is stationary at origin
  Pose3D T_WB_constant(
    Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Quaterniond::Identity()
  );

  // Add constant poses to buffer
  for (int i = 0; i < 10; ++i) {
    rclcpp::Time t(i, 0, RCL_ROS_TIME);
    buffer_->addSample(t, T_WB_constant);
  }

  // Static extrinsic: identity (livox_frame coincides with base_link)
  Pose3D T_BL = Pose3D(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
  Pose3D T_LB = T_BL.inverse();

  // Test point
  Eigen::Vector3d p_i_L(1.0, 2.0, 3.0);
  rclcpp::Time t_i(5, 0, RCL_ROS_TIME);
  rclcpp::Time t_n(5, 500000000, RCL_ROS_TIME);

  // Get poses
  Pose3D T_WB_n, T_WB_i;
  ASSERT_TRUE(buffer_->interpolatePose(t_n, T_WB_n));
  ASSERT_TRUE(buffer_->interpolatePose(t_i, T_WB_i));

  // Deskew math
  Pose3D T_WB_n_inv = T_WB_n.inverse();
  Pose3D T_Bn_Bi = T_WB_n_inv * T_WB_i;
  Eigen::Vector3d p_i_Bi = T_LB.transformPoint(p_i_L);
  Eigen::Vector3d p_i_Bn = T_Bn_Bi.transformPoint(p_i_Bi);
  Eigen::Vector3d p_i_Ln = T_BL.transformPoint(p_i_Bn);

  // For stationary robot, p_i_Ln should equal p_i_L
  EXPECT_NEAR(p_i_Ln.x(), p_i_L.x(), 1e-9);
  EXPECT_NEAR(p_i_Ln.y(), p_i_L.y(), 1e-9);
  EXPECT_NEAR(p_i_Ln.z(), p_i_L.z(), 1e-9);
}

// Test stationary robot with non-identity extrinsic
TEST_F(DeskewMathTest, StationaryRobotWithExtrinsic)
{
  // Robot is stationary
  Pose3D T_WB_constant(
    Eigen::Vector3d(5.0, 0.0, 0.0),  // Robot at x=5
    Eigen::Quaterniond::Identity()
  );

  for (int i = 0; i < 10; ++i) {
    rclcpp::Time t(i, 0, RCL_ROS_TIME);
    buffer_->addSample(t, T_WB_constant);
  }

  // Non-identity extrinsic: lidar is 1m above base_link
  Pose3D T_BL(
    Eigen::Vector3d(0.0, 0.0, 1.0),
    Eigen::Quaterniond::Identity()
  );
  Pose3D T_LB = T_BL.inverse();

  Eigen::Vector3d p_i_L(1.0, 2.0, 3.0);
  rclcpp::Time t_i(5, 0, RCL_ROS_TIME);
  rclcpp::Time t_n(5, 500000000, RCL_ROS_TIME);

  Pose3D T_WB_n, T_WB_i;
  ASSERT_TRUE(buffer_->interpolatePose(t_n, T_WB_n));
  ASSERT_TRUE(buffer_->interpolatePose(t_i, T_WB_i));

  Pose3D T_WB_n_inv = T_WB_n.inverse();
  Pose3D T_Bn_Bi = T_WB_n_inv * T_WB_i;
  Eigen::Vector3d p_i_Bi = T_LB.transformPoint(p_i_L);
  Eigen::Vector3d p_i_Bn = T_Bn_Bi.transformPoint(p_i_Bi);
  Eigen::Vector3d p_i_Ln = T_BL.transformPoint(p_i_Bn);

  // For stationary robot, p_i_Ln should equal p_i_L (regardless of extrinsic)
  EXPECT_NEAR(p_i_Ln.x(), p_i_L.x(), 1e-9);
  EXPECT_NEAR(p_i_Ln.y(), p_i_L.y(), 1e-9);
  EXPECT_NEAR(p_i_Ln.z(), p_i_L.z(), 1e-9);
}

// Test simple linear motion
TEST_F(DeskewMathTest, SimpleLinearMotion)
{
  // Robot moves 1m/s in X direction
  for (int i = 0; i < 10; ++i) {
    Pose3D pose(
      Eigen::Vector3d(i * 1.0, 0.0, 0.0),  // x = t
      Eigen::Quaterniond::Identity()
    );
    rclcpp::Time t(i, 0, RCL_ROS_TIME);
    buffer_->addSample(t, pose);
  }

  // Identity extrinsic
  Pose3D T_BL = Pose3D(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
  Pose3D T_LB = T_BL.inverse();

  // Point measured at t_i = 4s, reference time t_n = 5s
  rclcpp::Time t_i(4, 0, RCL_ROS_TIME);
  rclcpp::Time t_n(5, 0, RCL_ROS_TIME);

  // Point at (0, 0, 0) in lidar frame when measured
  Eigen::Vector3d p_i_L(0.0, 0.0, 0.0);

  Pose3D T_WB_n, T_WB_i;
  ASSERT_TRUE(buffer_->interpolatePose(t_n, T_WB_n));
  ASSERT_TRUE(buffer_->interpolatePose(t_i, T_WB_i));

  // At t_i=4s: robot at x=4
  // At t_n=5s: robot at x=5
  EXPECT_NEAR(T_WB_i.translation.x(), 4.0, 1e-9);
  EXPECT_NEAR(T_WB_n.translation.x(), 5.0, 1e-9);

  // Deskew
  Pose3D T_WB_n_inv = T_WB_n.inverse();
  Pose3D T_Bn_Bi = T_WB_n_inv * T_WB_i;
  Eigen::Vector3d p_i_Bi = T_LB.transformPoint(p_i_L);
  Eigen::Vector3d p_i_Bn = T_Bn_Bi.transformPoint(p_i_Bi);
  Eigen::Vector3d p_i_Ln = T_BL.transformPoint(p_i_Bn);

  // The point was at lidar origin (base origin with identity extrinsic)
  // Robot moved 1m forward between t_i and t_n
  // So the point appears 1m behind in the t_n frame
  EXPECT_NEAR(p_i_Ln.x(), -1.0, 1e-9);
  EXPECT_NEAR(p_i_Ln.y(), 0.0, 1e-9);
  EXPECT_NEAR(p_i_Ln.z(), 0.0, 1e-9);
}

// Test rotation motion
TEST_F(DeskewMathTest, SimpleRotationMotion)
{
  // Robot rotates 90 degrees around Z over 1 second
  for (int i = 0; i <= 10; ++i) {
    double angle = (i / 10.0) * (M_PI / 2);  // 0 to 90 degrees
    Pose3D pose(
      Eigen::Vector3d::Zero(),
      Eigen::Quaterniond(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()))
    );
    rclcpp::Time t(i, 0, RCL_ROS_TIME);
    buffer_->addSample(t, pose);
  }

  // Identity extrinsic
  Pose3D T_BL = Pose3D(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
  Pose3D T_LB = T_BL.inverse();

  // Point at (1, 0, 0) measured at t_i = 0s, reference t_n = 10s (full 90 deg rotation)
  rclcpp::Time t_i(0, 0, RCL_ROS_TIME);
  rclcpp::Time t_n(10, 0, RCL_ROS_TIME);

  Eigen::Vector3d p_i_L(1.0, 0.0, 0.0);

  Pose3D T_WB_n, T_WB_i;
  ASSERT_TRUE(buffer_->interpolatePose(t_n, T_WB_n));
  ASSERT_TRUE(buffer_->interpolatePose(t_i, T_WB_i));

  // Deskew
  Pose3D T_WB_n_inv = T_WB_n.inverse();
  Pose3D T_Bn_Bi = T_WB_n_inv * T_WB_i;
  Eigen::Vector3d p_i_Bi = T_LB.transformPoint(p_i_L);
  Eigen::Vector3d p_i_Bn = T_Bn_Bi.transformPoint(p_i_Bi);
  Eigen::Vector3d p_i_Ln = T_BL.transformPoint(p_i_Bn);

  // Point (1,0,0) measured at t_i=0 when robot faced +X
  // At t_n=10s robot has rotated 90 degrees
  // In t_n frame, the point should appear at (0, 1, 0)
  EXPECT_NEAR(p_i_Ln.x(), 0.0, 1e-6);
  EXPECT_NEAR(p_i_Ln.y(), 1.0, 1e-6);
  EXPECT_NEAR(p_i_Ln.z(), 0.0, 1e-6);
}

// Test that multiple points maintain geometric consistency
TEST_F(DeskewMathTest, GeometricConsistency)
{
  // Linear motion: 1m/s in X
  for (int i = 0; i < 10; ++i) {
    Pose3D pose(
      Eigen::Vector3d(i * 1.0, 0.0, 0.0),
      Eigen::Quaterniond::Identity()
    );
    rclcpp::Time t(i, 0, RCL_ROS_TIME);
    buffer_->addSample(t, pose);
  }

  Pose3D T_BL = Pose3D(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
  Pose3D T_LB = T_BL.inverse();

  rclcpp::Time t_n(5, 0, RCL_ROS_TIME);
  Pose3D T_WB_n;
  ASSERT_TRUE(buffer_->interpolatePose(t_n, T_WB_n));
  Pose3D T_WB_n_inv = T_WB_n.inverse();

  // Two points 1m apart in X, measured at same time
  Eigen::Vector3d p1_L(0.0, 0.0, 0.0);
  Eigen::Vector3d p2_L(1.0, 0.0, 0.0);
  rclcpp::Time t_i(3, 0, RCL_ROS_TIME);

  Pose3D T_WB_i;
  ASSERT_TRUE(buffer_->interpolatePose(t_i, T_WB_i));

  // Deskew both points
  Pose3D T_Bn_Bi = T_WB_n_inv * T_WB_i;

  Eigen::Vector3d p1_Bn = T_BL.transformPoint(T_Bn_Bi.transformPoint(T_LB.transformPoint(p1_L)));
  Eigen::Vector3d p2_Bn = T_BL.transformPoint(T_Bn_Bi.transformPoint(T_LB.transformPoint(p2_L)));

  // Distance between points should be preserved
  double original_dist = (p2_L - p1_L).norm();
  double deskewed_dist = (p2_Bn - p1_Bn).norm();
  EXPECT_NEAR(deskewed_dist, original_dist, 1e-9);
}

// Test Pose3D composition
TEST_F(DeskewMathTest, PoseComposition)
{
  // T1: translate by (1, 0, 0)
  Pose3D T1(Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Quaterniond::Identity());

  // T2: translate by (0, 1, 0)
  Pose3D T2(Eigen::Vector3d(0.0, 1.0, 0.0), Eigen::Quaterniond::Identity());

  // T1 * T2 should translate by (1, 1, 0)
  Pose3D T12 = T1 * T2;
  EXPECT_NEAR(T12.translation.x(), 1.0, 1e-9);
  EXPECT_NEAR(T12.translation.y(), 1.0, 1e-9);
  EXPECT_NEAR(T12.translation.z(), 0.0, 1e-9);
}

// Test Pose3D inverse
TEST_F(DeskewMathTest, PoseInverse)
{
  Pose3D T(
    Eigen::Vector3d(1.0, 2.0, 3.0),
    Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()))
  );

  Pose3D T_inv = T.inverse();
  Pose3D identity = T * T_inv;

  EXPECT_NEAR(identity.translation.x(), 0.0, 1e-9);
  EXPECT_NEAR(identity.translation.y(), 0.0, 1e-9);
  EXPECT_NEAR(identity.translation.z(), 0.0, 1e-9);

  // Rotation should be identity
  double angle = 2.0 * std::acos(std::abs(identity.rotation.w()));
  EXPECT_NEAR(angle, 0.0, 1e-9);
}

// Test point transformation
TEST_F(DeskewMathTest, PointTransformation)
{
  // 90 degree rotation around Z then translate by (1, 0, 0)
  Pose3D T(
    Eigen::Vector3d(1.0, 0.0, 0.0),
    Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()))
  );

  Eigen::Vector3d p(1.0, 0.0, 0.0);
  Eigen::Vector3d p_transformed = T.transformPoint(p);

  // Rotation turns (1,0,0) into (0,1,0), then translate adds (1,0,0)
  EXPECT_NEAR(p_transformed.x(), 1.0, 1e-9);
  EXPECT_NEAR(p_transformed.y(), 1.0, 1e-9);
  EXPECT_NEAR(p_transformed.z(), 0.0, 1e-9);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
