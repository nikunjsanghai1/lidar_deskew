#include <gtest/gtest.h>
#include <cmath>
#include <thread>

#include <rclcpp/rclcpp.hpp>

#include "livox_deskew/pose_buffer.hpp"
#include "livox_deskew/math_types.hpp"

using namespace livox_deskew;

class PoseBufferTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    buffer_ = std::make_unique<PoseBuffer>(10.0);  // 10 second buffer
  }

  std::unique_ptr<PoseBuffer> buffer_;
};

TEST_F(PoseBufferTest, EmptyBufferInterpolation)
{
  Pose3D result;
  rclcpp::Time query_time(1, 0, RCL_ROS_TIME);
  EXPECT_FALSE(buffer_->interpolatePose(query_time, result));
}

TEST_F(PoseBufferTest, SingleSampleInterpolation)
{
  Pose3D pose(Eigen::Vector3d(1.0, 2.0, 3.0), Eigen::Quaterniond::Identity());
  rclcpp::Time t1(1, 0, RCL_ROS_TIME);

  buffer_->addSample(t1, pose);

  Pose3D result;
  EXPECT_FALSE(buffer_->interpolatePose(t1, result));  // Need at least 2 samples
}

TEST_F(PoseBufferTest, ExactTimeMatch)
{
  Pose3D pose1(Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Quaterniond::Identity());
  Pose3D pose2(Eigen::Vector3d(10.0, 0.0, 0.0), Eigen::Quaterniond::Identity());

  rclcpp::Time t1(1, 0, RCL_ROS_TIME);
  rclcpp::Time t2(2, 0, RCL_ROS_TIME);

  buffer_->addSample(t1, pose1);
  buffer_->addSample(t2, pose2);

  Pose3D result;

  // Query at exact t1
  EXPECT_TRUE(buffer_->interpolatePose(t1, result));
  EXPECT_NEAR(result.translation.x(), 0.0, 1e-9);

  // Query at exact t2
  EXPECT_TRUE(buffer_->interpolatePose(t2, result));
  EXPECT_NEAR(result.translation.x(), 10.0, 1e-9);
}

TEST_F(PoseBufferTest, LinearInterpolationMidpoint)
{
  Pose3D pose1(Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Quaterniond::Identity());
  Pose3D pose2(Eigen::Vector3d(10.0, 20.0, 30.0), Eigen::Quaterniond::Identity());

  rclcpp::Time t1(1, 0, RCL_ROS_TIME);
  rclcpp::Time t2(2, 0, RCL_ROS_TIME);
  rclcpp::Time t_mid(1, 500000000, RCL_ROS_TIME);  // 1.5 seconds

  buffer_->addSample(t1, pose1);
  buffer_->addSample(t2, pose2);

  Pose3D result;
  EXPECT_TRUE(buffer_->interpolatePose(t_mid, result));

  // At midpoint, should be average of the two poses
  EXPECT_NEAR(result.translation.x(), 5.0, 1e-9);
  EXPECT_NEAR(result.translation.y(), 10.0, 1e-9);
  EXPECT_NEAR(result.translation.z(), 15.0, 1e-9);
}

TEST_F(PoseBufferTest, LinearInterpolationQuarter)
{
  Pose3D pose1(Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Quaterniond::Identity());
  Pose3D pose2(Eigen::Vector3d(100.0, 0.0, 0.0), Eigen::Quaterniond::Identity());

  rclcpp::Time t1(1, 0, RCL_ROS_TIME);
  rclcpp::Time t2(2, 0, RCL_ROS_TIME);
  rclcpp::Time t_quarter(1, 250000000, RCL_ROS_TIME);  // 1.25 seconds

  buffer_->addSample(t1, pose1);
  buffer_->addSample(t2, pose2);

  Pose3D result;
  EXPECT_TRUE(buffer_->interpolatePose(t_quarter, result));

  // At 0.25 of the way, should be 25% of the distance
  EXPECT_NEAR(result.translation.x(), 25.0, 1e-9);
}

TEST_F(PoseBufferTest, SlerpRotationMidpoint)
{
  // Pose 1: Identity rotation
  Eigen::Quaterniond q1 = Eigen::Quaterniond::Identity();

  // Pose 2: 90 degree rotation around Z axis
  Eigen::Quaterniond q2(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));

  Pose3D pose1(Eigen::Vector3d::Zero(), q1);
  Pose3D pose2(Eigen::Vector3d::Zero(), q2);

  rclcpp::Time t1(1, 0, RCL_ROS_TIME);
  rclcpp::Time t2(2, 0, RCL_ROS_TIME);
  rclcpp::Time t_mid(1, 500000000, RCL_ROS_TIME);

  buffer_->addSample(t1, pose1);
  buffer_->addSample(t2, pose2);

  Pose3D result;
  EXPECT_TRUE(buffer_->interpolatePose(t_mid, result));

  // At midpoint, should be 45 degree rotation around Z
  Eigen::Quaterniond expected(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));

  // Compare quaternions (handle sign ambiguity)
  double dot = std::abs(result.rotation.dot(expected));
  EXPECT_NEAR(dot, 1.0, 1e-6);
}

TEST_F(PoseBufferTest, OutOfRangeQuery)
{
  Pose3D pose1(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
  Pose3D pose2(Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Quaterniond::Identity());

  rclcpp::Time t1(2, 0, RCL_ROS_TIME);
  rclcpp::Time t2(3, 0, RCL_ROS_TIME);

  buffer_->addSample(t1, pose1);
  buffer_->addSample(t2, pose2);

  Pose3D result;

  // Query before range
  rclcpp::Time t_before(1, 0, RCL_ROS_TIME);
  EXPECT_FALSE(buffer_->interpolatePose(t_before, result));

  // Query after range
  rclcpp::Time t_after(4, 0, RCL_ROS_TIME);
  EXPECT_FALSE(buffer_->interpolatePose(t_after, result));
}

TEST_F(PoseBufferTest, PruningOldSamples)
{
  // Use a short buffer duration
  buffer_->setBufferDuration(1.0);

  Pose3D pose(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());

  // Add samples at 1s intervals
  for (int i = 0; i < 5; ++i) {
    rclcpp::Time t(i, 0, RCL_ROS_TIME);
    buffer_->addSample(t, pose);
  }

  // After adding sample at t=4s with 1s buffer, samples older than 3s should be pruned
  // Should have samples at t=3 and t=4 at minimum
  EXPECT_GE(buffer_->size(), 2u);
  EXPECT_LE(buffer_->size(), 3u);  // At most samples within the buffer duration

  // Old time should fail
  Pose3D result;
  rclcpp::Time t_old(0, 0, RCL_ROS_TIME);
  EXPECT_FALSE(buffer_->interpolatePose(t_old, result));

  // Recent time should work
  rclcpp::Time t_recent(3, 500000000, RCL_ROS_TIME);
  EXPECT_TRUE(buffer_->interpolatePose(t_recent, result));
}

TEST_F(PoseBufferTest, MultiSampleInterpolation)
{
  // Add 10 samples
  for (int i = 0; i < 10; ++i) {
    Pose3D pose(Eigen::Vector3d(i * 1.0, 0.0, 0.0), Eigen::Quaterniond::Identity());
    rclcpp::Time t(i, 0, RCL_ROS_TIME);
    buffer_->addSample(t, pose);
  }

  // Query at various times
  Pose3D result;

  rclcpp::Time t_query(4, 500000000, RCL_ROS_TIME);  // 4.5s
  EXPECT_TRUE(buffer_->interpolatePose(t_query, result));
  EXPECT_NEAR(result.translation.x(), 4.5, 1e-9);

  t_query = rclcpp::Time(7, 250000000, RCL_ROS_TIME);  // 7.25s
  EXPECT_TRUE(buffer_->interpolatePose(t_query, result));
  EXPECT_NEAR(result.translation.x(), 7.25, 1e-9);
}

TEST_F(PoseBufferTest, GetTimeRange)
{
  // Empty buffer
  auto range = buffer_->getTimeRange();
  EXPECT_FALSE(range.has_value());

  // Add one sample
  Pose3D pose(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
  buffer_->addSample(rclcpp::Time(1, 0, RCL_ROS_TIME), pose);

  range = buffer_->getTimeRange();
  EXPECT_FALSE(range.has_value());  // Need at least 2 samples

  // Add second sample
  buffer_->addSample(rclcpp::Time(5, 0, RCL_ROS_TIME), pose);

  range = buffer_->getTimeRange();
  EXPECT_TRUE(range.has_value());
  EXPECT_EQ(range->first.seconds(), 1.0);
  EXPECT_EQ(range->second.seconds(), 5.0);
}

TEST_F(PoseBufferTest, ThreadSafety)
{
  // Basic thread safety test - add samples from multiple threads
  std::vector<std::thread> threads;

  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([this, t]() {
        for (int i = 0; i < 100; ++i) {
          Pose3D pose(Eigen::Vector3d(t * 100 + i, 0, 0), Eigen::Quaterniond::Identity());
          rclcpp::Time time(t * 100 + i, 0, RCL_ROS_TIME);
          buffer_->addSample(time, pose);
        }
      });
  }

  for (auto & thread : threads) {
    thread.join();
  }

  // Buffer should have samples and be queryable
  EXPECT_GT(buffer_->size(), 0u);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
