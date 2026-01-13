#include <gtest/gtest.h>
#include <cmath>

#include <rclcpp/rclcpp.hpp>

#include "livox_deskew_cuda/pose_buffer.hpp"
#include "cuda_deskew.h"
#include "cuda_types.h"

using namespace livox_deskew_cuda;

class CudaDeskewTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Initialize CUDA
    CudaDeskewResult result = cuda_deskew_init();
    ASSERT_EQ(result, CUDA_DESKEW_SUCCESS) << "CUDA init failed: " << cuda_deskew_get_last_error();
    cuda_initialized_ = true;
  }

  void TearDown() override
  {
    if (cuda_initialized_) {
      cuda_deskew_cleanup();
    }
  }

  bool cuda_initialized_ = false;
};

TEST_F(CudaDeskewTest, Initialization)
{
  EXPECT_TRUE(cuda_deskew_is_initialized());
}

TEST_F(CudaDeskewTest, UploadPoseBuffer)
{
  // Create simple pose buffer
  std::vector<PoseSample_GPU> samples;

  for (int i = 0; i < 10; ++i) {
    PoseSample_GPU sample;
    sample.timestamp = 1.0 + i * 0.01;  // 100 Hz
    sample.pose.tx = i * 0.1f;
    sample.pose.ty = 0.0f;
    sample.pose.tz = 0.0f;
    sample.pose.qw = 1.0f;
    sample.pose.qx = 0.0f;
    sample.pose.qy = 0.0f;
    sample.pose.qz = 0.0f;
    samples.push_back(sample);
  }

  CudaDeskewResult result = cuda_upload_pose_buffer(samples.data(), samples.size());
  EXPECT_EQ(result, CUDA_DESKEW_SUCCESS);
}

TEST_F(CudaDeskewTest, UploadStaticTransforms)
{
  // Identity transforms
  Pose3D_GPU T_BL, T_LB;

  T_BL.tx = 0.0f; T_BL.ty = 0.0f; T_BL.tz = 0.0f;
  T_BL.qw = 1.0f; T_BL.qx = 0.0f; T_BL.qy = 0.0f; T_BL.qz = 0.0f;

  T_LB = T_BL;

  CudaDeskewResult result = cuda_upload_static_transforms(&T_BL, &T_LB);
  EXPECT_EQ(result, CUDA_DESKEW_SUCCESS);
}

TEST_F(CudaDeskewTest, PoseBufferExport)
{
  PoseBuffer buffer(2.0);

  // Add samples
  for (int i = 0; i < 10; ++i) {
    rclcpp::Time t(1, i * 10000000, RCL_ROS_TIME);  // 10ms apart
    Pose3D pose(
      Eigen::Vector3d(i * 0.1, 0, 0),
      Eigen::Quaterniond::Identity()
    );
    buffer.addSample(t, pose);
  }

  // Export to GPU format
  std::vector<PoseSample_GPU> gpu_samples = buffer.exportToGPU();

  EXPECT_EQ(gpu_samples.size(), 10u);
  EXPECT_NEAR(gpu_samples[5].pose.tx, 0.5f, 1e-5f);
}

TEST_F(CudaDeskewTest, Pose3DToGPU)
{
  Pose3D pose(
    Eigen::Vector3d(1.0, 2.0, 3.0),
    Eigen::Quaterniond(0.707, 0.0, 0.707, 0.0)  // 90 deg around Y
  );

  Pose3D_GPU gpu_pose = pose.toGPU();

  EXPECT_NEAR(gpu_pose.tx, 1.0f, 1e-5f);
  EXPECT_NEAR(gpu_pose.ty, 2.0f, 1e-5f);
  EXPECT_NEAR(gpu_pose.tz, 3.0f, 1e-5f);
  EXPECT_NEAR(gpu_pose.qw, 0.707f, 1e-3f);
  EXPECT_NEAR(gpu_pose.qy, 0.707f, 1e-3f);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
