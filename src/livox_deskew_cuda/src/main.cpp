#include <rclcpp/rclcpp.hpp>

#include "livox_deskew_cuda/deskew_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<livox_deskew_cuda::DeskewNode>();

  RCLCPP_INFO(node->get_logger(), "Starting CUDA-accelerated Livox deskew node");

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
