#include <rclcpp/rclcpp.hpp>

#include "livox_deskew/deskew_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<livox_deskew::DeskewNode>();

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
