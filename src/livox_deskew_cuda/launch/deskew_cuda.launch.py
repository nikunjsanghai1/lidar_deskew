from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/livox/lidar',
        description='Input point cloud topic'
    )

    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/livox/lidar_deskew',
        description='Output deskewed point cloud topic'
    )

    odom_frame_arg = DeclareLaunchArgument(
        'odom_frame',
        default_value='odom',
        description='Odometry frame name'
    )

    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base_link',
        description='Robot base frame name'
    )

    lidar_frame_arg = DeclareLaunchArgument(
        'lidar_frame',
        default_value='livox_frame',
        description='LiDAR sensor frame name'
    )

    buffer_seconds_arg = DeclareLaunchArgument(
        'buffer_seconds',
        default_value='2.0',
        description='Pose buffer duration in seconds'
    )

    max_missing_ratio_arg = DeclareLaunchArgument(
        'max_missing_ratio',
        default_value='0.02',
        description='Maximum ratio of points that can fail before dropping scan'
    )

    use_tf_arg = DeclareLaunchArgument(
        'use_tf',
        default_value='true',
        description='Use TF for pose data'
    )

    use_odom_fallback_arg = DeclareLaunchArgument(
        'use_odom_fallback',
        default_value='true',
        description='Use odometry topic as fallback'
    )

    rosbag_mode_arg = DeclareLaunchArgument(
        'rosbag_mode',
        default_value='true',
        description='true: measure processing time (for rosbag), false: measure wall clock latency (for runtime)'
    )

    # Create the CUDA deskew node
    deskew_node = Node(
        package='livox_deskew_cuda',
        executable='livox_deskew_cuda_node',
        name='livox_deskew_cuda_node',
        output='screen',
        parameters=[{
            'input_topic': LaunchConfiguration('input_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
            'odom_frame': LaunchConfiguration('odom_frame'),
            'base_frame': LaunchConfiguration('base_frame'),
            'lidar_frame': LaunchConfiguration('lidar_frame'),
            'buffer_seconds': LaunchConfiguration('buffer_seconds'),
            'max_missing_ratio': LaunchConfiguration('max_missing_ratio'),
            'use_tf': LaunchConfiguration('use_tf'),
            'use_odom_fallback': LaunchConfiguration('use_odom_fallback'),
            'rosbag_mode': LaunchConfiguration('rosbag_mode'),
        }]
    )

    return LaunchDescription([
        input_topic_arg,
        output_topic_arg,
        odom_frame_arg,
        base_frame_arg,
        lidar_frame_arg,
        buffer_seconds_arg,
        max_missing_ratio_arg,
        use_tf_arg,
        use_odom_fallback_arg,
        rosbag_mode_arg,
        deskew_node,
    ])
