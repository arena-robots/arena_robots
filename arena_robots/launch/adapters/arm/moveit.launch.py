"""MoveIt arm adapter launch."""

import json

from arena_robots.assembly import RequestPart
from arena_robots.moveit_factory import build_moveit_params
from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import Node, PushRosNamespace

from arena_rclpy_mixins.launch_argument import LaunchArgument


def generate_launch_description():
    ld_items = []
    LaunchArgument.auto_append(ld_items)

    robot = LaunchArgument("robot", description="Robot model name.")
    namespace = LaunchArgument("namespace", description="ROS namespace of the robot.")
    frame = LaunchArgument("frame", description="Frame prefix of the robot.")
    use_sim_time = LaunchArgument("use_sim_time", description="Whether nodes follow the simulator clock.")
    parts_json = LaunchArgument("parts_json", default_value="{}", description="JSON object mapping part type to the list of robot parts to plan for.")
    instance = LaunchArgument("instance", default_value="", description="Name of the arm instance the MoveIt stack serves (empty = the only arm).")

    def launch_setup(context, *args, **kwargs):
        ns = namespace.substitution.perform(context)
        robot_name = robot.substitution.perform(context)
        sim = use_sim_time.substitution.perform(context).lower() == "true"
        parts = {
            t: [RequestPart(variant=i["variant"], mount=i.get("mount")) for i in items]
            for t, items in json.loads(parts_json.substitution.perform(context)).items()
        }
        instance_name = instance.substitution.perform(context) or None

        tf_prefix = frame.substitution.perform(context)

        moveit_params = build_moveit_params(robot_name, tf_prefix=tf_prefix, parts=parts, instance=instance_name)
        if moveit_params is None:
            raise ValueError(f"{robot_name}: missing 'arm' cap or caps.moveit block; cannot launch moveit adapter")

        servo_config = {
            "move_group_name": "ur_manipulator",
            "planning_frame": f"{tf_prefix}arm_base_link",
            "ee_frame_name": f"{tf_prefix}arm_tool0",
            "command_in_type": "speed_units",
            "publish_period": 0.01,
             "scale": {
                "linear": 0.08,
                "rotational": 0.1,
                "joint": 0.1,
            },
            "command_out_type": "trajectory_msgs/JointTrajectory",
            "command_out_topic": "arm_controller/joint_trajectory",
            "joint_topic": "joint_states",
            "is_primary_planning_scene_monitor": False,
            "check_collisions": True,
            "lower_singularity_threshold": 40.0,
            "hard_stop_singularity_threshold": 80.0,
            "leaving_singularity_threshold_multiplier": 2.0,
        }

        move_group = Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                moveit_params,
                {"use_sim_time": sim},
            ],
        )

        servo_node = Node(
            package="moveit_servo",
            executable="servo_node",
            output="screen",
            parameters=[
                moveit_params,
                {"moveit_servo": servo_config},
                {"use_sim_time": sim},
            ],
        )

        return [PushRosNamespace(ns), move_group, servo_node]

    return LaunchDescription([*ld_items, OpaqueFunction(function=launch_setup)])
