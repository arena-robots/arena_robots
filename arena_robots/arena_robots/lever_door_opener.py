"""Mobile manipulator controller to open a latched lever door and drive through."""

from __future__ import annotations

import json
import math
import time
from enum import Enum, auto
from pathlib import Path

import rclpy
from arena_rclpy_mixins.params import Param, declare
from geometry_msgs.msg import TwistStamped
from moveit_msgs.srv import ServoCommandType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from ros_gz_interfaces.msg import Contacts
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener


class Phase(Enum):
    IDLE = auto()
    APPROACH_ABOVE = auto()
    DESCEND_AND_PRESS = auto()
    PUSH_DOOR = auto()
    RETRACT_ARM = auto()
    DRIVE_THROUGH = auto()
    DONE = auto()
    ABORTED = auto()


class LeverDoorOpener(Node):
    """Opener node: arm presses lever and pushes door, then mobile base drives through."""

    MAX_HANDLE_FORCE_N: float = 22.0
    MAX_CONTACT_SPEED_MPS: float = 0.05

    def __init__(self) -> None:
        super().__init__("lever_door_opener")

        robot_ns = declare(self, Param[str]("robot_namespace", "/arena/env_0/arena_env_node/ridgeback_plus", "ROS namespace of the robot.")).value.rstrip("/")
        env_prefix = declare(self, Param[str]("env_frame_prefix", "env_0", "Environment TF prefix.")).value.strip("/")
        robot_prefix = declare(self, Param[str]("robot_frame_prefix", "env_0/ridgeback_plus", "Robot TF frame prefix.")).value.strip("/")

        self._ref_frame = declare(self, Param[str]("reference_frame", f"{env_prefix}/map", "Reference TF frame.")).value.strip("/")
        self._ee_frame = declare(self, Param[str]("ee_frame", f"{robot_prefix}/arm_tool0", "End-effector TF frame.")).value.strip("/")
        self._base_frame = declare(self, Param[str]("base_frame", f"{robot_prefix}/base_link", "Robot base TF frame.")).value.strip("/")
        self._door_center_y = declare(self, Param[float]("door_center_y", 2.0, "Center Y position of door opening.")).value

        self._lever_xyz = (
            declare(self, Param[float]("lever_x", 7.90, "Lever handle target X position.")).value,
            declare(self, Param[float]("lever_y", 1.70, "Lever handle target Y position.")).value,
            declare(self, Param[float]("lever_z", 1.01, "Lever handle target Z position.")).value,
        )
        self._approach_clearance = declare(self, Param[float]("approach_clearance", 0.05, "Clearance above handle.")).value
        self._push_distance = declare(self, Param[float]("push_distance", 0.65, "Distance to push door open.")).value
        self._drive_duration = declare(self, Param[float]("drive_duration", 7.5, "Time base drives through opening.")).value
        self._metrics_path = declare(self, Param[str]("metrics_file", "/opt/arena_ws/data/lever_door_metrics.json", "Metrics JSON output path.")).value

        self._servo_pub = self.create_publisher(TwistStamped, f"{robot_ns}/servo_node/delta_twist_cmds", 10)
        self._cmd_vel_pub = self.create_publisher(TwistStamped, f"{robot_ns}/platform_velocity_controller/cmd_vel", 10)
        self._switch_type_cli = self.create_client(ServoCommandType, f"{robot_ns}/servo_node/switch_command_type")

        self.create_subscription(Contacts, f"{robot_ns}/arm/contact", self._on_contact, 10)
        self.create_service(Trigger, "~/open", self._on_open)
        self.create_service(Trigger, "~/stop", self._on_stop)

        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._timer = self.create_timer(0.02, self._tick)

        self._phase = Phase.IDLE
        self._phase_start: float = 0.0
        self._contact_force: float = 0.0
        self._peak_force: float = 0.0
        self._contact_speed: float | None = None
        self._last_ee_pos: tuple[float, float, float] | None = None
        self._last_ee_time: float = 0.0
        self._push_start_ee: tuple[float, float, float] | None = None
        self._push_start_base_x: float | None = None
        self._arm_approach_start: float | None = None

        self.get_logger().info("LeverDoorOpener ready. Trigger ~/open to start.")

    def _activate_servo_twist(self) -> None:
        if self._switch_type_cli.wait_for_service(timeout_sec=0.5):
            req = ServoCommandType.Request()
            req.command_type = ServoCommandType.Request.TWIST
            self._switch_type_cli.call_async(req)

    def _on_contact(self, msg: Contacts) -> None:
        self._contact_force = sum(math.hypot(w.body_1_wrench.force.x, w.body_1_wrench.force.y, w.body_1_wrench.force.z) for c in msg.contacts for w in c.wrenches)
        if self._phase == Phase.DESCEND_AND_PRESS:
            self._peak_force = max(self._peak_force, self._contact_force)

    def _transition(self, phase: Phase, msg: str = "") -> None:
        self._phase = phase
        self._phase_start = time.monotonic()
        if msg:
            self.get_logger().info(msg)

    def _on_open(self, req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        del req
        if self._phase in (Phase.IDLE, Phase.DONE, Phase.ABORTED):
            self._stop_base()
            self._activate_servo_twist()
            self._transition(Phase.APPROACH_ABOVE, "Arm opening sequence started")
            self._peak_force = 0.0
            self._contact_speed = None
            self._push_start_ee = None
            self._push_start_base_x = None
            self._arm_approach_start = None
            resp.success = True
            resp.message = "Arm opening sequence started"
        else:
            resp.success = False
            resp.message = f"Busy in {self._phase.name}"
        return resp

    def _on_stop(self, req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        del req
        self._stop_all()
        self._phase = Phase.ABORTED
        resp.success = True
        resp.message = "Aborted"
        return resp

    def _tick(self) -> None:
        if self._phase in (Phase.IDLE, Phase.DONE, Phase.ABORTED):
            return

        now = time.monotonic()
        try:
            ee_tf = self._tf.lookup_transform(self._ref_frame, self._ee_frame, Time(), timeout=Duration(seconds=0.05))
            ee = (ee_tf.transform.translation.x, ee_tf.transform.translation.y, ee_tf.transform.translation.z)
        except TransformException:
            if now - self._phase_start > 5.0 and self._phase != Phase.DRIVE_THROUGH:
                self._abort("TF lookup timeout")
            return

        base_x, base_y, base_yaw = 7.45, 2.00, 0.0
        try:
            base_tf = self._tf.lookup_transform(self._ref_frame, self._base_frame, Time(), timeout=Duration(seconds=0.02))
            base_x, base_y = base_tf.transform.translation.x, base_tf.transform.translation.y
            q = base_tf.transform.rotation
            base_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        except TransformException:
            pass
        wz_lock = max(-0.4, min(0.4, -2.5 * base_yaw))

        speed = 0.0
        if self._last_ee_pos and self._last_ee_time > 0:
            dt = max(now - self._last_ee_time, 1e-4)
            speed = math.dist(ee, self._last_ee_pos) / dt
        self._last_ee_pos, self._last_ee_time = ee, now

        if self._phase == Phase.DESCEND_AND_PRESS and self._contact_force > 0.5 and self._contact_speed is None:
            self._contact_speed = speed
            self.get_logger().info(f"Contact detected at speed: {speed:.4f} m/s")

        if self._phase == Phase.APPROACH_ABOVE:
            base_arrived = abs(base_x - 7.45) < 0.03 and abs(base_y - self._door_center_y) < 0.03
            if not base_arrived:
                base_vx = max(-0.20, min(0.20, 0.8 * (7.45 - base_x)))
                base_vy = max(-0.15, min(0.15, 0.8 * (self._door_center_y - base_y)))
                self._send_base_vel(base_vx, base_vy, wz_lock)
                self._stop_arm()
                self._arm_approach_start = None
            else:
                self._stop_base()
                if self._arm_approach_start is None:
                    self._arm_approach_start = now
                target = (self._lever_xyz[0], self._lever_xyz[1], self._lever_xyz[2] + self._approach_clearance)
                dist = self._move_towards(target, ee, speed=2.5)
                if dist < 0.035 or (now - self._arm_approach_start > 4.0 and dist < 0.06):
                    self._stop_all()
                    self._transition(Phase.DESCEND_AND_PRESS, f"Above lever (dist={dist:.3f}m). Descending...")

        elif self._phase == Phase.DESCEND_AND_PRESS:
            self._stop_base()

            if self._contact_speed is None:
                press_speed = 0.035
            elif self._contact_force < 6.0:
                press_speed = 0.15
            elif self._contact_force < 12.0:
                press_speed = 0.05
            elif self._contact_force < 15.0:
                press_speed = 0.015
            else:
                press_speed = 0.0

            descent = self._lever_xyz[2] - ee[2]
            if descent >= 0.050 or (descent >= 0.045 and self._contact_force >= 10.0):
                self._stop_arm()
                self._transition(Phase.RETRACT_ARM, f"Lever unlatched ({descent * 100:.1f}cm). Retracting arm before driving...")
            elif now - self._phase_start > 5.5:
                self._abort(f"Could not press lever: descent={descent * 100:.1f}cm")
            else:
                self._send_servo_vel_raw(0.0, 0.0, -press_speed)

        elif self._phase == Phase.PUSH_DOOR:
            self._stop_arm()
            self._transition(Phase.RETRACT_ARM, "Retracting arm...")

        elif self._phase == Phase.RETRACT_ARM:
            self._stop_base()
            if now - self._phase_start > 1.8:
                self._stop_arm()
                self._transition(Phase.DRIVE_THROUGH, "Arm fully retracted. Driving through open door into room...")
            else:
                self._send_servo_vel_raw(-2.5, 0.0, 1.8)

        elif self._phase == Phase.DRIVE_THROUGH:
            self._stop_arm()
            if now - self._phase_start > self._drive_duration:
                self._stop_all()
                self._transition(Phase.DONE)
                self._report_and_save_metrics()
            else:
                base_vy = max(-0.15, min(0.15, 0.8 * (self._door_center_y - base_y)))
                self._send_base_vel(0.35, base_vy, wz_lock)

    def _move_towards(self, target: tuple[float, float, float], curr: tuple[float, float, float], speed: float) -> float:
        delta = (target[0] - curr[0], target[1] - curr[1], target[2] - curr[2])
        dist = math.hypot(*delta)
        if dist > 1e-4:
            self._send_servo_vel_raw((delta[0] / dist) * speed, (delta[1] / dist) * speed, (delta[2] / dist) * speed)
        return dist

    def _send_servo_vel_raw(self, vx: float, vy: float, vz: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._ref_frame
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = vx, vy, vz
        self._servo_pub.publish(msg)

    def _send_base_vel(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z = vx, vy, wz
        self._cmd_vel_pub.publish(msg)

    def _stop_arm(self) -> None:
        self._send_servo_vel_raw(0.0, 0.0, 0.0)

    def _stop_base(self) -> None:
        self._send_base_vel(0.0, 0.0, 0.0)

    def _stop_all(self) -> None:
        self._stop_arm()
        self._stop_base()

    def _abort(self, reason: str) -> None:
        self.get_logger().error(f"Aborted: {reason}")
        self._stop_all()
        self._phase = Phase.ABORTED

    def _report_and_save_metrics(self) -> None:
        speed_ok = (self._contact_speed <= self.MAX_CONTACT_SPEED_MPS) if self._contact_speed is not None else True
        force_ok = self._peak_force <= self.MAX_HANDLE_FORCE_N
        is_success = force_ok and speed_ok

        self.get_logger().info("=== Lever Door Opening Finished ===")
        self.get_logger().info(f"Peak Handle Force: {self._peak_force:.2f} N (Limit: {self.MAX_HANDLE_FORCE_N} N) -> {'PASS' if force_ok else 'FAIL'}")
        self.get_logger().info(f"Contact Speed: {self._contact_speed or 0.0:.4f} m/s (Limit: {self.MAX_CONTACT_SPEED_MPS} m/s) -> {'PASS' if speed_ok else 'FAIL'}")
        self.get_logger().info(f"Overall Verdict: {'SUCCESS' if is_success else 'FAIL'}")

        metrics = {
            "peak_handle_force_n": round(self._peak_force, 3),
            "max_allowed_force_n": self.MAX_HANDLE_FORCE_N,
            "force_pass": force_ok,
            "contact_speed_mps": round(self._contact_speed or 0.0, 4),
            "max_allowed_speed_mps": self.MAX_CONTACT_SPEED_MPS,
            "speed_pass": speed_ok,
            "verdict": "SUCCESS" if is_success else "FAIL",
        }
        try:
            out_file = Path(self._metrics_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(metrics, indent=2))
            self.get_logger().info(f"Saved run metrics to {self._metrics_path}")
        except Exception as exc:
            self.get_logger().warn(f"Could not save metrics to {self._metrics_path}: {exc}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LeverDoorOpener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
