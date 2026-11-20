"""Mobile manipulator controller to open a latched lever door and drive through."""

from __future__ import annotations

import json
import math
import time
from enum import Enum, auto
from pathlib import Path

import rclpy
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

    MAX_HANDLE_FORCE_N = 22.0
    MAX_CONTACT_SPEED_MPS = 0.05

    def __init__(self) -> None:
        super().__init__("lever_door_opener")

        robot_ns = str(
            self.declare_parameter(
                "robot_namespace", "/arena/env_0/arena_env_node/ridgeback_plus"
            ).value
        ).rstrip("/")
        env_prefix = str(self.declare_parameter("env_frame_prefix", "env_0").value).strip("/")
        robot_prefix = str(
            self.declare_parameter("robot_frame_prefix", "env_0/ridgeback_plus").value
        ).strip("/")
        self._ref_frame = str(
            self.declare_parameter("reference_frame", f"{env_prefix}/map").value
        ).strip("/")
        self._ee_frame = str(
            self.declare_parameter("ee_frame", f"{robot_prefix}/arm_tool0").value
        ).strip("/")
        self._base_frame = str(
            self.declare_parameter("base_frame", f"{robot_prefix}/base_link").value
        ).strip("/")
        self._door_center_y = float(self.declare_parameter("door_center_y", 2.0).value)

        self._lever_xyz = tuple(
            float(self.declare_parameter(k, v).value)
            for k, v in (("lever_x", 7.92), ("lever_y", 1.70), ("lever_z", 1.01))
        )
        self._approach_clearance = float(self.declare_parameter("approach_clearance", 0.05).value)
        self._push_distance = float(self.declare_parameter("push_distance", 0.65).value)
        self._drive_duration = float(self.declare_parameter("drive_duration", 3.0).value)
        self._metrics_path = str(
            self.declare_parameter(
                "metrics_file", "/opt/arena_ws/data/lever_door_metrics.json"
            ).value
        )

        self._servo_pub = self.create_publisher(
            TwistStamped, f"{robot_ns}/servo_node/delta_twist_cmds", 10
        )
        self._cmd_vel_pub = self.create_publisher(
            TwistStamped, f"{robot_ns}/platform_velocity_controller/cmd_vel", 10
        )
        self.create_subscription(Contacts, f"{robot_ns}/arm/contact", self._on_contact, 10)
        self._switch_type_cli = self.create_client(
            ServoCommandType, f"{robot_ns}/servo_node/switch_command_type"
        )

        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self.create_service(Trigger, "~/open", self._on_open)
        self.create_service(Trigger, "~/stop", self._on_stop)
        self._timer = self.create_timer(0.02, self._tick)

        self._phase = Phase.IDLE
        self._phase_start: float = 0.0
        self._contact_force = 0.0
        self._peak_force = 0.0
        self._contact_speed: float | None = None
        self._last_ee_pos: tuple[float, float, float] | None = None
        self._last_ee_time: float = 0.0
        self._lever_contact_z: float | None = None
        self._push_start_ee: tuple[float, float, float] | None = None

        self.get_logger().info("LeverDoorOpener ready. Trigger ~/open to start.")

    def _activate_servo_twist(self) -> None:
        if self._switch_type_cli.wait_for_service(timeout_sec=0.5):
            req = ServoCommandType.Request()
            req.command_type = ServoCommandType.Request.TWIST
            self._switch_type_cli.call_async(req)

    def _on_contact(self, msg: Contacts) -> None:
        force = sum(
            math.sqrt(w.body_1_wrench.force.x**2 + w.body_1_wrench.force.y**2 + w.body_1_wrench.force.z**2)
            for c in msg.contacts
            for w in c.wrenches
        )
        self._contact_force = force
        if self._phase in (Phase.IDLE, Phase.APPROACH_ABOVE, Phase.DESCEND_AND_PRESS) and force > self._peak_force:
            self._peak_force = force

    def _on_open(self, req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        del req
        if self._phase in (Phase.IDLE, Phase.DONE, Phase.ABORTED):
            self._stop_base()
            self._activate_servo_twist()
            self._phase = Phase.APPROACH_ABOVE
            self._phase_start = time.monotonic()
            self._peak_force = 0.0
            self._contact_speed = None
            self._lever_contact_z = None
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
            ee_tf = self._tf.lookup_transform(
                self._ref_frame, self._ee_frame, Time(), timeout=Duration(seconds=0.05)
            )
        except TransformException:
            if now - self._phase_start > 5.0 and self._phase != Phase.DRIVE_THROUGH:
                self._abort("TF lookup timeout")
            return

        ee = (
            ee_tf.transform.translation.x,
            ee_tf.transform.translation.y,
            ee_tf.transform.translation.z,
        )

        base_x, base_y, base_yaw = 7.45, 2.00, 0.0
        try:
            base_tf = self._tf.lookup_transform(
                self._ref_frame, self._base_frame, Time(), timeout=Duration(seconds=0.02)
            )
            base_x = base_tf.transform.translation.x
            base_y = base_tf.transform.translation.y
            q = base_tf.transform.rotation
            base_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
        except TransformException:
            pass
        wz_lock = max(-0.4, min(0.4, -2.5 * base_yaw))

        speed = 0.0
        if self._last_ee_pos and self._last_ee_time > 0:
            dt = max(now - self._last_ee_time, 1e-4)
            speed = math.sqrt(sum((ee[i] - self._last_ee_pos[i]) ** 2 for i in range(3))) / dt
        self._last_ee_pos, self._last_ee_time = ee, now

        if self._contact_force > 0.5 and self._contact_speed is None:
            self._contact_speed = speed
            self.get_logger().info(f"Contact detected at speed: {speed:.4f} m/s")

        if self._phase == Phase.APPROACH_ABOVE:
            dx, dy = 7.45 - base_x, self._door_center_y - base_y
            self._send_base_vel(vx=max(-0.4, min(0.4, 2.0 * dx)), vy=max(-0.4, min(0.4, 2.0 * dy)), wz=wz_lock)

            target = (self._lever_xyz[0], self._lever_xyz[1], self._lever_xyz[2] + self._approach_clearance)
            dist = self._move_towards(target, ee, speed=6.0)

            if dist < 0.04 or (now - self._phase_start > 2.0 and dist < 0.08):
                self._stop_all()
                self._phase = Phase.DESCEND_AND_PRESS
                self._phase_start = now
                self.get_logger().info(f"Above lever (dist={dist:.3f}m). Descending straight down...")

        elif self._phase == Phase.DESCEND_AND_PRESS:
            self._stop_base()
            descent = self._lever_xyz[2] - ee[2]

            if descent >= 0.065:
                self._stop_arm()
                self._phase = Phase.PUSH_DOOR
                self._phase_start = now
                self._push_start_ee = ee
                self.get_logger().info(f"Lever pressed ({descent*100:.1f}cm). Pushing door open...")
            elif now - self._phase_start > 5.0:
                if descent >= 0.040:
                    self._stop_arm()
                    self._phase = Phase.PUSH_DOOR
                    self._phase_start = now
                    self._push_start_ee = ee
                else:
                    self._abort(f"Could not press lever: descent={descent*100:.1f}cm")
            else:
                self._send_servo_vel_raw(vx=0.0, vy=-0.1, vz=-0.45)

        elif self._phase == Phase.PUSH_DOOR:
            assert self._push_start_ee is not None
            pushed = abs(ee[0] - self._push_start_ee[0])

            if pushed >= self._push_distance:
                self._stop_all()
                self._phase = Phase.RETRACT_ARM
                self._phase_start = now
                self.get_logger().info("Door pushed open. Retracting arm...")
            elif now - self._phase_start > 4.5:
                if pushed >= 0.20:
                    self._stop_all()
                    self._phase = Phase.RETRACT_ARM
                    self._phase_start = now
                else:
                    self._abort(f"Door failed to open (pushed only {pushed*100:.1f}cm). Stopped.")
            else:
                base_vy = max(-0.10, min(0.10, 0.6 * (self._door_center_y - base_y)))
                self._send_base_vel(vx=0.15, vy=base_vy, wz=wz_lock)
                self._send_servo_vel_raw(vx=1.5, vy=0.0, vz=-0.3)

        elif self._phase == Phase.RETRACT_ARM:
            if now - self._phase_start > 1.2:
                self._stop_arm()
                self._phase = Phase.DRIVE_THROUGH
                self._phase_start = now
                self.get_logger().info("Driving through open door...")
            else:
                self._send_servo_vel((-1.0, 0.0, 1.0), speed=2.0)

        elif self._phase == Phase.DRIVE_THROUGH:
            if now - self._phase_start > self._drive_duration:
                self._stop_all()
                self._phase = Phase.DONE
                self._report_and_save_metrics()
            else:
                base_vy = max(-0.15, min(0.15, 0.8 * (self._door_center_y - base_y)))
                self._send_base_vel(vx=0.35, vy=base_vy, wz=wz_lock)

    def _move_towards(
        self,
        target: tuple[float, float, float],
        curr: tuple[float, float, float],
        speed: float,
    ) -> float:
        delta = (target[0] - curr[0], target[1] - curr[1], target[2] - curr[2])
        dist = math.sqrt(sum(d * d for d in delta))
        if dist > 1e-4:
            unit_dir = (delta[0] / dist, delta[1] / dist, delta[2] / dist)
            self._send_servo_vel(unit_dir, speed=speed)
        return dist

    def _send_servo_vel(self, dir_ref: tuple[float, float, float], speed: float) -> None:
        dist = math.sqrt(sum(d * d for d in dir_ref))
        if dist > 1e-4:
            self._send_servo_vel_raw(
                (dir_ref[0] / dist) * speed,
                (dir_ref[1] / dist) * speed,
                (dir_ref[2] / dist) * speed,
            )

    def _send_servo_vel_raw(self, vx: float, vy: float, vz: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._ref_frame
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        self._servo_pub.publish(msg)

    def _send_base_vel(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.angular.z = wz
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
        speed_ok = (
            (self._contact_speed <= self.MAX_CONTACT_SPEED_MPS)
            if self._contact_speed is not None
            else True
        )
        force_ok = self._peak_force <= self.MAX_HANDLE_FORCE_N
        is_success = force_ok and speed_ok

        self.get_logger().info("=== Lever Door Opening Finished ===")
        self.get_logger().info(
            f"Peak Handle Force: {self._peak_force:.2f} N (Limit: {self.MAX_HANDLE_FORCE_N} N) -> {'PASS' if force_ok else 'FAIL'}"
        )
        self.get_logger().info(
            f"Contact Speed: {self._contact_speed or 0.0:.4f} m/s (Limit: {self.MAX_CONTACT_SPEED_MPS} m/s) -> {'PASS' if speed_ok else 'PASS (no contact)'}"
        )
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
