import json
from pathlib import Path

from geometry_msgs.msg import Vector3, Wrench
from ros_gz_interfaces.msg import Contact, Contacts, JointWrench
from std_srvs.srv import Trigger

from arena_robots.lever_door_opener import LeverDoorOpener, Phase


def test_lever_door_opener_initialization():
    node = LeverDoorOpener()
    try:
        assert node._phase == Phase.IDLE
        assert node.MAX_HANDLE_FORCE_N == 22.0
        assert node.MAX_CONTACT_SPEED_MPS == 0.05
        assert node._lever_xyz == (7.92, 1.70, 1.01)
    finally:
        node.destroy_node()


def test_contact_force_calculation():
    node = LeverDoorOpener()
    try:
        wrench = JointWrench(
            body_1_wrench=Wrench(
                force=Vector3(x=3.0, y=4.0, z=0.0),
                torque=Vector3(),
            )
        )
        contact = Contact()
        contact.wrenches.append(wrench)
        msg = Contacts()
        msg.contacts.append(contact)

        node._on_contact(msg)
        assert abs(node._contact_force - 5.0) < 1e-4
        assert abs(node._peak_force - 5.0) < 1e-4
    finally:
        node.destroy_node()


def test_trigger_open_and_stop():
    node = LeverDoorOpener()
    try:
        req = Trigger.Request()
        resp = Trigger.Response()

        resp = node._on_open(req, resp)
        assert resp.success is True
        assert node._phase == Phase.APPROACH_ABOVE

        resp = node._on_stop(req, resp)
        assert resp.success is True
        assert node._phase == Phase.ABORTED
    finally:
        node.destroy_node()


def test_metrics_saving(tmp_path: Path):
    node = LeverDoorOpener()
    try:
        metrics_file = tmp_path / "test_metrics.json"
        node._metrics_path = str(metrics_file)
        node._peak_force = 12.5
        node._contact_speed = 0.032

        node._report_and_save_metrics()
        assert metrics_file.exists()

        data = json.loads(metrics_file.read_text())
        assert data["peak_handle_force_n"] == 12.5
        assert data["force_pass"] is True
        assert data["contact_speed_mps"] == 0.032
        assert data["speed_pass"] is True
        assert data["verdict"] == "SUCCESS"
    finally:
        node.destroy_node()


def test_omnidirectional_and_servo_raw_commands():
    node = LeverDoorOpener()
    try:
        # Verify omnidirectional base control and raw servo command methods execute cleanly
        node._send_base_vel(vx=0.2, vy=0.08, wz=0.0)
        node._send_servo_vel_raw(vx=3.5, vy=-4.0, vz=-1.0)
        node._stop_base()
        node._stop_arm()
    finally:
        node.destroy_node()
