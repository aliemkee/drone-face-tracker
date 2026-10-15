import time
import math
import threading
from dataclasses import dataclass, field

try:
    from pymavlink import mavutil
    _MAV_OK = True
except ImportError:
    _MAV_OK = False


@dataclass
class VelocityCmd:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class SimState:
    x:   float = 0.0
    y:   float = 0.0
    z:   float = 1.5
    yaw: float = 0.0
    log: list  = field(default_factory=list)

    def apply(self, cmd: VelocityCmd, dt: float = 0.033):
        self.x   += cmd.vx * dt
        self.y   += cmd.vy * dt
        self.z   -= cmd.vz * dt
        self.yaw += cmd.yaw_rate * dt
        entry = f"vx={cmd.vx:+.2f} vy={cmd.vy:+.2f} vz={cmd.vz:+.2f} yr={cmd.yaw_rate:+.2f}"
        self.log.append(entry)
        if len(self.log) > 200:
            self.log.pop(0)


class DroneController:
    def __init__(
        self,
        connection_string: str = "udp:127.0.0.1:14550",
        sim: bool = True,
        takeoff_altitude: float = 1.5,
        max_speed: float = 2.0,
        max_yaw_rate: float = 60.0,
    ):
        self.connection_string = connection_string
        self.sim               = sim or not _MAV_OK
        self.takeoff_altitude  = takeoff_altitude
        self.max_speed         = max_speed
        self.max_yaw_rate      = max_yaw_rate
        self._mav              = None
        self._connected        = False
        self._sim_state        = SimState()
        self._lock             = threading.Lock()

    def connect(self):
        if self.sim:
            print("[Drone] SIM mode — virtual drone ready.")
            self._connected = True
            return True
        if not _MAV_OK:
            raise RuntimeError("pymavlink not installed.")
        print(f"[Drone] Connecting → {self.connection_string} …")
        try:
            self._mav = mavutil.mavlink_connection(
                self.connection_string, autoreconnect=True, source_system=255
            )
            self._mav.wait_heartbeat(timeout=10)
            self._arm_and_takeoff()
            self._connected = True
            return True
        except Exception as e:
            print(f"[Drone] Connection failed: {e}")
            return False

    def send_velocity(self, cmd: VelocityCmd):
        if not self._connected:
            return
        vx = self._clamp(cmd.vx, self.max_speed)
        vy = self._clamp(cmd.vy, self.max_speed)
        vz = self._clamp(cmd.vz, self.max_speed)
        yr = self._clamp(cmd.yaw_rate, self.max_yaw_rate)
        safe = VelocityCmd(vx, vy, vz, yr)
        if self.sim:
            with self._lock:
                self._sim_state.apply(safe)
            return
        self._mav.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            self._mav.target_system,
            self._mav.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000_1111_0000_0111,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0,
            math.radians(yr),
        )

    def stop(self):
        self.send_velocity(VelocityCmd())

    def land(self):
        if self.sim:
            return
        if self._mav:
            self._mav.mav.command_long_send(
                self._mav.target_system, self._mav.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0,
            )

    def disconnect(self):
        self.stop()
        if self._mav:
            self._mav.close()
        self._connected = False

    def _arm_and_takeoff(self):
        self._mav.arducopter_arm()
        self._mav.motors_armed_wait()
        self._mav.mav.command_long_send(
            self._mav.target_system, self._mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, self.takeoff_altitude,
        )
        time.sleep(3)

    @staticmethod
    def _clamp(v, lim):
        return max(-lim, min(lim, v))

    @property
    def sim_state(self):
        return self._sim_state
