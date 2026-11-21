import math
import os
from typing import Optional, Set, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd, LowState

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False


JOINT_NAMES = [
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
]

# Пределы углов 12 суставов Unitree Go1 (рад)
Q_MIN = np.array([
    -0.863, -0.686, -2.818,
    -0.863, -0.686, -2.818,
    -0.863, -0.686, -2.818,
    -0.863, -0.686, -2.818,
], dtype=np.float64)

Q_MAX = np.array([
    0.863, 4.501, -0.888,
    0.863, 4.501, -0.888,
    0.863, 4.501, -0.888,
    0.863, 4.501, -0.888,
], dtype=np.float64)

MAX_STEP_RAD = math.radians(4.95)     
INTERP_STEP_RAD = math.radians(0.1)   
MAX_PENETRATION_M = 0.00095          
HOLD_TIMEOUT_SEC = 0.40              
PUBLISH_RATE_HZ = 100.0              

DEFAULT_HOLD_KP = 40.0
DEFAULT_HOLD_KD = 1.0
MAX_KP = 500.0
MAX_KD = 50.0
ACTIVE_MODE = 0x0A


class CollisionChecker:
    """Headless-проверка самоколлизий Go1 в MuJoCo."""

    def __init__(self, xml_path: str, logger):
        self.logger = logger
        self.is_active = False
        
        if not MUJOCO_AVAILABLE:
            self.logger.error('Библиотека mujoco не установлена. Проверка коллизий ОТКЛЮЧЕНА!')
            return
            
        if not os.path.isfile(xml_path):
            self.logger.error(f'Файл модели не найден: {xml_path}. Проверка коллизий ОТКЛЮЧЕНА!')
            return

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.qpos_indices = []
        for name in JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id < 0:
                raise RuntimeError(f'Сустав {name} не найден в {xml_path}')
            self.qpos_indices.append(int(self.model.jnt_qposadr[jnt_id]))


        self.adjacent_bodies: Set[Tuple[int, int]] = set()
        for b in range(1, self.model.nbody):
            p = int(self.model.body_parentid[b])
            if p > 0:
                self.adjacent_bodies.add((min(b, p), max(b, p)))

        self.is_active = True
        self.logger.info(f'Модель {xml_path} загружена (ngeom={self.model.ngeom}).')

    def _is_relevant_pair(self, geom1: int, geom2: int) -> bool:
        b1 = int(self.model.geom_bodyid[geom1])
        b2 = int(self.model.geom_bodyid[geom2])
        if b1 == 0 or b2 == 0 or b1 == b2:
            return False
        if (min(b1, b2), max(b1, b2)) in self.adjacent_bodies:
            return False
        return True

    def has_collision(self, q: np.ndarray) -> bool:
        if not self.is_active:
            return False

        self.data.qpos[:] = 0.0
        if self.model.nq >= 19:
            self.data.qpos[2] = 1.0
            self.data.qpos[3] = 1.0

        for idx, qpos_adr in enumerate(self.qpos_indices):
            self.data.qpos[qpos_adr] = float(q[idx])

        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist < -MAX_PENETRATION_M and self._is_relevant_pair(int(c.geom1), int(c.geom2)):
                return True
        return False

    def is_transition_safe(self, q_start: np.ndarray, q_end: np.ndarray) -> bool:
        if not self.is_active:
            return True
            
        if self.has_collision(q_end):
            return False

        max_delta = float(np.max(np.abs(q_end - q_start)))
        if max_delta < 1e-6:
            return True

        num_steps = int(math.ceil(max_delta / INTERP_STEP_RAD))
        for k in range(1, num_steps):
            q_mid = q_start + (k / num_steps) * (q_end - q_start)
            if self.has_collision(q_mid):
                return False

        return True


class SafetyBlockNode(Node):
    def __init__(self):
        super().__init__('safety_block')

        self.declare_parameter('go1_xml_path', '/opt/mujoco_menagerie/unitree_go1/go1.xml')
        xml_path = self.get_parameter('go1_xml_path').get_parameter_value().string_value

        self.checker = CollisionChecker(xml_path, self.get_logger())

        self.initialized: bool = False
        self.last_safe_q: np.ndarray = np.zeros(12, dtype=np.float32)
        self.last_safe_dq: np.ndarray = np.zeros(12, dtype=np.float32)
        self.last_safe_tau: np.ndarray = np.zeros(12, dtype=np.float32)
        self.last_safe_kp: np.ndarray = np.full(12, DEFAULT_HOLD_KP, dtype=np.float32)
        self.last_safe_kd: np.ndarray = np.full(12, DEFAULT_HOLD_KD, dtype=np.float32)

        self.target_q: Optional[np.ndarray] = None
        self.target_dq: np.ndarray = np.zeros(12, dtype=np.float32)
        self.target_tau: np.ndarray = np.zeros(12, dtype=np.float32)
        self.target_kp: np.ndarray = np.full(12, DEFAULT_HOLD_KP, dtype=np.float32)
        self.target_kd: np.ndarray = np.full(12, DEFAULT_HOLD_KD, dtype=np.float32)

        self.last_input_time_ns: Optional[int] = None
        self.in_hold_mode: bool = True

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub_state = self.create_subscription(LowState, '/lowstate', self._on_lowstate, qos)
        self.sub_bare = self.create_subscription(LowCmd, '/lowcmd_bare', self._on_lowcmd_bare, qos)
        self.pub_cmd = self.create_publisher(LowCmd, '/lowcmd', qos)

        self.timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self._on_timer)
        self.get_logger().info('SafetyBlockNode запущен.')

    @staticmethod
    def _is_cmd_finite(msg: LowCmd) -> bool:
        for i in range(12):
            m = msg.motor_cmd[i]
            if not (
                math.isfinite(m.q)
                and math.isfinite(m.dq)
                and math.isfinite(m.tau)
                and math.isfinite(m.kp)
                and math.isfinite(m.kd)
            ):
                return False
        return True

    @staticmethod
    def _within_joint_limits(q: np.ndarray) -> bool:
        return bool(np.all(q >= Q_MIN) and np.all(q <= Q_MAX))

    @staticmethod
    def _sanitize_gains(kp: np.ndarray, kd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.clip(kp, 0.0, MAX_KP).astype(np.float32),
            np.clip(kd, 0.0, MAX_KD).astype(np.float32),
        )

    def _on_lowstate(self, msg: LowState) -> None:
        if self.initialized:
            return
        q_meas = np.array([msg.motor_state[i].q for i in range(12)], dtype=np.float64)
        if not np.all(np.isfinite(q_meas)):
            return
        q_init = np.clip(q_meas, Q_MIN + 1e-4, Q_MAX - 1e-4).astype(np.float32)
        self.last_safe_q = q_init
        self.target_q = q_init.copy()
        self.initialized = True
        self.get_logger().info('Начальная поза получена из /lowstate.')

    def _on_lowcmd_bare(self, msg: LowCmd) -> None:
        now_ns = self.get_clock().now().nanoseconds

        if not self._is_cmd_finite(msg):
            self.get_logger().warn('Команда отклонена: NaN/Inf.', throttle_duration_sec=1.0)
            return

        q_in = np.array([msg.motor_cmd[i].q for i in range(12)], dtype=np.float32)
        dq_in = np.array([msg.motor_cmd[i].dq for i in range(12)], dtype=np.float32)
        tau_in = np.array([msg.motor_cmd[i].tau for i in range(12)], dtype=np.float32)
        kp_in = np.array([msg.motor_cmd[i].kp for i in range(12)], dtype=np.float32)
        kd_in = np.array([msg.motor_cmd[i].kd for i in range(12)], dtype=np.float32)

        if np.any(kp_in < 0.0) or np.any(kd_in < 0.0):
            self.get_logger().warn('Команда отклонена: kp/kd < 0.', throttle_duration_sec=1.0)
            return

        if not self._within_joint_limits(q_in.astype(np.float64)):
            self.get_logger().warn('Команда отклонена: выход за пределы суставов.', throttle_duration_sec=1.0)
            return
        if self.checker.has_collision(q_in.astype(np.float64)):
            self.get_logger().warn('Команда отклонена: коллизия в целевой позе.', throttle_duration_sec=1.0)
            return


        if not self.initialized:
            self.last_safe_q = q_in.copy()
            self.last_safe_kp, self.last_safe_kd = self._sanitize_gains(kp_in, kd_in)
            self.initialized = True

        self.target_q = q_in.copy()
        self.target_dq = dq_in.copy()
        self.target_tau = tau_in.copy()
        self.target_kp, self.target_kd = self._sanitize_gains(kp_in, kd_in)
        self.last_input_time_ns = now_ns
        self.in_hold_mode = False

    def _on_timer(self) -> None:
        if not self.initialized:
            return

        now_ns = self.get_clock().now().nanoseconds

        if self.last_input_time_ns is None:
            self.in_hold_mode = True
        else:
            dt_sec = (now_ns - self.last_input_time_ns) * 1e-9
            if dt_sec >= HOLD_TIMEOUT_SEC:
                if not self.in_hold_mode:
                    self.get_logger().warn(f'Потеря потока ({dt_sec:.3f} с) -> режим удержания.')
                self.in_hold_mode = True

        if self.in_hold_mode:
            self.target_q = self.last_safe_q.copy()
            hold_kp = np.where(self.last_safe_kp > 1.0, self.last_safe_kp, DEFAULT_HOLD_KP).astype(np.float32)
            hold_kd = np.where(self.last_safe_kd > 0.1, self.last_safe_kd, DEFAULT_HOLD_KD).astype(np.float32)
            self._publish_cmd(
                q=self.last_safe_q,
                dq=np.zeros(12, dtype=np.float32),
                tau=np.zeros(12, dtype=np.float32),
                kp=hold_kp,
                kd=hold_kd,
            )
            return

        assert self.target_q is not None
        delta = self.target_q.astype(np.float64) - self.last_safe_q.astype(np.float64)
        max_abs_delta = float(np.max(np.abs(delta)))

        if max_abs_delta > MAX_STEP_RAD:
            scale = MAX_STEP_RAD / max_abs_delta
            q_candidate = (self.last_safe_q.astype(np.float64) + delta * scale).astype(np.float32)
            reached_target = False
        else:
            q_candidate = self.target_q.astype(np.float32)
            reached_target = True

        q_cand_f64 = q_candidate.astype(np.float64)
        q_prev_f64 = self.last_safe_q.astype(np.float64)
        step_ok = float(np.max(np.abs(q_cand_f64 - q_prev_f64))) <= math.radians(5.0) - 1e-5
        bounds_ok = self._within_joint_limits(q_cand_f64)

        if bounds_ok and step_ok and self.checker.is_transition_safe(q_prev_f64, q_cand_f64):
            self.last_safe_q = q_candidate
            self.last_safe_kp = self.target_kp.copy()
            self.last_safe_kd = self.target_kd.copy()
            if reached_target:
                self.last_safe_dq = self.target_dq.copy()
                self.last_safe_tau = self.target_tau.copy()
            else:
                self.last_safe_dq = np.zeros(12, dtype=np.float32)
                self.last_safe_tau = np.zeros(12, dtype=np.float32)
        else:
            self.get_logger().warn('Переход небезопасен -> режим удержания.', throttle_duration_sec=1.0)
            self.target_q = self.last_safe_q.copy()
            self.last_safe_dq = np.zeros(12, dtype=np.float32)
            self.last_safe_tau = np.zeros(12, dtype=np.float32)

        self._publish_cmd(
            q=self.last_safe_q,
            dq=self.last_safe_dq,
            tau=self.last_safe_tau,
            kp=self.last_safe_kp,
            kd=self.last_safe_kd,
        )

    def _publish_cmd(self, q, dq, tau, kp, kd):
        msg = LowCmd()
        for i in range(12):
            msg.motor_cmd[i].mode = ACTIVE_MODE  # 0x0A (10) — позиционный режим
            msg.motor_cmd[i].q = float(q[i])
            msg.motor_cmd[i].dq = float(dq[i])
            msg.motor_cmd[i].tau = float(tau[i])
            msg.motor_cmd[i].kp = float(kp[i])
            msg.motor_cmd[i].kd = float(kd[i])
            

        self.pub_cmd.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = SafetyBlockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
