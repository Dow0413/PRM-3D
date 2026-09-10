#!/usr/bin/env python3
"""cmu_replan: CMU 局部规划的重规划监视节点（卡死触发）。

与 local_replan 的差别只有触发逻辑：

  local_replan：前视路径被障碍挡住（lookahead + robot_radius + 连续帧去抖）
               即触发重规划；
  cmu_replan：  CMU localPlanner 自己会做局部避障，被顶住时的表现不是报错，
               而是机器狗在原地反复求解不挪窝，所以用卡死触发——
               位置持续保持在 use_radius 半径内超过 use_time 秒（一旦移出
               半径立刻重新锚定重新计时）→ 判定 CMU 无法求解。

其余部分与 local_replan 完全一致（同名参数、同默认值、同管线）：

  障碍管线   cloud_step 采样跳步 + 分楼层 Z 带（maps.<env>.floors 展开，无
             楼层配置时退回全局 obstacle_z_min/max）+ 感知半径 +
             栅格化降采样 + TF 到 map；
  持续入图   obstacle_ingest_rate 节拍把过滤点云以 ingest_only 请求写入全局
             地图（只入图不规划不发布），记忆 obstacle_decay_sec（刻意短，
             避免绕已消失的障碍）；卡死触发的完整重规划直接消费这份记忆，
             本帧 0 点但记忆内有障碍时也可重规划；
  触发通道   replan_mode=service 调 ReplanPlan 服务（起点+原终点+障碍点云）；
             =goal 重发原终点到 goal_topic（与无服务的全局算法解耦，超时/
             新路径与旧相同按失败处理——本节点不碰速度，只记日志）；
  分楼层开关 enable_replan=false 的楼层（楼梯层）不触发也不入图。

enable_replan=false 时节点空转（不订阅、不定时器），CMU 行为完全不变。
到达终点附近（goal_tolerance 内）不算卡死，同样空转。
"""

import math

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener

from prm_interfaces.srv import ReplanPlan


class CmuReplan(Node):
    def __init__(self) -> None:
        super().__init__("replan")

        # ---- 卡死触发参数（cmu 特有）----
        self.declare_parameter("enable_replan", False)   # false = 纯 CMU，无重规划
        self.declare_parameter("cloud_step", 1)
        self.declare_parameter("replan_debug", False)
        self.declare_parameter("replan_debug_period", 1.0)
        # float 参数用动态类型声明：yaml 里写整数（use_time: 10）或小数（10.0）
        # 都收，锁 DOUBLE 会在整数配置时抛 InvalidParameterTypeException。
        self.use_radius = self._float_param("use_radius", 0.15)       # 卡死判定半径 (m)
        self.use_time = self._float_param("use_time", 5.0)            # 卡死判定时长 (s)
        self.check_rate = self._float_param("replan_check_rate", 10.0)
        self.interval_s = self._float_param("replan_interval_s", 3.0) # 请求最小间隔 (s)
        self.goal_tol = self._float_param("goal_tolerance", 0.5)      # 距终点小于此值不算卡死
        self.z_tol = self._float_param("floor_z_tolerance", 0.8)

        # ---- 共享管线参数（与 local_replan 同名同默认）----
        self.declare_parameter("replan_service", "/replan_plan")
        self.declare_parameter("replan_mode", "service")        # service | goal（重发终点话题）
        self.declare_parameter("goal_topic", "/goal_pose")      # goal 模式：重发原终点的目标点话题
        self.declare_parameter("replan_response_timeout", 2.0)  # goal 模式：等待新路径超时 (s)
        self.declare_parameter("obstacle_cloud_topic", "/front/rslidar_points")
        self.declare_parameter("cloud_target_frame", "map")
        self.declare_parameter("obstacle_perception_radius", 8.0)
        self.declare_parameter("obstacle_sparse_distance", 0.15)
        self.declare_parameter("obstacle_decay_sec", 8.0)       # 障碍记忆时长 (s)
        self.declare_parameter("obstacle_ingest_rate", 2.0)     # 持续入图频率 (Hz)
        # 无楼层配置时的兜底 Z 带（相对机器人高度）；maps 楼层配置生效时被覆盖
        self.obstacle_z_min = self._float_param("obstacle_z_min", -0.1)
        self.obstacle_z_max = self._float_param("obstacle_z_max", 1.2)
        # 分楼层（maps.<map_name>.floors 由 launch 展开成并行数组，同 local_replan）
        self.declare_parameter("floor_pngs", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("floor_z_mins", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("floor_z_maxs", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("floor_obstacle_z_mins", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("floor_obstacle_z_maxs", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("floor_enable_replans", Parameter.Type.BOOL_ARRAY)

        # ---- 话题 ----
        self.declare_parameter("path_topic", "/global_path")
        self.declare_parameter("odom_topic", "/Odometry")

        self.enable_replan = bool(self.get_parameter("enable_replan").value)
        self.use_radius = max(0.05, self.use_radius)
        self.use_time = max(0.5, self.use_time)
        self.check_rate = max(0.5, self.check_rate)
        self.interval_s = max(0.0, self.interval_s)
        self.goal_tol = max(0.05, self.goal_tol)
        self.z_tol = max(0.2, self.z_tol)
        self.perception_radius = self._num(
            self.get_parameter("obstacle_perception_radius").value, 8.0)
        self.sparse_distance = max(
            0.02, self._num(self.get_parameter("obstacle_sparse_distance").value, 0.15))
        self.decay_sec = max(1.0, self._num(
            self.get_parameter("obstacle_decay_sec").value, 8.0))
        self.ingest_rate = min(10.0, max(0.1, self._num(
            self.get_parameter("obstacle_ingest_rate").value, 2.0)))
        self.cloud_step = max(1, int(self.get_parameter("cloud_step").value))
        self.replan_debug = bool(self.get_parameter("replan_debug").value)
        self.debug_period = max(
            0.1, self._num(self.get_parameter("replan_debug_period").value, 1.0))
        self.replan_service = str(self.get_parameter("replan_service").value)
        self.goal_mode = (
            str(self.get_parameter("replan_mode").value).strip().lower() == "goal"
        )
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.response_timeout = max(
            0.5, self._num(self.get_parameter("replan_response_timeout").value, 2.0))
        self.cloud_topic = str(self.get_parameter("obstacle_cloud_topic").value)
        self.cloud_target_frame = (
            str(self.get_parameter("cloud_target_frame").value).strip() or "map"
        )

        # 分楼层数组（缺失/长度不齐 → 退回全局兜底 Z 带）
        self.floor_pngs = [str(v) for v in self.get_parameter("floor_pngs").value or []]
        self.floor_z_mins = [float(v) for v in self.get_parameter("floor_z_mins").value or []]
        self.floor_z_maxs = [float(v) for v in self.get_parameter("floor_z_maxs").value or []]
        self.floor_z_mins_obstacle = [
            float(v) for v in self.get_parameter("floor_obstacle_z_mins").value or []]
        self.floor_z_maxs_obstacle = [
            float(v) for v in self.get_parameter("floor_obstacle_z_maxs").value or []]
        self.floor_enables = [
            bool(v) for v in self.get_parameter("floor_enable_replans").value or []]
        n = len(self.floor_z_mins)
        if n and len(self.floor_z_maxs) == n and len(self.floor_z_mins_obstacle) == n \
                and len(self.floor_z_maxs_obstacle) == n and len(self.floor_enables) == n:
            self.use_floor_config = True
        else:
            self.use_floor_config = False
            if n or self.floor_pngs:
                self.get_logger().error(
                    "floor arrays missing/mismatched — falling back to the global "
                    "obstacle_z band; check maps.<map_name>.floors in local_replan.yaml"
                )

        if not self.enable_replan:
            self.get_logger().info(
                "cmu replan disabled (enable_replan=false) - pure CMU behaviour"
            )
            return

        path_topic = str(self.get_parameter("path_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        self._odom_topic_name = odom_topic
        check_rate = self.check_rate

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.x = self.y = self.z = self.yaw = None
        self.points = []            # [(x, y, z)] 当前全局路径
        self.latest_cloud = None
        self.anchor = None          # (x, y) 卡死判定锚点
        self.anchor_sec = None      # 锚点时刻
        self.last_request_sec = -1e9
        self._last_debug_state = None
        self._last_check_log_sec = -1e9
        self._last_diag_sec = -1e9
        self._zstats = (0.0, 0.0, 0)   # 变换后点云 z 范围（0 点过滤诊断用）
        self._last_ingest_sec = -1e9   # 最近一次带障碍点的入图时刻（卡死 0 点放宽判定用）
        self.active_floor = None       # 当前楼层（分楼层配置时）

        self.create_subscription(Path, path_topic, self.path_cb, 10)
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.create_subscription(
            PointCloud2, self.cloud_topic, self.cloud_cb, qos_profile_sensor_data
        )
        self.replan_cli = self.create_client(ReplanPlan, self.replan_service)
        # goal 降级通道：重发原终点触发对方规划器（无 ReplanPlan 服务的算法）
        self.goal_pub = (
            self.create_publisher(PoseStamped, self.goal_topic, 10)
            if self.goal_mode else None
        )
        self._awaiting_since = None    # goal 模式：发出终点后等待新路径的起始时刻
        self._last_path_sig = None     # 最近一条 /global_path 的签名（判新旧路径是否相同）
        self._sig_at_trigger = None    # 触发重规划那一刻的路径签名
        self.create_timer(1.0 / check_rate, self.check_cb)
        # 持续入图：与卡死检测并行的独立节拍（ingest 只写地图，不触发重规划）
        self.create_timer(1.0 / self.ingest_rate, self.ingest_cb)

        floors_desc = ""
        if self.use_floor_config:
            enabled = [p for p, e in zip(self.floor_pngs, self.floor_enables) if e]
            floors_desc = (
                f", {len(self.floor_z_mins)} floors [{','.join(enabled)}]"
                f" (band per floor)"
            )
        else:
            floors_desc = (
                f", global band [{self.obstacle_z_min:.2f}, "
                f"{self.obstacle_z_max:.2f}] (no floor config)"
            )
        self.get_logger().info(
            f"cmu replan monitor ready: stuck = within {self.use_radius:.2f} m "
            f"for {self.use_time:.1f} s, cloud={self.cloud_topic}, "
            f"service={self.replan_service}, odom={odom_topic}, path={path_topic}"
            f"{floors_desc}, "
            f"ingest={self.ingest_rate:.1f} Hz (memory {self.decay_sec:.0f} s), "
            f"trigger={'goal ' + self.goal_topic if self.goal_mode else 'service'}"
        )
        # 图级诊断：订阅建立 3 s 后报一次发布者发现情况（0 = 该话题上无人发布，
        # 检查话题名/仿真是否在跑；>0 但没数据 = 发布者在但不发，常见为仿真暂停）。
        self.create_timer(3.0, self._report_graph_once)

    # ---------------------------------------------------- param helpers
    def _float_param(self, name, default):
        """动态类型 float 参数：yaml/CLI 里整数或小数都能配（use_time: 10 == 10.0）。"""
        self.declare_parameter(
            name, None, ParameterDescriptor(dynamic_typing=True)
        )
        value = self.get_parameter(name).value
        return default if value is None else float(value)

    @staticmethod
    def _num(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ---------------------------------------------------- inputs
    def odom_cb(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        self.x, self.y, self.z = pos.x, pos.y, pos.z
        self.yaw = self.extract_yaw(msg.pose.pose.orientation)

    @staticmethod
    def extract_yaw(q) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def path_cb(self, msg: Path) -> None:
        pts = [(p.pose.position.x, p.pose.position.y, p.pose.position.z)
               for p in msg.poses]
        if len(pts) < 2:
            return
        self.points = pts
        self.anchor = None          # 新路径 → 重新锚定（成功重规划的自然恢复点）
        self._last_debug_state = None
        self._last_path_sig = self._path_signature(msg)
        if self.goal_mode and self._awaiting_since is not None:
            self._awaiting_since = None
            if self._last_path_sig == self._sig_at_trigger:
                # 原路重算：对方不知道障碍，跟着走只会再撞。本节点不碰速度，
                # 只按失败记日志；卡死保持，间隔后会再触发。
                self.get_logger().warn(
                    "[重规划情况] 失败: 新路径与原路径相同 (对方规划器未感知障碍?) "
                    "- CMU 继续自行尝试, 间隔后重试"
                )
        self.get_logger().info(
            f"[规划情况] 新路径: {len(pts)} 点, "
            f"终点 ({pts[-1][0]:.2f}, {pts[-1][1]:.2f}, {pts[-1][2]:.2f})"
        )

    def cloud_cb(self, msg: PointCloud2) -> None:
        self.latest_cloud = msg

    def _debug(self, state, msg) -> None:
        """replan_debug 诊断：高频状态行按 replan_debug_period 节流，其余按
        内容去重；WARN 级关键日志（触发/成败）不经此处、永远即时打印。"""
        if not self.replan_debug:
            return
        if state != "check" and msg == self._last_debug_state:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if state == "check" and now - self._last_check_log_sec < self.debug_period:
            return
        self._last_check_log_sec = now
        self._last_debug_state = msg
        self.get_logger().info(msg)

    # ---------------------------------------------------- floor helpers
    def _current_floor(self):
        """机器人当前楼层（分楼层配置时；无配置返回 None）。"""
        if not self.use_floor_config:
            return None
        return self._floor_of_z(self.z)

    def _floor_of_z(self, z):
        """Z 落在哪层的 [z_min, z_max]：首个命中即返回（与全局端 floorFromZ
        同规则）；落在层间缝隙时吸附到最近的带边缘。"""
        best, best_d = None, float("inf")
        for i, (lo, hi) in enumerate(zip(self.floor_z_mins, self.floor_z_maxs)):
            if lo <= z <= hi:
                return i
            d = (lo - z) if z < lo else (z - hi)
            if d < best_d:
                best_d, best = d, i
        return best

    def _obstacle_band(self, floor):
        """该楼层的避障 Z 带（相对机器人高度）；无楼层配置时用全局兜底带。"""
        if self.use_floor_config and floor is not None:
            return self.floor_z_mins_obstacle[floor], self.floor_z_maxs_obstacle[floor]
        return self.obstacle_z_min, self.obstacle_z_max

    def _resolve_floor(self):
        """楼层解析 + 切层播报 + 开关过滤。返回 (floor, ok)：ok=False 表示当前
        楼层退出了重规划（或 Z 无法定层），本周期既不触发也不入图——与
        local_replan 的楼层早退分支一致。"""
        if not self.use_floor_config:
            return None, True
        floor = self._floor_of_z(self.z)
        if floor is None:
            return None, False
        if floor != self.active_floor:
            self.active_floor = floor
            self.get_logger().info(
                f"[当前楼层] F{floor} ({self.floor_pngs[floor]}): "
                f"enable_replan={self.floor_enables[floor]}, "
                f"避障Z带=[{self.floor_z_mins_obstacle[floor]:.2f}, "
                f"{self.floor_z_maxs_obstacle[floor]:.2f}]"
            )
        if not self.floor_enables[floor]:
            self._debug(
                "disabled",
                f"replan watch: floor {floor} ({self.floor_pngs[floor]}) "
                "has enable_replan=false",
            )
            return floor, False
        return floor, True

    # ---------------------------------------------------- stuck detection
    def check_cb(self) -> None:
        if self.goal_mode:
            self._goal_watchdog()
        if self.x is None or not self.points:
            self._debug("idle", "replan watch: idle (no odom / no path)")
            return

        goal = self.points[-1]
        if (math.hypot(goal[0] - self.x, goal[1] - self.y) < self.goal_tol
                and abs(goal[2] - self.z) < self.z_tol):
            self.anchor = None      # 到达终点附近停车不是卡死
            self._debug("goal", "replan watch: at goal, idle")
            return

        floor, floor_ok = self._resolve_floor()
        if not floor_ok:
            self.anchor = None      # 退出重规划的楼层不累计卡死计时
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.anchor is None:
            self.anchor = (self.x, self.y)
            self.anchor_sec = now
            self._debug("anchor", f"replan watch: anchored at ({self.x:.2f}, {self.y:.2f})")
            return

        moved = math.hypot(self.x - self.anchor[0], self.y - self.anchor[1])
        if moved > self.use_radius:
            # 移出半径 = 在动（哪怕慢），重新锚定重新计时
            self.anchor = (self.x, self.y)
            self.anchor_sec = now
            self._debug(
                "moved", f"replan watch: moved {moved:.2f} m > radius, re-anchored"
            )
            return

        stuck_sec = now - self.anchor_sec
        if stuck_sec < self.use_time:
            self._debug(
                "check",
                f"replan watch: holding {stuck_sec:.1f}/{self.use_time:.1f} s "
                f"within {self.use_radius:.2f} m",
            )
            return
        if now - self.last_request_sec < self.interval_s:
            return

        # ---- 卡死确认：构建请求（与 local_replan 相同的点云管线）----
        self.last_request_sec = now
        if self.latest_cloud is None:
            pubs = self.count_publishers(self.cloud_topic)
            self._warn_diag(
                f"[触发重规划] 推迟: 卡死但 {self.cloud_topic} 上还没收到点云 "
                f"(图上发布者 {pubs} 个) - 空请求会做无障碍重规划陷入死循环"
            )
            return
        pts = self._filter_cloud(self.latest_cloud, *self._obstacle_band(floor))
        if pts is None:
            self._warn_diag(
                f"[触发重规划] 推迟: 点云 TF "
                f"{self.latest_cloud.header.frame_id or '?'} -> "
                f"{self.cloud_target_frame} 不可用, 检查 TF 树"
            )
            return
        if not pts:
            # 本帧 0 点 ≠ 不能重规划：近 decay_sec 内入图过 → 全局地图记忆里
            # 有障碍（服务端按记忆规划），依然值得重规划；只有「一直没有任何
            # 障碍记忆」才推迟（否则空请求原样重规划 → 卡死-重规划死循环）。
            if now - self._last_ingest_sec < self.decay_sec:
                self.get_logger().info(
                    f"[触发重规划] 本帧 0 障碍点, 按地图记忆重规划 "
                    f"(障碍入图于 {now - self._last_ingest_sec:.1f} s 前)"
                )
            else:
                zmin, zmax, n_seen = self._zstats
                self._warn_diag(
                    f"[触发重规划] 推迟: 过滤后 0 障碍点且无记忆 "
                    f"(点云 z=[{zmin:.2f}, {zmax:.2f}], 机器人 z={self.z:.2f}, "
                    f"避障带=[{self.z + self.obstacle_z_min:.2f}, "
                    f"{self.z + self.obstacle_z_max:.2f}], "
                    f"感知半径={self.perception_radius:.0f} m) - "
                    "放宽 obstacle_z_min/max 或 obstacle_perception_radius"
                )
                return

        self.anchor_sec = now       # 触发后重新计时：新路径仍卡住会再次触发
        if self.goal_mode:
            self._awaiting_since = now
            self._sig_at_trigger = self._last_path_sig
            self._publish_goal(floor)
        else:
            self._request_replan(pts, floor)

    def _warn_diag(self, msg) -> None:
        """触发条件已满足但无法构建带障碍的请求：节流告警（卡死未解除期间
        每 5 s 提醒一次），不调服务。"""
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_diag_sec < 5.0:
            return
        self._last_diag_sec = now
        self.get_logger().warn(msg)

    # ---------------------------------------------------- obstacle ingest
    # 持续入图（与 local_replan 相同）：障碍不等触发——obstacle_ingest_rate
    # 频率把同一管线过滤出的点云以 ingest_only 请求写入全局地图（只入图不
    # 规划不发布），记忆时长随请求传 obstacle_decay_sec。重规划只由卡死检测
    # 触发，触发时全局端已有这份累积记忆可绕。
    def ingest_cb(self) -> None:
        if self.latest_cloud is None or self.x is None:
            return
        floor, floor_ok = self._resolve_floor()
        if not floor_ok:
            return  # 退出重规划的楼层（楼梯层）也不入图，同 local_replan
        pts = self._filter_cloud(self.latest_cloud, *self._obstacle_band(floor))
        if not pts:                     # None（TF 暂不可用）或 []（本帧没障碍）
            return
        if not self.replan_cli.service_is_ready():
            self._debug("ingest", "ingest: replan service not ready yet")
            return

        req = ReplanPlan.Request()
        req.start.header.frame_id = self.cloud_target_frame
        req.start.header.stamp = self.get_clock().now().to_msg()
        req.start.pose.position.x = self.x
        req.start.pose.position.y = self.y
        req.start.pose.position.z = self.z
        req.start.pose.orientation.z = math.sin(self.yaw / 2.0)
        req.start.pose.orientation.w = math.cos(self.yaw / 2.0)
        req.obstacles = pc2.create_cloud_xyz32(
            req.start.header, [(p[0], p[1], p[2]) for p in pts]
        )
        req.ingest_only = True          # 只入图累积，不规划不发布
        req.decay_sec = self.decay_sec  # 短记忆：障碍挪走后很快过期
        self._last_ingest_sec = self.get_clock().now().nanoseconds / 1e9
        self._debug(
            "ingest", f"ingest: {len(pts)} obstacle pts accumulated "
            f"(memory {self.decay_sec:.0f} s)"
        )
        future = self.replan_cli.call_async(req)
        future.add_done_callback(self._on_ingest_response)

    def _on_ingest_response(self, future) -> None:
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001 - 服务调用本身失败
            self.get_logger().warn(f"[障碍入图] 服务调用失败: {exc}")
            return
        if not resp.success:
            self.get_logger().warn("[障碍入图] 服务端拒绝 ingest-only 请求")

    # ---------------------------------------------------- replan service call
    def _request_replan(self, pts, floor) -> None:
        if not self.replan_cli.service_is_ready():
            self.get_logger().warn(
                f"[触发重规划] 失败: 重规划服务 {self.replan_service} 未就绪, "
                "本次请求丢弃 (间隔后自动重试)"
            )
            return

        req = ReplanPlan.Request()
        req.start.header.frame_id = self.cloud_target_frame
        req.start.header.stamp = self.get_clock().now().to_msg()
        req.start.pose.position.x = self.x
        req.start.pose.position.y = self.y
        req.start.pose.position.z = self.z
        req.start.pose.orientation.z = math.sin(self.yaw / 2.0)
        req.start.pose.orientation.w = math.cos(self.yaw / 2.0)

        gx, gy, gz = self.points[-1]        # 原终点（未被障碍影响的任务目标）
        req.goal.header = req.start.header
        req.goal.pose.position.x = gx
        req.goal.pose.position.y = gy
        req.goal.pose.position.z = gz
        req.goal.pose.orientation.w = 1.0

        req.obstacles = pc2.create_cloud_xyz32(
            req.start.header, [(p[0], p[1], p[2]) for p in pts]
        )
        req.decay_sec = self.decay_sec  # 完整重规划送的点也用同样的短记忆

        where = (f"楼层 F{floor} ({self.floor_pngs[floor]}): "
                 if floor is not None else "")
        self.get_logger().warn(
            f"[触发重规划] {where}卡死 {self.use_time:.0f} s "
            f"(半径 {self.use_radius:.2f} m 内未移出), {len(pts)} 个障碍点; "
            f"起点 ({self.x:.2f}, {self.y:.2f}, {self.z:.2f}) "
            f"-> 终点 ({gx:.2f}, {gy:.2f}, {gz:.2f}) -> 调用重规划服务"
        )
        future = self.replan_cli.call_async(req)
        future.add_done_callback(self._on_replan_response)

    def _on_replan_response(self, future) -> None:
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001 - 服务调用本身失败
            self.get_logger().error(f"[重规划情况] 失败: 服务调用异常 {exc}")
            return
        if resp.success and len(resp.path.poses) >= 2:
            # 成功详情由全局端 [规划情况] 行（点数/cost/耗时）+ 本节点 path_cb 的
            # [规划情况] 新路径 行报告，这里不再重复打一条。
            return
        self.get_logger().warn(
            f"[重规划情况] 失败: 各级间距均无路 - CMU 继续自行尝试, "
            f"卡死期间每 {self.interval_s:.0f} s 重试 (障碍过期/挪走后自动恢复)"
        )

    # ---------------------------------------------------- goal-mode fallback
    # replan_mode=goal：不调服务，把原任务终点重发到 goal_topic 触发对方全局
    # 规划器（它用自己的起点/障碍信息重规划）。服务模式里免费拿到的失败反馈
    # 在这里自己补：发出终点后 replan_response_timeout 内没有新路径、或新路径
    # 与旧完全相同（盲重规划的典型结果）→ 按失败记日志（本节点不碰速度，
    # CMU 继续自行尝试）。持续入图不受影响——服务在就继续入图（对自家 PRM，
    # goal 触发的重规划仍会绕开记忆中的障碍），服务不在则自动静默跳过。
    @staticmethod
    def _path_signature(msg) -> tuple:
        return tuple(
            (round(p.pose.position.x, 3), round(p.pose.position.y, 3),
             round(p.pose.position.z, 3))
            for p in msg.poses
        )

    def _goal_watchdog(self) -> None:
        """goal 模式看门狗：发出终点后超时没有新路径 → 按规划失败记日志。"""
        if self._awaiting_since is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._awaiting_since <= self.response_timeout:
            return
        self._awaiting_since = None
        self.get_logger().warn(
            f"[重规划情况] 失败: 重发终点后 {self.response_timeout:.1f} s 内未收到"
            "新路径 - CMU 继续自行尝试, 间隔后重试"
        )

    def _publish_goal(self, floor) -> None:
        """goal 模式触发：原任务终点原样重发（对方规划器自行取当前位姿为
        起点）。起点/障碍不随行——这是与 service 模式的本质差别。"""
        msg = PoseStamped()
        msg.header.frame_id = self.cloud_target_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        gx, gy, gz = self.points[-1]   # 原终点（未被障碍影响的任务目标）
        msg.pose.position.x = gx
        msg.pose.position.y = gy
        msg.pose.position.z = gz
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)
        where = (f"楼层 F{floor} ({self.floor_pngs[floor]}): "
                 if floor is not None else "")
        self.get_logger().warn(
            f"[触发重规划] {where}卡死 {self.use_time:.0f} s "
            f"(半径 {self.use_radius:.2f} m 内未移出), CMU 无法求解 "
            f"-> [发布终点] 原终点 ({gx:.2f}, {gy:.2f}, {gz:.2f}) "
            f"重发到 {self.goal_topic}"
        )

    # ---------------------------------------------------- cloud filter
    # 与 local_replan._cloud_transform/_filter_cloud 相同（含分楼层 Z 带参数；
    # 两边若要改需同步）。
    def _cloud_transform(self, cloud):
        """Rotation+translation mapping cloud points into cloud_target_frame.

        Returns () when no transform is needed, a 12-tuple otherwise, or None
        when TF is unavailable (caller should skip this cycle)."""
        src = cloud.header.frame_id or self.cloud_target_frame
        if src == self.cloud_target_frame:
            return ()
        tf = None
        try:
            tf = self.tf_buffer.lookup_transform(
                self.cloud_target_frame, src, Time.from_msg(cloud.header.stamp),
                Duration(seconds=0.05),
            )
        except Exception:
            try:  # fall back to the latest available transform
                tf = self.tf_buffer.lookup_transform(self.cloud_target_frame, src, Time())
            except Exception:
                return None
        t = tf.transform
        x, y, z, w = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
        return (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), t.translation.x,
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), t.translation.y,
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), t.translation.z,
        )

    def _filter_cloud(self, cloud, z_min, z_max):
        """PointCloud2 -> [(x, y, z)] in cloud_target_frame。采样跳步 + 高度带
        （相对机器人，排除地面/天花板）+ 感知半径过滤 + 栅格化降采样。"""
        if cloud is None:
            return []
        xf = self._cloud_transform(cloud)
        if xf is None:
            return None
        pts = []
        radius_sq = self.perception_radius * self.perception_radius
        inv_sparse = 1.0 / self.sparse_distance
        z_low = self.z + z_min
        z_high = self.z + z_max
        seen = set()
        zmin = zmax = None
        try:
            gen = pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)
        except (ValueError, KeyError, TypeError):
            return []
        for i, (px, py, pz) in enumerate(gen):
            if i % self.cloud_step:
                continue  # 采样跳步
            if xf:
                qx = xf[0] * px + xf[1] * py + xf[2] * pz + xf[3]
                qy = xf[4] * px + xf[5] * py + xf[6] * pz + xf[7]
                pz = xf[8] * px + xf[9] * py + xf[10] * pz + xf[11]
                px, py = qx, qy
            if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
                continue  # inf/nan 无效回波：skip_nans 不滤 inf，变换后成 NaN 会炸网格键
            if zmin is None:
                zmin = zmax = pz
            else:
                zmin = min(zmin, pz)
                zmax = max(zmax, pz)
            if pz < z_low or pz > z_high:
                continue
            dx = px - self.x
            dy = py - self.y
            if dx * dx + dy * dy > radius_sq:
                continue
            key = (int(math.floor(px * inv_sparse)), int(math.floor(py * inv_sparse)))
            if key in seen:
                continue
            seen.add(key)
            pts.append((px, py, pz))
        self._zstats = (zmin or 0.0, zmax or 0.0, i + 1)
        return pts


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmuReplan()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
