#!/usr/bin/env python3
"""local_replan local planner: pure-pursuit tracking + dynamic-obstacle replanning.

safeplanner-2 架构的局部端，一个节点两件事：

1. Trajectory tracking (inherited from path_follower.PathFollower): /global_path
   + odometry -> /cmd_vel, with the same 3D progress/goal logic so stacked
   floors never hijack tracking.

2. Collision check + replan request (20 Hz background thread):
   - 点云处理：接收雷达点云，过滤掉 Z 轴范围以外及感知半径以外的点
     （采样跳步 + 高度带 + 半径），栅格化降采样（哈希表记录已占据网格），
     构建 scipy cKDTree 局部障碍物 KD 树（无 scipy 时退化为暴力遍历）。
   - 持续入图（与 cmu_replan 相同）：过滤出的障碍点以 obstacle_ingest_rate
     频率经 ingest_only 服务请求直接写入全局地图（只入图、不规划、不发布），
     记忆 obstacle_decay_sec——不等触发；触发重规划时全局端已有累积记忆，
     新路径直接绕开提前入图的障碍。
   - 触发通道（replan_mode）：service（默认）= ReplanPlan 服务，障碍点云随
     请求原子送达 + 失败反馈；goal = 把原任务终点重发到 goal_topic，由对方
     全局规划器自行重规划（与没有该服务的全局算法解耦）。goal 模式下障碍
     不随行——对方避障与否取决于它自己的障碍层；本节点用「新路径超时」和
     「新路径与旧路径完全相同」（盲重规划的典型结果）两个信号把失败补回
     来：停车保持、按 replan_interval_s 重试。
   - 碰撞检测：提取机器人前方一段局部预测路径，遍历路径上每个点，在
     KD 树中查最近障碍物；任意点与障碍物距离的平方 < 禁行距离的平方
     （robot_radius）→ 判定需要重规划（连续 N 帧去抖）。
   - 构建请求：当前位姿为起点、原终点为终点，连同转换到全局世界系
     （map 帧）的障碍点云一起，调用全局规划服务 ReplanPlan。
   - 下发并恢复追踪：成功 → 用返回的新路径替换跟踪路径（全局端同时
     发布 /global_path 供 RViz）；失败（障碍封死所有通道，含降级重试后
     仍无路）→ 停车保持（hold），触发器继续按间隔请求，障碍过期
     （obstacle_decay_sec）或挪走后自动恢复。

The active floor re-evaluates as the robot moves, so walking onto a stair floor
turns the trigger off mid-mission. Cloud frames: a map-frame cloud is used
as-is; a sensor-frame cloud (e.g. /front/lidar_points) is TF-transformed into
cloud_target_frame (default "map") before use — if TF is not available yet that
check cycle is skipped.
"""

import math
import threading
import time

import rclpy
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener

try:  # 局部障碍物 KD 树（safeplanner-2 用 Nanoflann，Python 侧用 scipy）
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - 退化路径
    cKDTree = None

from prm_interfaces.srv import ReplanPlan

from galileo_nav.path_follower import PathFollower


class LocalReplan(PathFollower):
    def __init__(self, node_name: str = "local_replan") -> None:
        # node_name 可被子类覆盖（wzh_replan 复用本类全部监控逻辑）。
        super().__init__(node_name=node_name)

        # ---- replan parameters ----
        # Common (all floors): service, debounce, robot size, perception radius.
        self.declare_parameter("replan_service", "/replan_plan")
        self.declare_parameter("replan_mode", "service")        # service | goal（重发终点话题）
        self.declare_parameter("goal_topic", "/goal_pose")      # goal 模式：重发原终点的目标点话题
        self.declare_parameter("replan_response_timeout", 2.0)  # goal 模式：等待新 /global_path 超时 (s)
        self.declare_parameter("obstacle_cloud_topic", "/local_cloud_map")
        self.declare_parameter("cloud_target_frame", "map")  # sensor-frame clouds are TF'd into this
        self.declare_parameter("replan_check_rate", 20.0)    # 碰撞检测线程频率 (Hz)
        self.declare_parameter("replan_lookahead_m", 3.0)
        self.declare_parameter("replan_consecutive_frames", 3)
        self.declare_parameter("replan_interval_s", 1.0)
        self.declare_parameter("robot_radius", 0.5)          # 禁行距离 (m)
        self.declare_parameter("obstacle_perception_radius", 8.0)
        self.declare_parameter("obstacle_sparse_distance", 0.15)
        self.declare_parameter("obstacle_ingest_rate", 2.0)   # 持续入图频率 (Hz)：不等触发直接写图
        self.declare_parameter("obstacle_decay_sec", 8.0)     # 入图障碍记忆时长 (s)
        self.declare_parameter("cloud_step", 1)              # 采样跳步：每 N 个点取 1 个
        self.declare_parameter("replan_debug", False)
        self.declare_parameter("replan_debug_period", 1.0)   # check 行日志节流周期 (s)
        # Per-floor (flattened from maps.<map_name>.floors by the launch):
        # each parallel array is indexed by floor, matching the PNG maps in
        # global_prm.yaml. The Z filter band and the enable switch are per PNG.
        self.declare_parameter("floor_pngs", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("floor_z_mins", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("floor_z_maxs", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("floor_obstacle_z_mins", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("floor_obstacle_z_maxs", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("floor_enable_replans", Parameter.Type.BOOL_ARRAY)

        self.replan_service = str(self.get_parameter("replan_service").value)
        self.goal_mode = (
            str(self.get_parameter("replan_mode").value).strip().lower() == "goal"
        )
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.response_timeout = max(
            0.5, float(self.get_parameter("replan_response_timeout").value)
        )
        self.cloud_topic = str(self.get_parameter("obstacle_cloud_topic").value)
        self.cloud_target_frame = str(self.get_parameter("cloud_target_frame").value).strip() or "map"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.robot_radius = max(0.05, float(self.get_parameter("robot_radius").value))
        self.perception_radius = float(self.get_parameter("obstacle_perception_radius").value)
        self.sparse_distance = max(0.02, float(self.get_parameter("obstacle_sparse_distance").value))
        self.obstacle_decay_sec = max(1.0, float(self.get_parameter("obstacle_decay_sec").value))
        self.ingest_rate = min(10.0, max(0.1, float(self.get_parameter("obstacle_ingest_rate").value)))
        self._ingest_period = 1.0 / self.ingest_rate
        self.cloud_step = max(1, int(self.get_parameter("cloud_step").value))
        self.lookahead_m = max(0.5, float(self.get_parameter("replan_lookahead_m").value))
        self.consecutive_frames = max(1, int(self.get_parameter("replan_consecutive_frames").value))
        self.interval_s = float(self.get_parameter("replan_interval_s").value)
        self.check_rate = max(0.1, float(self.get_parameter("replan_check_rate").value))
        self.replan_debug = bool(self.get_parameter("replan_debug").value)
        self.debug_period = max(
            0.1, float(self.get_parameter("replan_debug_period").value)
        )
        self._last_check_log_sec = -1e9
        self._last_debug_state = None

        self.floor_pngs = [str(v) for v in self.get_parameter("floor_pngs").value]
        self.floor_z_mins = [float(v) for v in self.get_parameter("floor_z_mins").value]
        self.floor_z_maxs = [float(v) for v in self.get_parameter("floor_z_maxs").value]
        self.floor_z_mins_obstacle = [
            float(v) for v in self.get_parameter("floor_obstacle_z_mins").value
        ]
        self.floor_z_maxs_obstacle = [
            float(v) for v in self.get_parameter("floor_obstacle_z_maxs").value
        ]
        self.floor_enables = [bool(v) for v in self.get_parameter("floor_enable_replans").value]
        n = len(self.floor_z_mins)
        if not (n and len(self.floor_z_maxs) == n and len(self.floor_z_mins_obstacle) == n
                and len(self.floor_z_maxs_obstacle) == n and len(self.floor_enables) == n):
            self.floor_z_mins = []
            self.get_logger().error(
                "floor arrays missing/mismatched (floor_pngs/floor_z_mins/...); "
                "replan trigger inactive — check maps.<map_name>.floors in local_replan.yaml"
            )

        self.latest_cloud = None
        self.consecutive_blocked = 0
        self.last_request_sec = -1e9
        self._last_ingest_sec = -1e9    # 最近一次入图时刻（按 obstacle_ingest_rate 节拍）
        self.active_floor = None
        self._stop_evt = threading.Event()
        self._thread = None

        # 全局规划服务客户端（请求 = 起点 + 原终点 + map 帧障碍点云）
        self.replan_cli = self.create_client(ReplanPlan, self.replan_service)
        # goal 降级通道：重发原终点触发对方规划器（无 ReplanPlan 服务的算法）
        self.goal_pub = (
            self.create_publisher(PoseStamped, self.goal_topic, 10)
            if self.goal_mode else None
        )
        self._awaiting_since = None    # goal 模式：发出终点后等待新路径的起始时刻
        self._last_path_sig = None     # 最近一条 /global_path 的签名（判新旧路径是否相同）
        self._sig_at_trigger = None    # 触发重规划那一刻的路径签名

        if self.floor_z_mins:
            self.cloud_sub = self.create_subscription(
                PointCloud2, self.cloud_topic, self.cloud_cb, qos_profile_sensor_data
            )
            enabled = [p for p, e in zip(self.floor_pngs, self.floor_enables) if e]
            self.get_logger().info(
                f"replan trigger ready: cloud={self.cloud_topic}, "
                f"service={self.replan_service}, "
                f"{len(self.floor_z_mins)} floors [{','.join(enabled)}], "
                f"forbid={self.robot_radius:.2f} m, "
                f"lookahead={self.lookahead_m:.1f} m, check={self.check_rate:.0f} Hz, "
                f"ingest={self.ingest_rate:.1f} Hz (memory {self.obstacle_decay_sec:.0f} s), "
                f"trigger={'goal ' + self.goal_topic if self.goal_mode else 'service'}"
                + ("" if cKDTree is not None else " (scipy missing: brute-force nearest)")
            )
        else:
            self.cloud_sub = None

    # ---------------------------------------------------- lifecycle helpers
    def start_collision_thread(self) -> None:
        """启动 20Hz 碰撞检测后台线程（main() 里调用）。"""
        if self._thread is not None or not self.floor_z_mins:
            return
        self._thread = threading.Thread(
            target=self._collision_loop, name="collision_check", daemon=True)
        self._thread.start()

    def stop_collision_thread(self) -> None:
        if self._thread is None:
            return
        self._stop_evt.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _collision_loop(self) -> None:
        period = 1.0 / self.check_rate
        while not self._stop_evt.is_set():
            t0 = time.perf_counter()
            try:
                self._collision_check()
            except Exception as exc:  # noqa: BLE001 - 后台线程不能死
                self.get_logger().error(f"collision check failed: {exc}")
            self._stop_evt.wait(max(0.0, period - (time.perf_counter() - t0)))

    # ---------------------------------------------------- cloud ingestion
    def cloud_cb(self, msg: PointCloud2) -> None:
        self.latest_cloud = msg

    def _debug(self, state, msg) -> None:
        """replan_debug diagnostics: 常规 check 行按 replan_debug_period 节流
        （20 Hz 全打每秒 20 行会刷屏，默认 1 s 一条）；早退原因只在变化时打；
        WARN 级关键日志（触发/成功/失败）不经此处、永远即时打印。"""
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

    # ---------------------------------------------------- collision check
    def _collision_check(self) -> None:
        if self.goal_mode:
            self._goal_watchdog()
        if self.x is None or self.done or not self.points:
            self._debug("idle", "replan check: idle (no odom / no active path / done)")
            return
        if self.latest_cloud is None:
            self._debug(
                "nocloud", f"replan check: no cloud received yet on {self.cloud_topic}"
            )
            return

        # Which floor PNG is the robot on? (re-checked every cycle: crossing a
        # stairwell switches the active band/switch mid-mission)
        floor = self._current_floor()
        if floor is None:
            return
        if floor != self.active_floor:
            self.active_floor = floor
            self.consecutive_blocked = 0
            self.get_logger().info(
                f"[当前楼层] F{floor} ({self.floor_pngs[floor]}): "
                f"enable_replan={self.floor_enables[floor]}, "
                f"避障Z带=[{self.floor_z_mins_obstacle[floor]:.2f}, "
                f"{self.floor_z_maxs_obstacle[floor]:.2f}]"
            )
        if not self.floor_enables[floor]:
            self._debug(
                "disabled",
                f"replan check: floor {floor} ({self.floor_pngs[floor]}) "
                "has enable_replan=false",
            )
            return  # this floor opted out (e.g. stair floor)

        # Refresh the progress projection (3D, reuses the tracker's own logic).
        s = self.update_progress()
        waypoints = self._future_waypoints(s, self.lookahead_m)
        if not waypoints:
            self._debug("nowp", "replan check: no waypoints in the lookahead window")
            return

        pts = self._filter_cloud(
            self.latest_cloud, self.floor_z_mins_obstacle[floor],
            self.floor_z_maxs_obstacle[floor],
        )
        if pts is None:
            self._debug(
                "notf",
                f"replan check: cloud frame "
                f"'{self.latest_cloud.header.frame_id or '?'}' -> "
                f"'{self.cloud_target_frame}' TF unavailable, skipping cycle",
            )
            return
        if not pts:
            self.consecutive_blocked = 0
            self._debug(
                "nopts",
                f"replan check: 0 cloud points pass the filter "
                f"(z=[{self.floor_z_mins_obstacle[floor]:+.2f}, "
                f"{self.floor_z_maxs_obstacle[floor]:+.2f}] rel, "
                f"radius={self.perception_radius:.0f} m) — obstacle not in the cloud?",
            )
            return

        # 持续入图（与 cmu_replan 相同的机制）：不等触发，过滤出的障碍按
        # obstacle_ingest_rate 节拍以 ingest_only 请求直接写入全局地图（只入图
        # 不规划不发布），记忆 obstacle_decay_sec；下面碰撞检测/触发逻辑不变。
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_ingest_sec >= self._ingest_period:
            self._last_ingest_sec = now
            self._ingest_obstacles(pts)

        # Nearest-obstacle query over the local path (KD tree; brute force
        # fallback). Blocked iff any path point's squared distance to the
        # nearest obstacle is below the squared forbid distance.
        forbid_sq = self.robot_radius * self.robot_radius
        if cKDTree is not None:
            tree = cKDTree([(p[0], p[1]) for p in pts])
            dists, _ = tree.query(waypoints, k=1)
            min_d2 = float(min(d * d for d in dists))
        else:
            min_d2 = float("inf")
            for wx, wy in waypoints:
                for px, py, _pz in pts:
                    dx = px - wx
                    dy = py - wy
                    d2 = dx * dx + dy * dy
                    if d2 < min_d2:
                        min_d2 = d2
        blocked = min_d2 < forbid_sq

        if blocked:
            self.consecutive_blocked += 1
        else:
            self.consecutive_blocked = 0

        self._debug(
            "check",
            f"replan check: floor {floor} ({self.floor_pngs[floor]}), "
            f"wp={len(waypoints)}, pts={len(pts)}, "
            f"min_d={math.sqrt(min_d2):.2f} m (trigger < {self.robot_radius:.2f}), "
            f"consec={self.consecutive_blocked}/{self.consecutive_frames}",
        )

        now = self.get_clock().now().nanoseconds / 1e9
        if (
            self.consecutive_blocked >= self.consecutive_frames
            and now - self.last_request_sec >= self.interval_s
        ):
            self.last_request_sec = now
            self.consecutive_blocked = 0
            if self.goal_mode:
                self._awaiting_since = now
                self._sig_at_trigger = self._last_path_sig
                self._publish_goal(floor)
            else:
                self._request_replan(floor, pts)

    # ---------------------------------------------------- obstacle ingest
    # 持续入图（与 cmu_replan 相同）：障碍不等触发——过滤降采样后的点云以
    # ingest_only 请求写入全局地图（只入图不规划不发布），记忆时长随请求传
    # obstacle_decay_sec（短记忆，障碍挪走后很快过期）。触发重规划时全局端
    # 已有这份累积记忆，新路径直接绕开提前入图的障碍。
    def _ingest_obstacles(self, pts) -> None:
        if not self.replan_cli.service_is_ready():
            return  # 服务未就绪：跳过本拍，下个节拍重试

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
        req.decay_sec = self.obstacle_decay_sec
        future = self.replan_cli.call_async(req)
        future.add_done_callback(self._on_ingest_response)

    def _on_ingest_response(self, future) -> None:
        try:
            resp = future.result()
        except Exception:  # noqa: BLE001 - 入图失败下个节拍自然重试
            return
        if not resp.success:
            self.get_logger().warn("[障碍入图] 服务端拒绝 ingest-only 请求")

    # ---------------------------------------------------- replan service call
    def _request_replan(self, floor, pts) -> None:
        """构建带动态障碍物的请求并调用全局规划服务：当前位置为起点、原终点
        为终点、过滤+降采样后的 map 帧障碍点云随请求原子送达。"""
        if not self.replan_cli.service_is_ready():
            self.get_logger().warn(
                f"[触发重规划] 失败: 重规划服务 {self.replan_service} 未就绪, "
                "本次请求丢弃 (间隔后自动重试)"
            )
            return

        req = ReplanPlan.Request()
        stamp = self.get_clock().now().to_msg()

        req.start.header.frame_id = self.cloud_target_frame
        req.start.header.stamp = stamp
        req.start.pose.position.x = self.x
        req.start.pose.position.y = self.y
        req.start.pose.position.z = self.z
        req.start.pose.orientation.z = math.sin(self.yaw / 2.0)
        req.start.pose.orientation.w = math.cos(self.yaw / 2.0)

        gx, gy, gz = self.points[-1]  # 原终点（未被障碍影响的任务目标）
        req.goal.header = req.start.header
        req.goal.pose.position.x = gx
        req.goal.pose.position.y = gy
        req.goal.pose.position.z = gz
        req.goal.pose.orientation.w = 1.0

        req.obstacles = pc2.create_cloud_xyz32(
            req.start.header, [(p[0], p[1], p[2]) for p in pts]
        )

        gfloor = self._floor_of_z(gz)
        self.get_logger().warn(
            f"[触发重规划] 楼层 F{floor} ({self.floor_pngs[floor]}): 前方路径被 "
            f"{len(pts)} 个障碍点挡住; 起点 ({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
            f" F{floor} -> 终点 ({gx:.2f}, {gy:.2f}, {gz:.2f}) "
            f"F{gfloor if gfloor is not None else '?'} -> 调用重规划服务"
        )
        future = self.replan_cli.call_async(req)
        future.add_done_callback(self._on_replan_response)

    def _on_replan_response(self, future) -> None:
        """服务响应：成功 → 新路径替换跟踪路径并恢复行驶；失败 → 停车保持
        （hold），触发器继续按间隔请求，障碍过期/挪走后自动恢复。"""
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001 - 服务调用本身失败
            self.get_logger().error(f"replan service call failed: {exc}")
            return
        if resp.success and len(resp.path.poses) >= 2:
            # 成功详情由全局端 [规划情况] 行（点数/cost/耗时/起终点楼层）+
            # 跟踪端 [规划情况] 新路径 行报告，这里不再重复打一条。
            self._ingest_path_msg(resp.path)
        else:
            self.hold = True
            self.get_logger().warn(
                f"[重规划情况] 失败: 各级间距均无路 -> 停车保持, "
                f"每 {self.interval_s:.0f} s 重试 (障碍过期/挪走后自动恢复)"
            )

    def _ingest_path_msg(self, msg) -> None:
        """用一条新路径替换跟踪状态（服务响应与 /global_path 订阅共用）。"""
        pts = [(p.pose.position.x, p.pose.position.y, p.pose.position.z)
               for p in msg.poses]
        if len(pts) < 2:
            return
        self.hold = False
        self.points = pts
        self.cum = [0.0]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            dz = pts[i][2] - pts[i - 1][2]
            self.cum.append(self.cum[-1] + math.sqrt(dx * dx + dy * dy + dz * dz))
        self.progress = 0
        self.done = False

    # ---------------------------------------------------- goal-mode fallback
    # replan_mode=goal：不调服务，把原任务终点重发到 goal_topic 触发对方全局
    # 规划器（它用自己的起点/障碍信息重规划）。服务模式里免费拿到的失败
    # 反馈在这里要自己补：发出终点后 replan_response_timeout 内没有新路径，
    # 或新路径与旧路径完全相同（盲重规划的典型结果）→ 都按 service 失败
    # 处理：停车保持、按 replan_interval_s 重试。持续入图不受影响——服务
    # 在就继续入图（对自家 PRM，goal 触发的重规划仍会绕开记忆中的障碍），
    # 服务不在则自动静默跳过。
    def path_cb(self, msg: Path) -> None:
        """覆盖基类：goal 模式下顺带记录路径签名，判定重发终点是否带来了
        「真的不一样」的新路径。service 模式行为与基类完全一致。"""
        super().path_cb(msg)
        if not self.goal_mode:
            return
        if len(msg.poses) < 2:
            # 空/退化路径：基类已按「全局规划失败」置 hold，结束本次等待
            self._awaiting_since = None
            return
        self._last_path_sig = self._path_signature(msg)
        if self._awaiting_since is not None:
            self._awaiting_since = None
            if self._last_path_sig == self._sig_at_trigger:
                self.hold = True   # 原路重算：对方不知道障碍，跟着走只会再撞
                self.get_logger().warn(
                    "[重规划情况] 失败: 新路径与原路径相同 (对方规划器未感知障碍?) "
                    "-> 停车保持, 间隔后重试"
                )

    @staticmethod
    def _path_signature(msg) -> tuple:
        return tuple(
            (round(p.pose.position.x, 3), round(p.pose.position.y, 3),
             round(p.pose.position.z, 3))
            for p in msg.poses
        )

    def _goal_watchdog(self) -> None:
        """goal 模式看门狗：发出终点后超时没有新路径 → 按规划失败处理。"""
        if self._awaiting_since is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._awaiting_since <= self.response_timeout:
            return
        self._awaiting_since = None
        self.hold = True
        self.get_logger().warn(
            f"[重规划情况] 失败: 重发终点后 {self.response_timeout:.1f} s 内未收到"
            f"新路径 -> 停车保持, 间隔后重试"
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
        self.get_logger().warn(
            f"[触发重规划] 楼层 F{floor} ({self.floor_pngs[floor]}): 前方路径被挡 "
            f"-> [发布终点] 原终点 ({gx:.2f}, {gy:.2f}, {gz:.2f}) "
            f"重发到 {self.goal_topic}"
        )

    # ---------------------------------------------------- floor helpers
    def _current_floor(self):
        """Floor whose [z_min, z_max] contains the robot's Z; first match wins
        (same rule as the global planner's floorFromZ), and a Z in a gap snaps
        to the nearest band edge."""
        return self._floor_of_z(self.z)

    def _floor_of_z(self, z):
        """同 _current_floor 的带判定，但作用于任意 Z（如原终点所在楼层）。"""
        best, best_d = None, float("inf")
        for i, (lo, hi) in enumerate(zip(self.floor_z_mins, self.floor_z_maxs)):
            if lo <= z <= hi:
                return i
            d = (lo - z) if z < lo else (z - hi)
            if d < best_d:
                best_d, best = d, i
        return best

    def _future_waypoints(self, s, lookahead):
        """Waypoints from arc position ``s`` forward up to ``lookahead`` metres,
        restricted to the robot's current floor height so an obstacle on another
        floor (or across a stair gap) never triggers a replan. Returns [(x, y)]."""
        out = []
        limit = s + lookahead
        for i in range(self.progress, len(self.points)):
            if self.cum[i] > limit:
                break
            w = self.points[i]
            if abs(w[2] - self.z) <= self.z_tol:
                out.append((w[0], w[1]))
        if not out and self.progress < len(self.points):
            out.append((self.points[self.progress][0], self.points[self.progress][1]))
        return out

    def _cloud_transform(self, cloud):
        """Rotation+translation mapping cloud points into cloud_target_frame.

        Returns () when no transform is needed (cloud already in the target
        frame), a 12-tuple (r00..r22, tx, ty, tz) otherwise, or None when TF is
        unavailable (caller should skip this cycle rather than compare points
        across frames)."""
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
        """PointCloud2 -> [(x, y, z)] in cloud_target_frame (sensor-frame clouds
        are TF-transformed first). 采样跳步 + 高度带（相对机器人，排除地面/
        天花板）+ 感知半径过滤，然后栅格化降采样（哈希表记录已占据网格）。
        Z 带检查后保留，供服务请求里的障碍点云使用。"""
        xf = self._cloud_transform(cloud)
        if xf is None:
            return None
        pts = []
        radius_sq = self.perception_radius * self.perception_radius
        inv_sparse = 1.0 / self.sparse_distance
        z_low = self.z + z_min
        z_high = self.z + z_max
        seen = set()
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
        return pts


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalReplan()
    node.start_collision_thread()
    try:
        rclpy.spin(node)
    finally:
        node.stop_collision_thread()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
