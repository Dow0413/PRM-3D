#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manual_z_path_tool.py —— 在 2D 栅格地图上手动打点标定 Z 值，生成平滑 3D 路径

背景
----
PRM 全局路径的 XY 是准的，Z 前期先不追算法：本工具把「人工经验」直接
编码成路径——在地图上按顺序点击途经点，每点手动输入一个 Z，工具负责
样条平滑 + 加密，输出 Nx3 (x, y, z) 的 .npy，供下游（控制器 / 仿真 /
局部规划）做离线 3D 路径测试。

坐标约定（与 global_prm.yaml / process_multi_floor_elevations.py 一致）
----
  x = origin_x + u * resolution          u 为像素列 col
  y = origin_y + v * resolution          v 为像素行 row（flip_y=true 时取 H-1-row）
  即 ROS 地图约定：origin 在左下角像素，OpenCV 行号向下增长，需要翻转。

用法
----
  python3 tools/manual_z_path_tool.py                          # 用 CONFIG 里的默认地图
  python3 tools/manual_z_path_tool.py expMap/35/356_1.png      # 指定地图
  python3 tools/manual_z_path_tool.py --out my_path.npy        # 指定输出文件名

  若地图旁边有同名 .yaml / .yml（如 356_1.yaml）且含 resolution / origin，
  会自动读取并覆盖 CONFIG 的默认值（终端会打印最终采用的标定值）。

操作
----
  左键    打一个路径点 → 程序暂停，在终端输入该点 Z 值(米)
            回车      = 沿用上一点的 Z（平地连续点很省事）
            b         = 取消刚点的这个点
            1,5 / 1.5 逗号小数自动替换为小数点
  右键    撤销上一个已确认的点（连按可退多个）
  s       结束打点 → 平滑插值 → 保存 .npy（最少 2 个点）
  q / ESC 放弃退出（不保存）

输出
----
  <out>.npy              Nx3 float64 (x, y, z)，弧长间距约 DENSE_SPACING_M
  <out>.keypoints.json   关键点原样记录（像素+物理+Z、地图标定参数），
                         便于事后检查 / 重跑平滑参数
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

# ==========================================
# 1. 配置
# ==========================================
# 地图标定（若地图旁有同名 yaml 则被自动覆盖）
MAP_CONFIG = {
    "map_path": "/home/dow/DOW/PRM-3D/expMap/35/356_1.png",
    "resolution": 0.05,
    "origin_x": -15.471000,
    "origin_y": -7.330000,
    "flip_y": True,   # ROS 约定: origin 在左下角; OpenCV 行向下增长需翻转
}

DISPLAY_SCALE = 3        # 显示放大倍数（557x220 的图太小，点不准）
SMOOTH_TOL_M = 0.05      # 样条允许的平均偏离（米）; s = 点数 * tol^2
DENSE_SPACING_M = 0.05   # 加密后相邻点弧长间距（米）

# 配色 (BGR)
C_POINT = (0, 0, 255)      # 关键点（红）
C_LINE = (0, 220, 0)       # 关键点折线（绿）
C_SMOOTH = (255, 180, 0)   # 平滑后曲线（青蓝）
C_HINT = (255, 255, 255)   # 提示文字（白）

WIN = "manual_z_path"


# ==========================================
# 2. 坐标映射：像素 -> 物理坐标
# ==========================================
def pixel_to_metric(col, row, cfg, map_h):
    """像素坐标 (列 col, 行 row) -> 物理坐标 (x, y)。

    flip_y=True 时遵循 ROS 地图约定（origin 为左下角像素，行号向下增长，
    需翻转）；flip_y=False 时 origin 即左上角像素。
    """
    x = cfg["origin_x"] + col * cfg["resolution"]
    v = (map_h - 1 - row) if cfg["flip_y"] else row
    y = cfg["origin_y"] + v * cfg["resolution"]
    return x, y


def try_load_map_yaml(map_path, cfg):
    """若地图旁有同名 .yaml/.yml（map_server 风格: resolution + origin: [x,y,yaw]），
    读取并覆盖 cfg 的标定值。返回更新后的 cfg（找不到 / 解析失败则原样返回）。"""
    base, _ = os.path.splitext(map_path)
    for ext in (".yaml", ".yml"):
        p = base + ext
        if not os.path.exists(p):
            continue
        try:
            import yaml
            d = yaml.safe_load(open(p, encoding="utf-8"))
            res = float(d["resolution"])
            ox, oy = float(d["origin"][0]), float(d["origin"][1])
            cfg.update(map_path=map_path, resolution=res, origin_x=ox, origin_y=oy)
            print(f"[标定] 从 {p} 读取: resolution={res}, origin=({ox}, {oy})")
            return cfg
        except Exception as e:
            print(f"[标定] {p} 解析失败({e})，忽略，继续用 CONFIG 默认值")
    cfg["map_path"] = map_path
    return cfg


# ==========================================
# 3. 轨迹平滑插值：稀疏关键点 -> 稠密平滑 3D 路径
# ==========================================
def smooth_path_xyz(keypoints, spacing=DENSE_SPACING_M, tol=SMOOTH_TOL_M):
    """对 Nx3 关键点做样条平滑 + 等弧长加密。

    - 参数化: scipy.interpolate.splprep 按弦长参数化，x/y/z 联合拟合，
      s = 点数 * tol^2（容许约 tol 量级的平均偏离，兼顾贴合与平滑）;
    - 点数 < 4 时自动降为二次/线性样条；样条失败（重复点/共线等）时
      回退为分段线性插值，保证总能出路径;
    - 加密: 目标相邻点弧长 ≈ spacing（米）。
    返回 (dense Nx3 float64, 方法说明字符串)。
    """
    from scipy.interpolate import splev, splprep

    pts = np.asarray(keypoints, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("至少需要 2 个关键点才能生成路径")

    # 去掉相邻过近的点（<1cm 视为重复点，splprep 对重复点敏感）
    keep = [0] + [i for i in range(1, len(pts))
                  if np.linalg.norm(pts[i, :2] - pts[i - 1, :2]) > 0.01]
    pts = pts[keep]
    if len(pts) < 2:
        raise ValueError("去重后不足 2 个关键点（点击位置太近）")

    # 累计弦长，用于估算加密点数
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    total_len = float(np.sum(seg))
    n_dense = max(int(total_len / spacing) + 1, 4 * len(pts))

    k = 3 if len(pts) >= 4 else (2 if len(pts) == 3 else 1)
    s = len(pts) * tol * tol
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=s, k=k)
        xs, ys, zs = splev(np.linspace(0.0, 1.0, n_dense), tck)
        dense = np.column_stack([xs, ys, zs])
        how = f"scipy 样条 (k={k}, s={s:.2f})"
    except Exception as e:  # 样条失败 -> 线性插值兜底
        u = np.concatenate([[0.0], np.cumsum(seg)]) / max(total_len, 1e-9)
        u_new = np.linspace(0.0, 1.0, n_dense)
        dense = np.column_stack([np.interp(u_new, u, pts[:, j]) for j in range(3)])
        how = f"线性插值兜底（样条失败: {e}）"

    return dense, how


def path_report(dense):
    """打印路径统计：总长、Z 范围、最大坡度 |dz/ds| 及其位置。"""
    d = np.diff(dense, axis=0)
    step = np.linalg.norm(d, axis=1)
    slope = np.abs(d[:, 2]) / np.maximum(step, 1e-9)
    i = int(np.argmax(slope))
    arc = np.concatenate([[0.0], np.cumsum(step)])
    print(f"[路径] 总长 {arc[-1]:.2f} m, {len(dense)} 点 (间距~{arc[-1]/max(len(dense)-1,1):.3f} m)")
    print(f"[路径] Z 范围 [{dense[:,2].min():.2f}, {dense[:,2].max():.2f}] m")
    print(f"[路径] 最大坡度 |dz/ds| = {slope[i]:.2f} @ 第{i}点 "
          f"(x={dense[i,0]:.2f}, y={dense[i,1]:.2f}, z={dense[i,2]:.2f})"
          + ("  ⚠ 比楼梯(≈0.7)还陡, 请检查 Z 输入" if slope[i] > 1.0 else ""))


# ==========================================
# 4. 交互打点
# ==========================================
class PathAnnotator:
    """OpenCV 窗口打点 + 终端录入 Z。鼠标回调只记录事件，Z 输入在主循环
    里进行（input() 会阻塞，不能放在回调里）。"""

    def __init__(self, cfg):
        self.cfg = cfg
        img = cv2.imread(cfg["map_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"地图读取失败: {cfg['map_path']}")
        self.map_h, self.map_w = img.shape
        self.base = cv2.resize(img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                               interpolation=cv2.INTER_NEAREST)
        self.base = cv2.cvtColor(self.base, cv2.COLOR_GRAY2BGR)
        self.points = []        # 已确认关键点: dict(u=col, v=row, x, y, z)
        self.pending = None     # 左键点击后等待终端录入 Z 的像素
        self.finished = False
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)

    # ---- 鼠标回调：只改状态，不做任何阻塞操作 ----
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            u = int(np.clip(x / DISPLAY_SCALE, 0, self.map_w - 1))
            v = int(np.clip(y / DISPLAY_SCALE, 0, self.map_h - 1))
            self.pending = (u, v)
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            p = self.points.pop()
            print(f"[撤销] 去掉点 {len(self.points)} (z={p['z']})")

    # ---- 终端录入 Z：回车沿用上一个, b 取消该点 ----
    def ask_z(self, idx, u, v):
        default = self.points[-1]["z"] if self.points else None
        while True:
            tip = f"点 {idx} 像素(u={u},v={v})"
            tip += f" Z[{default:.2f}]回车沿用" if default is not None else " Z"
            raw = input(f"{tip} : ").strip().replace("，", ".").replace(",", ".")
            if raw.lower() == "b":
                return None
            if raw == "" and default is not None:
                return default
            try:
                return float(raw)
            except ValueError:
                print("  解析失败，请输入数字（如 -1.25 / 0.55），b=取消")

    # ---- 绘制 ----
    def draw(self, smooth_px=None):
        frame = self.base.copy()
        if len(self.points) > 1:  # 关键点折线
            poly = [(p["u"] * DISPLAY_SCALE, p["v"] * DISPLAY_SCALE) for p in self.points]
            cv2.polylines(frame, [np.int32(poly)], False, C_LINE, 2, cv2.LINE_AA)
        for i, p in enumerate(self.points):  # 关键点 + 序号 + Z
            x, y = p["u"] * DISPLAY_SCALE, p["v"] * DISPLAY_SCALE
            cv2.circle(frame, (x, y), 5, C_POINT, -1, cv2.LINE_AA)
            cv2.putText(frame, f"{i}:{p['z']:.2f}", (x + 8, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_POINT, 1, cv2.LINE_AA)
        if smooth_px is not None:  # 平滑后曲线
            cv2.polylines(frame, [np.int32(smooth_px)], False, C_SMOOTH, 2, cv2.LINE_AA)
        if self.pending is not None:  # 待录入状态的提示
            x, y = self.pending[0] * DISPLAY_SCALE, self.pending[1] * DISPLAY_SCALE
            cv2.circle(frame, (x, y), 5, (0, 255, 255), 2, cv2.LINE_AA)
            bar = "waiting Z in terminal..."
            cv2.rectangle(frame, (0, frame.shape[0] - 26), (230, frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, bar, (6, frame.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_HINT, 1)
        else:
            cv2.rectangle(frame, (0, frame.shape[0] - 26), (560, frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, "LMB:point  RMB:undo  s:finish&save  q:quit",
                        (6, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_HINT, 1)
        cv2.imshow(WIN, frame)

    # ---- 主循环 ----
    def run(self):
        print("在窗口中左键打点（每次到终端输入 Z），s 结束保存, q 退出")
        while True:
            self.draw()
            key = cv2.waitKey(30) & 0xFF
            if self.pending is not None:          # 主循环里处理 Z 录入（不阻塞回调）
                u, v = self.pending
                z = self.ask_z(len(self.points) + 1, u, v)
                if z is not None:
                    x, y = pixel_to_metric(u, v, self.cfg, self.map_h)
                    self.points.append(dict(u=u, v=v, x=x, y=y, z=z))
                    print(f"  -> 点 {len(self.points)}: 像素(u={u},v={v}) 物理({x:.3f}, {y:.3f}) z={z}")
                self.pending = None
                continue
            if key == ord("s") and len(self.points) >= 2:
                self.finished = True
                break
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                print("至少 2 个点才能生成路径，继续打点")
        cv2.destroyAllWindows()
        return self.finished


# ==========================================
# 5. 主流程
# ==========================================
def main():
    ap = argparse.ArgumentParser(description="手动打点标定 Z，生成平滑 3D 路径 .npy")
    ap.add_argument("map", nargs="?", default=MAP_CONFIG["map_path"], help="地图 PNG/PGM 路径")
    ap.add_argument("--out", default=None, help="输出 .npy 路径（默认带时间戳）")
    args = ap.parse_args()

    cfg = try_load_map_yaml(args.map, dict(MAP_CONFIG))
    print(f"[标定] map={cfg['map_path']}\n"
          f"        resolution={cfg['resolution']} origin=({cfg['origin_x']}, {cfg['origin_y']}) "
          f"flip_y={cfg['flip_y']}")

    anno = PathAnnotator(cfg)
    if not anno.run():
        print("未保存，退出")
        return

    kp = np.array([[p["x"], p["y"], p["z"]] for p in anno.points], dtype=np.float64)
    dense, how = smooth_path_xyz(kp)
    path_report(dense)
    print(f"[平滑] {how}")

    # 平滑结果回画到窗口确认
    cols = (dense[:, 0] - cfg["origin_x"]) / cfg["resolution"]
    rows = dense[:, 1] - cfg["origin_y"]
    rows = (anno.map_h - 1) - rows if cfg["flip_y"] else rows
    smooth_px = np.column_stack([cols, rows]) * DISPLAY_SCALE
    anno.pending = None
    cv2.namedWindow(WIN)
    anno.draw(smooth_px)
    print("平滑曲线已叠加(青色)，任意键关闭窗口")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存 .npy + 关键点 sidecar
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"manual_path_{time.strftime('%Y%m%d_%H%M%S')}.npy")
    np.save(out, dense)
    sidecar = os.path.splitext(out)[0] + ".keypoints.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(dict(calibration={k: cfg[k] for k in
                                    ("map_path", "resolution", "origin_x", "origin_y", "flip_y")},
                       smooth=dict(method=how, tol=SMOOTH_TOL_M, spacing=DENSE_SPACING_M),
                       keypoints=anno.points), f, ensure_ascii=False, indent=1)
    print(f"[保存] {out}  shape={dense.shape}")
    print(f"[保存] {sidecar}")

    # Z 剖面图（弧长-Z），直观确认有无跳变；无显示环境时跳过
    try:
        import matplotlib.pyplot as plt
        step = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(step)])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(arc, dense[:, 2], "-", color="steelblue", label="smooth z")
        # 关键点的真实弧长位置 = 稠密路径上离它 (x,y) 最近的点
        near = [int(np.argmin(np.hypot(dense[:, 0] - kx, dense[:, 1] - ky)))
                for kx, ky in kp[:, :2]]
        ax.scatter(arc[near], dense[near, 2], c="red", zorder=3, label="keypoints")
        ax.set_xlabel("弧长 (m)"); ax.set_ylabel("z (m)"); ax.legend(); ax.grid(True)
        plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"[剖面图] 跳过 ({e})")


if __name__ == "__main__":
    main()
