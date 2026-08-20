#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# ==========================================
# 1. 核心配置参数
# ==========================================
PCD_PATH = "/home/dow/maps/zjut.pcd"

# 流水线保存配置
FLOORS_CONFIG = [
    {
        "index": 0,
        "SAVE_PNG_PATH": "/home/dow/DOW/PRM-3D/expMap/zjut/zjut_1.png",
        "SAVE_YAML_PATH": "/home/dow/DOW/PRM-3D/expMap/zjut/zjut.yaml",
    },
    {
        "index": 1,
        "SAVE_PNG_PATH": "/home/dow/DOW/PRM-3D/expMap/zjut/zjut_2.png",
        "SAVE_YAML_PATH": "",  # 留空则不保存 yaml
    },
    {
        "index": 2,
        "SAVE_PNG_PATH": "/home/dow/DOW/PRM-3D/expMap/zjut/zjut_3.png",
        "SAVE_YAML_PATH": "",  
    }
]

# 地图分辨率 (固定不变)
RES = 0.05

# 3D 渲染降采样体素大小 (调大可提升超大地图的交互帧率，不影响最终PNG地图精度)
VOXEL_SIZE = 0.05

# 2D 窗口最大显示尺寸限制 (像素)
MAX_2D_WIN_SIZE = 800


class PCDSlicerApp:
    def __init__(self, pcd_path):
        # 1. 加载原始未降采样的点云
        print(f"正在加载点云: {pcd_path} ...")
        raw_pcd = o3d.io.read_point_cloud(pcd_path)
        raw_points = np.asarray(raw_pcd.points, dtype=np.float32)
        if len(raw_points) == 0:
            raise ValueError("点云为空！")

        # ----------------------------------------------------
        # 基于未降采样的原始点云，获取绝对精准的物理原点和边界
        # ----------------------------------------------------
        self.min_x = float(np.min(raw_points[:, 0]))
        self.max_x = float(np.max(raw_points[:, 0]))
        self.min_y = float(np.min(raw_points[:, 1]))
        self.max_y = float(np.max(raw_points[:, 1]))
        
        print(f"[原点自动计算] MAP_ORIGIN_X: {self.min_x:.6f}, MAP_ORIGIN_Y: {self.min_y:.6f}")

        self.res = RES
        self.W_grid = max(1, int(np.floor((self.max_x - self.min_x) / self.res)) + 1)
        self.H_grid = max(1, int(np.floor((self.max_y - self.min_y) / self.res)) + 1)

        # 2. 降采样用于 3D 渲染 (保证交互流畅)
        legacy_pcd = raw_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE) 
        self.points = np.asarray(legacy_pcd.points, dtype=np.float32)

        # ----------------------------------------------------
        # 【极致优化核心】预计算所有映射矩阵，免去拖拽时的浮点除法运算
        # ----------------------------------------------------
        print("[优化器] 正在构建高速空间映射表，准备起飞...")
        self.z_all = self.points[:, 2].copy()  # 提取独立的连续内存用于 Z 值比较
        
        # 预计算所有点投影到 2D 上的 U, V 索引
        u_floats = (self.points[:, 0] - self.min_x) / self.res
        v_floats = (self.points[:, 1] - self.min_y) / self.res
        self.u_all = np.clip(np.floor(u_floats).astype(np.int64), 0, self.W_grid - 1)
        v_all_int = np.clip(np.floor(v_floats).astype(np.int64), 0, self.H_grid - 1)
        
        # 翻转 Y 轴适应图像坐标，存起来直接查表使用
        self.v_inv_all = self.H_grid - 1 - v_all_int

        # 初始化高速预分配内存
        self.base_map_buffer = np.zeros((self.H_grid, self.W_grid), dtype=np.uint8)

        # 构建 Tensor 点云
        self.device = o3c.Device("CPU:0")
        self.t_pcd = o3d.t.geometry.PointCloud(self.device)
        self.t_pcd.point.positions = o3c.Tensor(self.points, o3c.float32, self.device)
        
        init_colors = np.full((len(self.points), 3), [0.3, 0.3, 0.3], dtype=np.float32)
        self.t_pcd.point.colors = o3c.Tensor(init_colors, o3c.float32, self.device)

        # 3. 初始化控制状态
        self.z_min = float(np.min(self.z_all))
        self.z_max = float(np.max(self.z_all))
        self.z_step = 0.05
        
        self.dirty_z = True   # 【同步锁标记】解绑按键和渲染的耦合，拒绝卡顿
        self.dirty_2d = True
        
        # OpenCV 2D 绘制参数
        self.render_scale = 1.0  
        self.base_map = np.zeros((self.H_grid, self.W_grid), dtype=np.uint8)
        
        # 多边形标注与流水线状态
        self.polygons = []        
        self.current_poly = []    
        self.current_label = 1    
        self.hover_pt = None
        self.current_floor_idx = 0  

        # 4. 初始化 Open3D GUI
        self.app = gui.Application.instance
        self.app.initialize()

        # ========================================
        # 创建 3D 窗口与控制面板
        # ========================================
        self.win3d = self.app.create_window("3D 点云切片视图", 1024, 768)
        
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.win3d.renderer)
        self.win3d.add_child(self.widget3d)

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 4.0 
        
        self.widget3d.scene.add_geometry("pcd", self.t_pcd, self.mat)
        
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(self.points))
        self.widget3d.setup_camera(60.0, bbox, bbox.get_center())

        em = self.win3d.theme.font_size
        self.panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        
        lbl_cam = gui.Label("【3D 视角操作】")
        lbl_cam.text_color = gui.Color(1.0, 0.7, 0.0)
        self.panel.add_child(lbl_cam)
        self.panel.add_child(gui.Label("旋转: 左键拖动 | 平移: 右键拖动"))
        self.panel.add_child(gui.Label(" ")) 
        
        lbl_ctrl = gui.Label("【Z 轴截断控制】(W:增加 | S:减少)")
        lbl_ctrl.text_color = gui.Color(0.0, 0.8, 1.0) 
        self.panel.add_child(lbl_ctrl)

        self.radio = gui.RadioButton(gui.RadioButton.VERT)
        self.radio.set_items(["当前控制: z_min (底层/红色)", "当前控制: z_max (顶层/黄色)"])
        self.panel.add_child(self.radio)

        self.lbl_zmin = gui.Label(f"z_min: {self.z_min:.3f} m")
        self.lbl_zmax = gui.Label(f"z_max: {self.z_max:.3f} m")
        self.panel.add_child(self.lbl_zmin)
        self.panel.add_child(self.lbl_zmax)

        self.win3d.add_child(self.panel)

        self.win3d.set_on_layout(self.on_layout_3d)
        self.win3d.set_on_key(self.on_key_3d)
        self.win3d.set_on_tick_event(self.on_tick_loop)

        # ========================================
        # 创建 OpenCV 2D 绘制窗口
        # ========================================
        cv2.namedWindow("2D Map Editor", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("2D Map Editor", self.on_mouse_cv)

    def on_layout_3d(self, layout_context):
        r = self.win3d.content_rect
        self.widget3d.frame = r
        pref = self.panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.panel.frame = gui.Rect(r.x, r.y, pref.width, pref.height)

    # ------------------ 真正的实时解耦响应系统 ------------------
    def on_key_3d(self, event):
        if event.type == gui.KeyEvent.DOWN:
            if event.key == gui.KeyName.W:
                self.adjust_z_value(1)
                return True
            elif event.key == gui.KeyName.S:
                self.adjust_z_value(-1)
                return True
        return False

    def adjust_z_value(self, direction):
        """只更新数值并标记更新状态，绝不在按键回调中处理耗时渲染"""
        delta = self.z_step * direction
        
        if self.radio.selected_index == 0:
            new_z = self.z_min + delta
            if new_z <= self.z_max: self.z_min = new_z
        else:
            new_z = self.z_max + delta
            if new_z >= self.z_min: self.z_max = new_z

        self.lbl_zmin.text = f"z_min: {self.z_min:.3f} m"
        self.lbl_zmax.text = f"z_max: {self.z_max:.3f} m"
        
        # 通知异步线程：数据已改，快去渲染
        self.dirty_z = True

    def on_tick_loop(self):
        """主循环心跳：同时处理 OpenCV 的刷新与按键排队"""
        # OpenCV 等待 5 毫秒进行高速轮询
        key = cv2.waitKeyEx(5)
        if key != -1:
            if key in (ord('w'), ord('W')): self.adjust_z_value(1)
            elif key in (ord('s'), ord('S')): self.adjust_z_value(-1)
            elif key == ord('1'):
                self.current_label = 1
                self.dirty_2d = True
            elif key == ord('2'):
                self.current_label = 2
                self.dirty_2d = True
            elif key in (8, 127):  
                if self.current_poly:
                    self.current_poly.pop()
                    self.dirty_2d = True
            elif key in (ord('f'), ord('F')):  
                if len(self.current_poly) >= 3:
                    self.polygons.append({
                        "label": self.current_label,
                        "points": self.current_poly.copy()
                    })
                self.current_poly = []
                self.dirty_2d = True
            elif key in (ord('x'), ord('X')):  
                if self.polygons:
                    self.polygons.pop()
                    self.dirty_2d = True
            elif key in (ord('c'), ord('C')):  
                self.polygons = []
                self.current_poly = []
                print("[操作] 已手动清空当前地图的绘制记录。")
                self.dirty_2d = True
            elif key in (ord('p'), ord('P')):  
                self.save_map()

        needs_3d_redraw = False

        if self.dirty_z:
            # 高速渲染：1毫秒内完成颜色与掩膜提取
            self._fast_update_z_slice()
            self.dirty_z = False
            self.dirty_2d = True
            needs_3d_redraw = True

        if self.dirty_2d:
            self.render_2d_map()
            self.dirty_2d = False

        if needs_3d_redraw:
            self.win3d.post_redraw()

        return True # 保持高频轮询以吸收键盘输入

    def _fast_update_z_slice(self):
        """【性能黑魔法】全查表、零分配的高速计算层"""
        # 1. 找出区间遮罩 (1D numpy array operation，极其快)
        mask = (self.z_all >= self.z_min) & (self.z_all <= self.z_max)
        
        # 2. 更新 3D 点云颜色 (避免繁重运算)
        colors = np.full((len(self.points), 3), [0.25, 0.25, 0.25], dtype=np.float32)
        if np.any(mask):
            z_in_range = self.z_all[mask]
            if self.z_max > self.z_min:
                ratio = (z_in_range - self.z_min) / (self.z_max - self.z_min)
            else:
                ratio = np.zeros(len(z_in_range), dtype=np.float32)
            colors[mask, 0] = 1.0          
            colors[mask, 1] = ratio        
            colors[mask, 2] = 0.0          
            
        self.t_pcd.point.colors = o3c.Tensor(colors, o3c.float32, self.device)
        self.widget3d.scene.scene.update_geometry("pcd", self.t_pcd, rendering.Scene.UPDATE_COLORS_FLAG)

        # 3. 更新 2D 底图映射 (利用预分配缓冲与预计算坐标，摒弃所有运算)
        self.base_map_buffer.fill(0)
        if np.any(mask):
            u_sel = self.u_all[mask]
            v_sel = self.v_inv_all[mask]
            self.base_map_buffer[v_sel, u_sel] = 255
            
        self.base_map = self.base_map_buffer

    # ------------------ 2D OpenCV 交互与导出 ------------------
    def on_mouse_cv(self, event, x, y, flags, param):
        # 逆向映射屏幕坐标回到真实物理栅格坐标
        u = int(x / self.render_scale)
        v = int(y / self.render_scale)
        
        # 边界保护
        if not (0 <= u < self.W_grid and 0 <= v < self.H_grid):
            return
            
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_pt = (u, v)
            self.dirty_2d = True
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.current_poly.append((u, v))
            self.dirty_2d = True

    def render_2d_map(self):
        disp_base = cv2.cvtColor(self.base_map, cv2.COLOR_GRAY2BGR)
        overlay = disp_base.copy()
        
        for poly in self.polygons:
            pts = np.array(poly["points"], np.int32)
            color = (95, 170, 40) if poly["label"] == 1 else (50, 60, 210)
            cv2.fillPoly(overlay, [pts], color)
            
        disp_img = cv2.addWeighted(overlay, 0.45, disp_base, 0.55, 0)
        
        if self.current_poly:
            color = (95, 170, 40) if self.current_label == 1 else (50, 60, 210)
            pts = np.array(self.current_poly, np.int32)
            cv2.polylines(disp_img, [pts], False, color, 1, cv2.LINE_AA)
            for pt in self.current_poly:
                cv2.circle(disp_img, pt, 1, color, -1)
            
            if self.hover_pt is not None:
                cv2.line(disp_img, self.current_poly[-1], self.hover_pt, color, 1, cv2.LINE_AA)
        
        # 动态计算缩放比例，限制最大显示窗口大小
        H, W = disp_img.shape[:2]
        self.render_scale = min(MAX_2D_WIN_SIZE / float(W), MAX_2D_WIN_SIZE / float(H))
        self.render_scale = min(self.render_scale, 4.0) 
        
        new_W = max(1, int(W * self.render_scale))
        new_H = max(1, int(H * self.render_scale))
        
        disp_large = cv2.resize(disp_img, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        
        progress = min(self.current_floor_idx + 1, len(FLOORS_CONFIG))
        mode_str = "Traversable (Green -> White Mask)" if self.current_label == 1 else "Blocked (Red -> Black Mask)"
        cv2.putText(disp_large, f"[Map {progress}/{len(FLOORS_CONFIG)}] Mode: {mode_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp_large, "1: Traversable | 2: Blocked | c: Clear All Polygons", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(disp_large, "LMB: Draw | f: Finish Poly | Bksp: Undo | x: Del Poly | p: Save", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("2D Map Editor", disp_large)

    def save_map(self):
        if self.current_floor_idx >= len(FLOORS_CONFIG):
            print("\n[提示] 所有地图配置已全部完成并保存。若需重新编辑，请重置进度或退出重启。")
            return

        config = FLOORS_CONFIG[self.current_floor_idx]
        png_path = config.get("SAVE_PNG_PATH", "")
        yaml_path = config.get("SAVE_YAML_PATH", "")

        final_map = np.zeros_like(self.base_map)
        for poly in self.polygons:
            pts = np.array(poly["points"], np.int32)
            color = 255 if poly["label"] == 1 else 0
            cv2.fillPoly(final_map, [pts], color)
        
        print("\n" + "="*50)
        print(f"[正在保存] 地图 {self.current_floor_idx + 1}/{len(FLOORS_CONFIG)}")
        print(f"  - 截断高度: z_min = {self.z_min:.3f} m, z_max = {self.z_max:.3f} m")
        
        if png_path:
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            cv2.imwrite(png_path, final_map)
            print(f"  - 导出 PNG: {png_path}")
            
        if yaml_path:
            img_filename = os.path.basename(png_path) if png_path else "unknown.png"
            yaml_content = f"""image: {img_filename}
resolution: {self.res:.6f}
origin: [{self.min_x:.6f}, {self.min_y:.6f}, 0.000000]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            with open(yaml_path, "w") as f:
                f.write(yaml_content)
            print(f"  - 导出 YAML: {yaml_path}")
            
        print("="*50 + "\n")

        if self.current_floor_idx < len(FLOORS_CONFIG) - 1:
            self.current_floor_idx += 1
            self.polygons = []
            self.current_poly = []
            self.dirty_2d = True
            print(f"[提示] 已自动清理 2D 绘制状态，准备进行 Map {self.current_floor_idx + 1} 的绘制。")
        else:
            self.current_floor_idx += 1
            print("[完毕] 全部楼层掩膜已绘制并保存完毕！")


def main():
    if not os.path.exists(PCD_PATH):
        sys.exit(f"点云文件不存在: {PCD_PATH}")
        
    app = PCDSlicerApp(PCD_PATH)
    gui.Application.instance.run()

if __name__ == "__main__":
    main()