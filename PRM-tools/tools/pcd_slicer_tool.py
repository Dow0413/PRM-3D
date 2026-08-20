#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import yaml
import argparse
import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class PCDSlicerApp:
    def __init__(self, config_path, config_data):
        self.config_path = config_path
        self.config_data = config_data
        
        # 1. 从配置提取参数
        pcd_path = self.config_data['global']['pcd_path']
        self.res = float(self.config_data['global']['resolution'])
        self.voxel_size = float(self.config_data['global'].get('voxel_size', 0.05))
        self.max_2d_win_size = int(self.config_data['global'].get('max_2d_win_size', 800))
        self.floors_config = self.config_data.get('floors', [])

        if not self.floors_config:
            raise ValueError("配置文件中未找到 floors 列表！")

        print(f"正在加载点云: {pcd_path} ...")
        raw_pcd = o3d.io.read_point_cloud(pcd_path)
        raw_points = np.asarray(raw_pcd.points, dtype=np.float32)
        if len(raw_points) == 0:
            raise ValueError("点云为空！")

        # ----------------------------------------------------
        # 自动计算边界，并回写至全局配置内存
        # ----------------------------------------------------
        self.min_x = float(np.min(raw_points[:, 0]))
        self.max_x = float(np.max(raw_points[:, 0]))
        self.min_y = float(np.min(raw_points[:, 1]))
        self.max_y = float(np.max(raw_points[:, 1]))
        
        # 保存原点到 config 内存 (转换为原生 float 避免 yaml 序列化乱码)
        self.config_data['global']['origin_x'] = float(self.min_x)
        self.config_data['global']['origin_y'] = float(self.min_y)
        
        print(f"[原点自动计算] MAP_ORIGIN_X: {self.min_x:.6f}, MAP_ORIGIN_Y: {self.min_y:.6f}")

        self.W_grid = max(1, int(np.floor((self.max_x - self.min_x) / self.res)) + 1)
        self.H_grid = max(1, int(np.floor((self.max_y - self.min_y) / self.res)) + 1)

        # 2. 降采样用于 3D 渲染 (保证交互流畅)
        legacy_pcd = raw_pcd.voxel_down_sample(voxel_size=self.voxel_size) 
        self.points = np.asarray(legacy_pcd.points, dtype=np.float32)

        # 预计算所有映射矩阵
        print("[优化器] 正在构建高速空间映射表，准备起飞...")
        self.z_all = self.points[:, 2].copy()  
        
        u_floats = (self.points[:, 0] - self.min_x) / self.res
        v_floats = (self.points[:, 1] - self.min_y) / self.res
        self.u_all = np.clip(np.floor(u_floats).astype(np.int64), 0, self.W_grid - 1)
        v_all_int = np.clip(np.floor(v_floats).astype(np.int64), 0, self.H_grid - 1)
        self.v_inv_all = self.H_grid - 1 - v_all_int

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
        
        self.dirty_z = True   
        self.dirty_2d = True
        
        # OpenCV 2D 绘制参数
        self.render_scale = 1.0  
        self.base_map = np.zeros((self.H_grid, self.W_grid), dtype=np.uint8)
        
        # 标注与流水线状态
        self.polygons = []        
        self.current_poly = []    
        self.current_label = 1    
        self.hover_pt = None
        self.current_floor_idx = 0  

        # 4. 初始化 GUI
        self.app = gui.Application.instance
        self.app.initialize()

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

        # 2D 绘制窗口
        cv2.namedWindow("2D Map Editor", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("2D Map Editor", self.on_mouse_cv)

    def on_layout_3d(self, layout_context):
        r = self.win3d.content_rect
        self.widget3d.frame = r
        pref = self.panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.panel.frame = gui.Rect(r.x, r.y, pref.width, pref.height)

    def on_key_3d(self, event):
        if event.type == gui.KeyEvent.DOWN:
            if event.key == gui.KeyName.W:
                self.adjust_z_value(1)
                return True
            elif event.key == gui.KeyName.S:
                self.adjust_z_value(-1)
                return True
            # 【新增】支持在 3D 窗口按 Q 或 ESC 退出
            elif event.key == gui.KeyName.Q or event.key == gui.KeyName.ESCAPE:
                gui.Application.instance.quit()
                return True
        return False

    def adjust_z_value(self, direction):
        delta = self.z_step * direction
        
        if self.radio.selected_index == 0:
            new_z = self.z_min + delta
            if new_z <= self.z_max: self.z_min = new_z
        else:
            new_z = self.z_max + delta
            if new_z >= self.z_min: self.z_max = new_z

        self.lbl_zmin.text = f"z_min: {self.z_min:.3f} m"
        self.lbl_zmax.text = f"z_max: {self.z_max:.3f} m"
        
        self.dirty_z = True

    def on_tick_loop(self):
        key = cv2.waitKeyEx(5)
        if key != -1:
            if key in (ord('w'), ord('W')): self.adjust_z_value(1)
            elif key in (ord('s'), ord('S')): self.adjust_z_value(-1)
            elif key in (ord('q'), ord('Q'), 27):
                cv2.destroyAllWindows()
                gui.Application.instance.quit()
                return False
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
            self._fast_update_z_slice()
            self.dirty_z = False
            self.dirty_2d = True
            needs_3d_redraw = True

        if self.dirty_2d:
            self.render_2d_map()
            self.dirty_2d = False

        if needs_3d_redraw:
            self.win3d.post_redraw()

        return True 

    def _fast_update_z_slice(self):
        mask = (self.z_all >= self.z_min) & (self.z_all <= self.z_max)
        
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

        self.base_map_buffer.fill(0)
        if np.any(mask):
            u_sel = self.u_all[mask]
            v_sel = self.v_inv_all[mask]
            self.base_map_buffer[v_sel, u_sel] = 255
            
        self.base_map = self.base_map_buffer

    def on_mouse_cv(self, event, x, y, flags, param):
        u = int(x / self.render_scale)
        v = int(y / self.render_scale)
        
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
        
        H, W = disp_img.shape[:2]
        self.render_scale = min(self.max_2d_win_size / float(W), self.max_2d_win_size / float(H))
        self.render_scale = min(self.render_scale, 4.0) 
        
        new_W = max(1, int(W * self.render_scale))
        new_H = max(1, int(H * self.render_scale))
        
        disp_large = cv2.resize(disp_img, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        
        progress = min(self.current_floor_idx + 1, len(self.floors_config))
        mode_str = "Traversable (Green -> White Mask)" if self.current_label == 1 else "Blocked (Red -> Black Mask)"
        cv2.putText(disp_large, f"[Map {progress}/{len(self.floors_config)}] Mode: {mode_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp_large, "1: Traversable | 2: Blocked | c: Clear All Polygons", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(disp_large, "LMB: Draw | f: Finish Poly | Bksp: Undo | x: Del Poly | p: Save", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("2D Map Editor", disp_large)

    def save_map(self):
        if self.current_floor_idx >= len(self.floors_config):
            print("\n[提示] 所有地图配置已全部完成并保存。若需重新编辑，请退出重启。")
            return

        config = self.floors_config[self.current_floor_idx]
        png_path = config.get("map_png", "")
        yaml_path = config.get("map_yaml", "")

        final_map = np.zeros_like(self.base_map)
        for poly in self.polygons:
            pts = np.array(poly["points"], np.int32)
            color = 255 if poly["label"] == 1 else 0
            cv2.fillPoly(final_map, [pts], color)
        
        print("\n" + "="*50)
        print(f"[正在保存] 地图 {self.current_floor_idx + 1}/{len(self.floors_config)}: {config.get('name', '')}")
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

        # ----------------------------------------------------
        # 核心：将产生的 Z 轴边界写回到 yaml 配置文件中
        # ----------------------------------------------------
        self.config_data['floors'][self.current_floor_idx]['z_min'] = float(self.z_min)
        self.config_data['floors'][self.current_floor_idx]['z_max'] = float(self.z_max)
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(self.config_data, f, sort_keys=False, default_flow_style=False)
            print(f"  - 回写配置: 已更新 {self.config_path} 中的 z_min 和 z_max")
        except Exception as e:
            print(f"  [警告] 无法回写配置文件: {e}")
            
        print("="*50 + "\n")

        # 流水线推进与状态清理
        if self.current_floor_idx < len(self.floors_config) - 1:
            self.current_floor_idx += 1
            self.polygons = []
            self.current_poly = []
            self.dirty_2d = True
            print(f"[提示] 已自动清理 2D 绘制状态，准备进行 Map {self.current_floor_idx + 1} 的绘制。")
        else:
            self.current_floor_idx += 1
            print("[完毕] 全部楼层掩膜已绘制并保存完毕！")


def main():
    parser = argparse.ArgumentParser(description="PCD 切片与流水线标注工具")
    parser.add_argument("--config", type=str, default="map_pipeline.yaml", help="全局 YAML 配置文件路径")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"配置文件不存在: {args.config}\n请先在当前目录下创建 map_pipeline.yaml 文件。")

    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)

    pcd_path = config_data.get('global', {}).get('pcd_path', '')
    if not pcd_path or not os.path.exists(pcd_path):
        sys.exit(f"点云文件不存在或配置有误: {pcd_path}")
        
    app = PCDSlicerApp(args.config, config_data)
    gui.Application.instance.run()

if __name__ == "__main__":
    main()