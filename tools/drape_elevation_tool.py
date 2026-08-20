#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_navmesh_tool.py —— 批量流水线：纯手动导航网格 (NavMesh) 铺设工具

操作说明：
  - 左键单击 (LMB)：点击空白区域 -> 创建新顶点；点击现有顶点 -> 选中/取消选中。
  - 中键按住 (MMB)：按住现有顶点拖动 -> 实时改变顶点 X,Y 坐标（限制在白区）。
  - c：清空当前的选中状态 (Clear Selection)。
  - w / s：选中了单个顶点时，微调该顶点的 Z 值 (e/x 为大步调)。
  - f (Face)：当刚好选中 3 个顶点时，按下 f 键将它们连成一个三角面。
  - x / delete / backspace：彻底删除选中的节点（会自动解除关联的面）。
  - p 键：保存当前提取的合并高程图与拓扑结构，并自动无缝切换到下一张地图！
  - q 键：退出。
"""

import os
import sys
import threading
import time
import json

import cv2
import numpy as np

# ==========================================
# 1. 批量多楼层配置
# ==========================================
PCD_PATH = "/home/dow/maps/356_process_3.pcd"

# 统一输出前缀 (save_path 会自动加上 .npy, .json, _preview.png)
# read_json: 填入路径则加载配置，留空 "" 则从零开始
FLOORS_CONFIG = [
    {
        "index": 0,
        "map": "/home/dow/DOW/PRM-3D/expMap/35_all/356_1.png",
        "save_path": "/home/dow/DOW/PRM-3D/expMap/35_all/356_1",
        "read_json": ""
    },
    {
        "index": 1,
        "map": "/home/dow/DOW/PRM-3D/expMap/35_all/356_2.png",
        "save_path": "/home/dow/DOW/PRM-3D/expMap/35_all/356_2",
        "read_json": ""
    },
    {
        "index": 2,
        "map": "/home/dow/DOW/PRM-3D/expMap/35_all/356_3.png",
        "save_path": "/home/dow/DOW/PRM-3D/expMap/35_all/356_3",
        "read_json": ""
    },
    {
        "index": 3,
        "map": "/home/dow/DOW/PRM-3D/expMap/35_all/356_4.png",
        "save_path": "/home/dow/DOW/PRM-3D/expMap/35_all/356_4",
        "read_json": ""
    },
    {
        "index": 4,
        "map": "/home/dow/DOW/PRM-3D/expMap/35_all/356_5.png",
        "save_path": "/home/dow/DOW/PRM-3D/expMap/35_all/356_5",
        "read_json": ""
    },
    {
        "index": 5,
        "map": "/home/dow/DOW/PRM-3D/expMap/35_all/356_6.png",
        "save_path": "/home/dow/DOW/PRM-3D/expMap/35_all/356_6",
        "read_json": ""
    }
]

# --- 数据保存配置 ---
SAVE_PNG = False
SAVE_JSON = True

# 全局原点与分辨率参数
RES, OX, OY = 0.05, -15.471, -7.330     

Z_STEP, Z_BIG = 0.02, 0.10   
DISPLAY_SCALE = 3            
VOXEL_SIZE = 0.05

KEY_Z_UP, KEY_Z_DN = ord("w"), ord("s")        
KEY_Z_UP2, KEY_Z_DN2 = ord("e"), ord("x")      

WIN2D = "2D Map - NavMesh Editor"
WIN3D = "3D NavMesh Preview"

# ==========================================
# 2. 核心插值算法 (平面方程)
# ==========================================
def bake_triangles_to_grid(vertices, faces, W, H):
    grid = np.full((W, H), np.nan)
    
    for face in faces:
        p0 = vertices[face[0]]
        p1 = vertices[face[1]]
        p2 = vertices[face[2]]
        
        pts_cv = np.array([
            [p0[0], H - 1 - p0[1]],
            [p1[0], H - 1 - p1[1]],
            [p2[0], H - 1 - p2[1]]
        ], np.int32)
        
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_cv], 1)
        y_idx, x_idx = np.nonzero(mask)
        
        if len(x_idx) == 0: continue
            
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        Nx, Ny, Nz = normal
        
        if abs(Nz) < 1e-6: continue
            
        u_pts = x_idx.astype(float)
        v_pts = (H - 1 - y_idx).astype(float)
        z_pts = p0[2] - (Nx * (u_pts - p0[0]) + Ny * (v_pts - p0[1])) / Nz
        grid[u_pts.astype(int), v_pts.astype(int)] = z_pts
        
    return grid

# ==========================================
# 3. 3D 预览线程
# ==========================================
class NavMeshState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.dirty3d = False
        self.vertices = []
        self.faces = []

    def push_3d(self, vertices, faces):
        with self.lock:
            self.vertices = [[OX + u * RES, OY + v * RES, z] for u, v, z in vertices]
            self.faces = faces.copy()
            self.dirty3d = True

class Viewer3D(threading.Thread):
    def __init__(self, global_pcd_xyz, state):
        super().__init__(daemon=True)
        self.global_pcd_xyz = global_pcd_xyz
        self.state = state

    def run(self):
        import open3d as o3d
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        vis = o3d.visualization.Visualizer()
        vis.create_window(WIN3D, 1100, 800)

        # 静态全局背景层
        pcd_bg = o3d.geometry.PointCloud()
        if self.global_pcd_xyz is not None and len(self.global_pcd_xyz) > 0:
            pcd_bg.points = o3d.utility.Vector3dVector(self.global_pcd_xyz)
            h = (self.global_pcd_xyz[:, 2] - self.global_pcd_xyz[:, 2].min()) / max(np.ptp(self.global_pcd_xyz[:, 2]), 1e-6)
            bgr = cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
            pcd_bg.colors = o3d.utility.Vector3dVector(0.3 * bgr[:, ::-1].astype(np.float64) / 255.0)
            vis.add_geometry(pcd_bg)

        mesh = o3d.geometry.TriangleMesh()
        line = o3d.geometry.LineSet()
        nodes_pcd = o3d.geometry.PointCloud()
        
        added = False

        while self.state.running:
            v_data, f_data = None, None
            if self.state.dirty3d:
                with self.state.lock:
                    v_data = np.array(self.state.vertices, dtype=np.float64)
                    f_data = np.array(self.state.faces, dtype=np.int32)
                    self.state.dirty3d = False
            
                if added:
                    vis.remove_geometry(mesh, reset_bounding_box=False)
                    vis.remove_geometry(line, reset_bounding_box=False)
                    vis.remove_geometry(nodes_pcd, reset_bounding_box=False)
                    added = False
            
                if v_data is not None and len(v_data) > 0:
                    nodes_pcd.points = o3d.utility.Vector3dVector(v_data)
                    nodes_pcd.paint_uniform_color([0.0, 1.0, 1.0]) 
                    
                    mesh.vertices = o3d.utility.Vector3dVector(v_data)
                    line.points = o3d.utility.Vector3dVector(v_data)
                    
                    if len(f_data) > 0:
                        faces_list = f_data.tolist()
                        double_sided_faces = []
                        for f in faces_list:
                            double_sided_faces.append([f[0], f[1], f[2]])
                            double_sided_faces.append([f[0], f[2], f[1]]) 
                            
                        mesh.triangles = o3d.utility.Vector3iVector(double_sided_faces)
                        mesh.paint_uniform_color([0.88, 0.12, 0.10])
                        mesh.compute_vertex_normals()
                        
                        edges = set()
                        for face in faces_list:
                            edges.add(tuple(sorted((face[0], face[1]))))
                            edges.add(tuple(sorted((face[1], face[2]))))
                            edges.add(tuple(sorted((face[2], face[0]))))
                        line.lines = o3d.utility.Vector2iVector(list(edges))
                        line.paint_uniform_color([1.0, 0.9, 0.0])
                    else:
                        mesh.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))
                        line.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))

                    vis.add_geometry(mesh, reset_bounding_box=False)
                    vis.add_geometry(line, reset_bounding_box=False)
                    vis.add_geometry(nodes_pcd, reset_bounding_box=False)
                    vis.get_render_option().point_size = 8.0 
                    added = True

            if not vis.poll_events(): break
            vis.update_renderer()
            time.sleep(0.02)
        vis.destroy_window()

# ==========================================
# 4. 2D 交互界面
# ==========================================
class BatchNavMeshEditor2D:
    def __init__(self, state):
        self.state = state
        self.current_floor_idx = 0

        cv2.namedWindow(WIN2D, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN2D, self.on_mouse)
        
        self.load_floor_data()

    def load_floor_data(self):
        floor_config = FLOORS_CONFIG[self.current_floor_idx]
        self.map_path = floor_config["map"]
        
        raw_save_path = floor_config["save_path"]
        self.out_prefix = os.path.splitext(raw_save_path)[0] # 去除可能误填的扩展名
        
        read_json_path = floor_config.get("read_json", "")
        
        print(f"\n{'='*50}")
        print(f"[流水线] 正在加载地图 {self.current_floor_idx+1}/{len(FLOORS_CONFIG)}: {os.path.basename(self.map_path)}")
        
        self.img = cv2.imread(self.map_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            print(f"[错误] 无法加载地图 {self.map_path}，跳过此层。")
            self.next_floor()
            return
            
        self.H, self.W = self.img.shape
        
        # 状态重置
        self.vertices = []  
        self.faces = []     
        self.selected_indices = [] 
        self.dragging_idx = None

        # 根据独立开关决定是否读取旧进度
        if read_json_path and os.path.exists(read_json_path):
            print(f"[配置] 发现 read_json 指令，正在加载进度: {read_json_path}")
            try:
                with open(read_json_path, 'r') as f:
                    data = json.load(f)
                    self.vertices = [np.array(v) for v in data.get('vertices', [])]
                    self.faces = data.get('faces', [])
            except Exception as e:
                print(f"[错误] 解析 JSON 失败: {e}")
        else:
            if read_json_path:
                print(f"[提示] 指定的 read_json 文件不存在，将从零开始: {read_json_path}")
            else:
                print(f"[配置] read_json 为空，开启全新画布。")

        self.recompute()

    def recompute(self):
        self.state.push_3d(self.vertices, self.faces)

    def on_mouse(self, event, x, y, flags, param):
        if y >= self.H * DISPLAY_SCALE: return
            
        mx, my = x / DISPLAY_SCALE, y / DISPLAY_SCALE
        u, v = float(mx), float(self.H - 1 - my)
        
        closest_idx = None
        if self.vertices:
            pts = np.array(self.vertices)[:, :2]
            dists = np.hypot(pts[:, 0] - u, pts[:, 1] - v)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < 3.0: 
                closest_idx = min_idx

        # --- 左键负责选中和创建 ---
        if event == cv2.EVENT_LBUTTONDOWN:
            if closest_idx is not None:
                if closest_idx in self.selected_indices:
                    self.selected_indices.remove(closest_idx)
                else:
                    if len(self.selected_indices) < 3:
                        self.selected_indices.append(closest_idx)
                    else:
                        self.selected_indices.pop(0)
                        self.selected_indices.append(closest_idx)
            else:
                px_x, px_y = int(mx), int(my)
                if 0 <= px_x < self.W and 0 <= px_y < self.H:
                    if self.img[px_y, px_x] > 128:
                        new_z = self.vertices[self.selected_indices[-1]][2] if self.selected_indices else 0.0
                        self.vertices.append(np.array([u, v, new_z]))
                        new_idx = len(self.vertices) - 1
                        
                        if len(self.selected_indices) >= 3:
                            self.selected_indices.pop(0)
                        self.selected_indices.append(new_idx)
                    else:
                        print("[限制] 只能在白色可行驶区域添加节点！")
            self.recompute()
            
        elif event == cv2.EVENT_MBUTTONDOWN:
            if closest_idx is not None:
                self.dragging_idx = closest_idx
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_idx is not None:
                px_x, px_y = int(mx), int(my)
                if 0 <= px_x < self.W and 0 <= px_y < self.H and self.img[px_y, px_x] > 128:
                    self.vertices[self.dragging_idx][0] = u
                    self.vertices[self.dragging_idx][1] = v
                    self.recompute()
                    
        elif event == cv2.EVENT_MBUTTONUP:
            self.dragging_idx = None

    def draw(self):
        frame = cv2.resize(self.img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        for face in self.faces:
            pts_cv = np.array([[
                self.vertices[idx][0] * DISPLAY_SCALE,
                (self.H - 1 - self.vertices[idx][1]) * DISPLAY_SCALE
            ] for idx in face], np.int32)
            cv2.polylines(frame, [pts_cv], True, (0, 0, 255), 1, cv2.LINE_AA)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_cv], (0, 0, 255))
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        if len(self.selected_indices) == 2:
            p1 = self.vertices[self.selected_indices[0]]
            p2 = self.vertices[self.selected_indices[1]]
            cv2.line(frame, 
                     (int(p1[0]*DISPLAY_SCALE), int((self.H-1-p1[1])*DISPLAY_SCALE)),
                     (int(p2[0]*DISPLAY_SCALE), int((self.H-1-p2[1])*DISPLAY_SCALE)),
                     (255, 255, 255), 2, cv2.LINE_AA)
        elif len(self.selected_indices) == 3:
            pts_cv = np.array([[
                self.vertices[idx][0] * DISPLAY_SCALE,
                (self.H - 1 - self.vertices[idx][1]) * DISPLAY_SCALE
            ] for idx in self.selected_indices], np.int32)
            cv2.polylines(frame, [pts_cv], True, (255, 255, 255), 2, cv2.LINE_AA)

        for i, pt in enumerate(self.vertices):
            px, py = int(pt[0] * DISPLAY_SCALE), int((self.H - 1 - pt[1]) * DISPLAY_SCALE)
            if i in self.selected_indices:
                cv2.circle(frame, (px, py), 6, (0, 0, 255), -1, cv2.LINE_AA) 
                cv2.circle(frame, (px, py), 8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"{pt[2]:.2f}", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            else:
                cv2.circle(frame, (px, py), 4, (0, 200, 0), -1, cv2.LINE_AA) 

        canvas_h = self.H * DISPLAY_SCALE + 60
        canvas_w = self.W * DISPLAY_SCALE
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[0:self.H * DISPLAY_SCALE, 0:canvas_w] = frame
        cv2.rectangle(canvas, (0, self.H * DISPLAY_SCALE), (canvas_w, canvas_h), (30, 30, 30), -1)

        # 进度与状态提示
        current_map_name = os.path.basename(FLOORS_CONFIG[self.current_floor_idx]["map"])
        bar1 = f"Map {self.current_floor_idx+1}/{len(FLOORS_CONFIG)}: {current_map_name} | p: Save & Next Map | q: Quit"
        
        if len(self.selected_indices) == 1:
            idx = self.selected_indices[0]
            v = self.vertices[idx]
            bar2 = f"Selected 1 pt. w/s to adjust Z. Z={v[2]:.2f}"
        elif len(self.selected_indices) == 3:
            bar2 = "3 points selected! Press 'f' to create a face."
        else:
            bar2 = f"LMB: Add/Sel | MMB: Drag | c: Clear Sel | x/del: Del | f: Face"
            
        cv2.putText(canvas, bar1, (6, self.H*DISPLAY_SCALE + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, bar2, (6, self.H*DISPLAY_SCALE + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN2D, canvas)

    def adjust_z(self, dz):
        if not self.selected_indices: return
        idx = self.selected_indices[-1]
        self.vertices[idx][2] += dz
        self.recompute()

    def on_key(self, key):
        if key == KEY_Z_UP: self.adjust_z(+Z_STEP)
        elif key == KEY_Z_DN: self.adjust_z(-Z_STEP)
        elif key == KEY_Z_UP2: self.adjust_z(+Z_BIG)
        elif key == KEY_Z_DN2: self.adjust_z(-Z_BIG)
        
        elif key == ord("c"):
            self.selected_indices.clear()
            self.recompute()
            
        elif key == ord("f"):
            if len(self.selected_indices) == 3:
                new_face = tuple(sorted(self.selected_indices))
                existing_faces = [tuple(sorted(f)) for f in self.faces]
                if new_face not in existing_faces:
                    self.faces.append(list(self.selected_indices))
                    self.selected_indices.clear()
                    self.recompute()
                    print("[建面] 三角面创建成功！")
                else:
                    print("[建面] 该三角面已存在。")
                    
        elif key in (8, 127, ord("x")): 
            if self.selected_indices:
                idx_to_remove = self.selected_indices[-1]
                self.faces = [f for f in self.faces if idx_to_remove not in f]
                
                new_faces = []
                for face in self.faces:
                    new_face = []
                    for idx in face:
                        if idx > idx_to_remove:
                            new_face.append(idx - 1)
                        else:
                            new_face.append(idx)
                    new_faces.append(new_face)
                self.faces = new_faces
                self.vertices.pop(idx_to_remove)
                self.selected_indices.clear()
                
                print("[删除] 节点及其关联的面已彻底删除！")
                self.recompute()
                
        elif key == ord("p"): self.save_and_next()
        elif key in (ord("q"), 27): return False
        return True

    def save_and_next(self):
        if not self.faces:
            print("[警告] 当前地图没有创建任何面，将保存为空的(NaN)高程图。")
            final_grid = np.full((self.W, self.H), np.nan)
        else:
            print("[生成中] 正在烘焙手工 NavMesh ...")
            final_grid = bake_triangles_to_grid(self.vertices, self.faces, self.W, self.H)
        
        # 1. 保存 NPY
        npy_path = self.out_prefix + ".npy"
        np.save(npy_path, final_grid)
        print(f"[保存成功] 生成 {npy_path}")
        
        # 2. 保存 JSON 供后续重新读取编辑
        if SAVE_JSON:
            json_path = self.out_prefix + ".json"
            json_data = {
                "vertices": [[float(val) for val in pt] for pt in self.vertices],
                "faces": self.faces
            }
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"[保存成功] 写入 {json_path}")
            
        # 3. 保存 PNG 预览
        if SAVE_PNG:
            z = final_grid[np.isfinite(final_grid)]
            preview_path = self.out_prefix + "_preview.png"
            if len(z) > 0:
                z0, z1 = float(z.min()), float(z.max())
                prev = np.where(np.isfinite(final_grid), (final_grid - z0) / max(z1 - z0, 1e-6) * 255, 0).astype(np.uint8).T
                cv2.imwrite(preview_path, cv2.applyColorMap(prev, cv2.COLORMAP_VIRIDIS))
            else:
                cv2.imwrite(preview_path, np.zeros((self.H, self.W, 3), dtype=np.uint8))
            print(f"[保存成功] 渲染 {preview_path}")

        # 切图
        self.next_floor()

    def next_floor(self):
        if self.current_floor_idx < len(FLOORS_CONFIG) - 1:
            self.current_floor_idx += 1
            self.load_floor_data()
        else:
            print("\n" + "="*50)
            print("[完毕] 所有楼层地图的手工铺设已处理完毕！您现在可以按 'q' 退出程序。")
            print("="*50)

    def run(self):
        try:
            while self.state.running:
                self.draw()
                key = cv2.waitKeyEx(30)
                if key == -1: continue
                if not self.on_key(key): break
        except KeyboardInterrupt:
            print("\n[退出] 接收到中断信号，正在安全退出...")
            self.state.running = False
            
        cv2.destroyAllWindows()


# ==========================================
# 5. 主流程
# ==========================================
def main():
    import open3d as o3d
    print(f"[加载] 正在读取全局点云 {PCD_PATH} (整个过程只加载一次)...")
    if not os.path.exists(PCD_PATH):
        sys.exit(f"点云文件不存在: {PCD_PATH}")
        
    pcd = o3d.io.read_point_cloud(PCD_PATH)
    # 稍微降采样可以让 3D 渲染更顺滑
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE) 
    global_pcd_pts = np.asarray(pcd.points)

    state = NavMeshState()
    Viewer3D(global_pcd_pts, state).start()
    
    gui = BatchNavMeshEditor2D(state)
    gui.run()

if __name__ == "__main__":
    main()