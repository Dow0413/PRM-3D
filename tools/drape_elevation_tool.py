#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_navmesh_tool.py —— 纯手动导航网格 (NavMesh) 铺设工具

操作说明：
  - 右键单击 (RMB)：在白色区域点击 -> 创建新顶点。
  - 左键单击 (LMB)：点击现有顶点 -> 选中 / 取消选中（用于连面或调Z）。
  - 中键按住 (MMB)：按住现有顶点拖动 -> 实时改变顶点 X,Y 坐标（限制在白区）。
  - w / s：选中了单个顶点时，微调该顶点的 Z 值 (e/x 为大步调)。
  - f (Face)：当刚好选中 3 个顶点时，按下 f 键将它们连成一个三角面。
  - x / delete / backspace：彻底删除选中的节点（会自动解除关联的面）。
  - p：保存结果。
"""

import os
import sys
import threading
import time
import json

import cv2
import numpy as np

# ==========================================
# 1. 配置
# ==========================================
PCD_PATH = "/home/dow/maps/356_process_3.pcd"
MAP_PATH = "/home/dow/DOW/PRM-3D/expMap/35/356_2.png" 
OUT_PREFIX = "/home/dow/DOW/PRM-3D/tools/navmesh_out"

# --- [新增] 数据保存与加载配置开关 ---
SAVE_PNG = False
SAVE_JSON = True
# 若要读取进度，请填入完整 json 路径，例如: "/home/dow/DOW/PRM-3D/tools/navmesh_out.json"
# 留空 "" 则表示从零开始全新创建
LOAD_JSON_PATH = "" 

# 全局原点与分辨率参数
RES, OX, OY = 0.05, -15.471, -7.330     

Z_STEP, Z_BIG = 0.02, 0.10   
DISPLAY_SCALE = 3            

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
    def __init__(self, pcd_xyz, state):
        super().__init__(daemon=True)
        self.pcd_xyz = pcd_xyz
        self.state = state

    def run(self):
        import open3d as o3d
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        vis = o3d.visualization.Visualizer()
        vis.create_window(WIN3D, 1100, 800)
        
        p = self.pcd_xyz
        h = (p[:, 2] - p[:, 2].min()) / max(np.ptp(p[:, 2]), 1e-6)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p)
        bgr = cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
        colors = bgr[:, ::-1].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(0.3 * colors) 
        vis.add_geometry(pcd)

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
                        # 【修复 1】强制构建双面网格解决背向导致无颜色的问题
                        faces_list = f_data.tolist()
                        double_sided_faces = []
                        for f in faces_list:
                            double_sided_faces.append([f[0], f[1], f[2]])
                            double_sided_faces.append([f[0], f[2], f[1]]) # 反向法线
                            
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
class NavMeshEditor2D:
    def __init__(self, map_img, out_prefix, state):
        self.img = map_img
        self.out = out_prefix
        self.state = state
        self.H, self.W = map_img.shape
        
        self.vertices = []  
        self.faces = []     
        self.selected_indices = [] 
        
        self.dragging_idx = None # 记录当前正在中键拖拽的节点索引

        # 【修复 2】加载历史进度
        if LOAD_JSON_PATH and os.path.exists(LOAD_JSON_PATH):
            print(f"[配置] 正在加载历史 JSON 进度: {LOAD_JSON_PATH}")
            try:
                with open(LOAD_JSON_PATH, 'r') as f:
                    data = json.load(f)
                    self.vertices = [np.array(v) for v in data.get('vertices', [])]
                    self.faces = data.get('faces', [])
            except Exception as e:
                print(f"[错误] 解析 JSON 失败: {e}")

        cv2.namedWindow(WIN2D, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN2D, self.on_mouse)
        self.recompute()

    def recompute(self):
        self.state.push_3d(self.vertices, self.faces)

    def on_mouse(self, event, x, y, flags, param):
        if y >= self.H * DISPLAY_SCALE: return
            
        mx, my = x / DISPLAY_SCALE, y / DISPLAY_SCALE
        u, v = float(mx), float(self.H - 1 - my)
        
        # 寻找最近的节点
        closest_idx = None
        if self.vertices:
            pts = np.array(self.vertices)[:, :2]
            dists = np.hypot(pts[:, 0] - u, pts[:, 1] - v)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < 3.0: 
                closest_idx = min_idx

        # --- 【交互重构】 ---
        # 1. 鼠标左键：纯粹用作选中/取消选中
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
                self.selected_indices.clear()
            self.recompute()
            
        # 2. 鼠标右键：专职在空地创建节点
        elif event == cv2.EVENT_RBUTTONDOWN:
            px_x, px_y = int(mx), int(my)
            if 0 <= px_x < self.W and 0 <= px_y < self.H:
                if self.img[px_y, px_x] > 128:
                    new_z = self.vertices[self.selected_indices[-1]][2] if self.selected_indices else 0.0
                    self.vertices.append(np.array([u, v, new_z]))
                    new_idx = len(self.vertices) - 1
                    
                    if len(self.selected_indices) >= 3:
                        self.selected_indices.pop(0)
                    self.selected_indices.append(new_idx)
                    self.recompute()
                else:
                    print("[限制] 只能在白色可行驶区域添加节点！")

        # 3. 鼠标中键按住：进入拖动状态
        elif event == cv2.EVENT_MBUTTONDOWN:
            if closest_idx is not None:
                self.dragging_idx = closest_idx
                
        # 4. 鼠标移动：实现拖拽平滑更新
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_idx is not None:
                px_x, px_y = int(mx), int(my)
                # 严格限制：拖动依然不能越界，不能拉进黑色墙壁里
                if 0 <= px_x < self.W and 0 <= px_y < self.H and self.img[px_y, px_x] > 128:
                    self.vertices[self.dragging_idx][0] = u
                    self.vertices[self.dragging_idx][1] = v
                    self.recompute()
                    
        # 5. 鼠标中键松开：结束拖拽
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

        bar1 = "LMB: Select | RMB: Add | MMB: Drag | x/del: Del | f: Face | p: Save | q: Quit"
        if len(self.selected_indices) == 1:
            idx = self.selected_indices[0]
            v = self.vertices[idx]
            bar2 = f"Selected 1 pt. w/s to adjust Z. Z={v[2]:.2f}"
        elif len(self.selected_indices) == 3:
            bar2 = "3 points selected! Press 'f' to create a face."
        else:
            bar2 = f"{len(self.selected_indices)} points selected."
            
        cv2.putText(canvas, bar1, (6, self.H*DISPLAY_SCALE + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, bar2, (6, self.H*DISPLAY_SCALE + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
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
                
        elif key == ord("p"): self.save()
        elif key in (ord("q"), 27): return False
        return True

    def save(self):
        if not self.faces:
            print("[保存] 当前没有面，保存的数据将为空。")
            
        print("[生成中] 正在烘焙网格并执行保存任务...")
        final_grid = bake_triangles_to_grid(self.vertices, self.faces, self.W, self.H)
        
        # 无论如何保存 .npy (可能是空矩阵，方便下游清空状态)
        np.save(self.out + ".npy", final_grid)
        print(f"[保存成功] 生成 {self.out}.npy")
        
        # 【修复 2】按照配置开关执行额外的保存
        if SAVE_JSON:
            json_data = {
                "vertices": [[float(val) for val in pt] for pt in self.vertices],
                "faces": self.faces
            }
            with open(self.out + ".json", 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"[保存成功] 写入 {self.out}.json")
            
        if SAVE_PNG:
            z = final_grid[np.isfinite(final_grid)]
            if len(z) > 0:
                z0, z1 = float(z.min()), float(z.max())
                prev = np.where(np.isfinite(final_grid), (final_grid - z0) / max(z1 - z0, 1e-6) * 255, 0).astype(np.uint8).T
                cv2.imwrite(self.out + "_preview.png", cv2.applyColorMap(prev, cv2.COLORMAP_VIRIDIS))
                print(f"[保存成功] 渲染 {self.out}_preview.png")

    def run(self):
        while self.state.running:
            self.draw()
            key = cv2.waitKeyEx(30)
            if key == -1: continue
            if not self.on_key(key): break
        self.state.running = False
        cv2.destroyAllWindows()

# ==========================================
# 5. 主流程
# ==========================================
def main():
    map_img = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    if map_img is None: 
        sys.exit(f"地图读取失败, 请检查路径是否正确: {MAP_PATH}")

    import open3d as o3d
    print(f"[加载] 正在读取点云 {PCD_PATH} 用于背景参考...")
    try:
        pcd_pts = np.asarray(o3d.io.read_point_cloud(PCD_PATH).points)
    except Exception as e:
        sys.exit(f"点云读取失败，请检查路径是否正确: {PCD_PATH}\n错误信息: {e}")

    state = NavMeshState()
    Viewer3D(pcd_pts, state).start()
    
    gui = NavMeshEditor2D(map_img, OUT_PREFIX, state)
    gui.run()

if __name__ == "__main__":
    main()