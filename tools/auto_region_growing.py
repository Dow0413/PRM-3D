#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_region_growing.py —— 批量流水线：节点交互式 3D 区域生长提取工具 (自动桥接断层版)

操作说明：
  - 左键单击 (LMB)：点击空白处 -> 创建新种子点；点击已有种子点 -> 选中。
  - 中键按住 (MMB)：拖动种子点，松开鼠标瞬间会自动重新生长并刷新 3D。
  - x / delete / backspace：彻底删除选中的种子点并刷新提取区域。
  - c：清空当前的选中状态。
  - p 键：保存当前提取的合并高程图为 .npy，并自动无缝切换到下一张地图！
  - q 键：退出。
"""

import os
import sys
import time
import threading
from collections import deque

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

# ==========================================
# 1. 批量多楼层配置
# ==========================================
PCD_PATH = "/home/dow/maps/zjut.pcd"

FLOORS_CONFIG = [
    {
        "index": 0,
        "z_min": -1.620,
        "z_max": -0.603,
        "map": "/home/dow/DOW/PRM-3D/expMap/zjut/zjut_1.png",
        "elevation_npy": "/home/dow/DOW/PRM-3D/expMap/zjut/zjut_1.npy"
    },
    {
        "index": 1,
        "z_min": -0.770,
        "z_max": 3.897,
        "map": "/home/dow/DOW/PRM-3D/expMap/zjut/zjut_2.png",
        "elevation_npy": "/home/dow/DOW/PRM-3D/expMap/zjut/zjut_2.npy"
    },
    {
        "index": 2,
        "z_min": 3.580,
        "z_max": 5.197,
        "map": "/home/dow/DOW/PRM-3D/expMap/zjut/zjut_3.png",
        "elevation_npy": "/home/dow/DOW/PRM-3D/expMap/zjut/zjut_3.npy"
    }
]

# FLOORS_CONFIG = [
#     {
#         "index": 0,
#         "z_min": -1.4,
#         "z_max": 0.6,
#         "map": "/home/dow/DOW/PRM-3D/expMap/35_all/356_1.png",
#         "elevation_npy": "/home/dow/DOW/PRM-3D/expMap/35_all_2/356_1.npy"
#     },
#     {
#         "index": 1,
#         "z_min": -1.4,
#         "z_max": 2.5,
#         "map": "/home/dow/DOW/PRM-3D/expMap/35_all/356_2.png",
#         "elevation_npy": "/home/dow/DOW/PRM-3D/expMap/35_all_2/356_2.npy"
#     },
#     {
#         "index": 2,
#         "z_min": 2.3,
#         "z_max": 2.5,
#         "map": "/home/dow/DOW/PRM-3D/expMap/35_all/356_3.png",
#         "elevation_npy": "/home/dow/DOW/PRM-3D/expMap/35_all_2/356_3.npy"
#     },
#     {
#         "index": 3,
#         "z_min": 2.3,
#         "z_max": 5.2,
#         "map": "/home/dow/DOW/PRM-3D/expMap/35_all/356_4.png",
#         "elevation_npy": "/home/dow/DOW/PRM-3D/expMap/35_all_2/356_4.npy"
#     },
#     {
#         "index": 4,
#         "z_min": 4.9,
#         "z_max": 6.4,
#         "map": "/home/dow/DOW/PRM-3D/expMap/35_all/356_5.png",
#         "elevation_npy": "/home/dow/DOW/PRM-3D/expMap/35_all_2/356_5.npy"
#     },
#     {
#         "index": 5,
#         "z_min": 6.15,
#         "z_max": 7.0,
#         "map": "/home/dow/DOW/PRM-3D/expMap/35_all/356_6.png",
#         "elevation_npy": "/home/dow/DOW/PRM-3D/expMap/35_all_2/356_6.npy"
#     }
# ]

# --- 全局对齐参数 ---
RES, OX, OY = 0.05, -20.099316, -55.290977

# --- 【新增】界面与渲染控制参数 ---
MAX_2D_WIN_SIZE = 800     # 2D 窗口最大显示尺寸限制 (像素)，防止超大地图撑爆屏幕
VOXEL_SIZE = 0.05         # 3D 点云全局降采样体素大小 (米)，调大可提升超大点云帧率

# --- 机器人高度补偿与断层填补 ---
ROBOT_HEIGHT = 0.60       
GAP_FILL_RADIUS_PX = 20   

# --- 动态跳跃生长的核心参数 ---
SEARCH_RADIUS = 0.4       # 搜索半径极大扩张，允许算法隔空跨越 80 厘米的物理断层
MAX_SLOPE = 1.2           # 允许的最大坡度 (dz/dxy，约 50 度)，防止其跳跃时爬上垂直墙壁
BASE_Z_STEP = 0.20        # 相邻点允许的基础台阶高度 (米)

WIN2D = "2D Map - Interactive Seed Editor"
WIN3D = "3D Region Preview (Real-time)"

# ==========================================
# 2. 核心算法：区域生长
# ==========================================
def run_global_region_growing(pcd_points, kdtree, seeds_uv, W, H):
    global_visited = np.zeros(len(pcd_points), dtype=bool)
    total_extracted = set()
    valid_3d_seeds = []
    
    for (u, v) in seeds_uv:
        x, y = OX + u * RES, OY + v * RES
        dists_xy = np.hypot(pcd_points[:, 0] - x, pcd_points[:, 1] - y)
        close_mask = dists_xy < 1.0 
        
        if np.any(close_mask):
            close_indices = np.where(close_mask)[0]
            valid_close = [idx for idx in close_indices if not global_visited[idx]]
            if valid_close:
                seed_idx = valid_close[np.argmin(pcd_points[valid_close, 2])]
                valid_3d_seeds.append(pcd_points[seed_idx])
                
                global_visited[seed_idx] = True
                queue = deque([seed_idx])
                total_extracted.add(seed_idx)
                
                while queue:
                    curr_idx = queue.popleft()
                    curr_pt = pcd_points[curr_idx]
                    neighbors = kdtree.query_ball_point(curr_pt, SEARCH_RADIUS)
                    
                    for n_idx in neighbors:
                        if not global_visited[n_idx]:
                            n_pt = pcd_points[n_idx]
                            
                            dz = abs(curr_pt[2] - n_pt[2])
                            dxy = ((curr_pt[0] - n_pt[0])**2 + (curr_pt[1] - n_pt[1])**2)**0.5
                            
                            # 动态跳跃约束：距离越远，允许跳跃的高差越大（只要满足正常坡度）
                            if dz <= BASE_Z_STEP + MAX_SLOPE * dxy:
                                global_visited[n_idx] = True
                                queue.append(n_idx)
                                total_extracted.add(n_idx)
                                
    return list(total_extracted), np.array(valid_3d_seeds) if valid_3d_seeds else np.zeros((0, 3))

def generate_dense_surface(valid_points, W, H, white_mask_grid):
    grid = np.full((W, H), np.nan)
    if len(valid_points) == 0:
        return grid, np.zeros((0, 3))
        
    u_idx = np.round((valid_points[:, 0] - OX) / RES).astype(int)
    v_idx = np.round((valid_points[:, 1] - OY) / RES).astype(int)
    
    valid_mask = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)
    u_idx, v_idx = u_idx[valid_mask], v_idx[valid_mask]
    z_vals = valid_points[valid_mask, 2]
    
    from collections import defaultdict
    grid_dict = defaultdict(list)
    for u, v, z in zip(u_idx, v_idx, z_vals):
        grid_dict[(u, v)].append(z)
        
    for (u, v), zs in grid_dict.items():
        grid[u, v] = np.median(zs)
        
    extracted_mask = ~np.isnan(grid)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GAP_FILL_RADIUS_PX, GAP_FILL_RADIUS_PX))
    dilated_mask = cv2.dilate(extracted_mask.astype(np.uint8), kernel) > 0
    target_mask = dilated_mask & white_mask_grid
    
    u_valid, v_valid = np.nonzero(extracted_mask)
    z_valid = grid[extracted_mask]
    
    hole_mask = target_mask & ~extracted_mask
    u_hole, v_hole = np.nonzero(hole_mask)
    
    if len(u_hole) > 0:
        # 在线性空间上把中间空缺的数据补上
        fill_z = griddata((u_valid, v_valid), z_valid, (u_hole, v_hole), method='linear')
        nan_fill = np.isnan(fill_z)
        if nan_fill.any():
            fill_z[nan_fill] = griddata((u_valid, v_valid), z_valid, (u_hole[nan_fill], v_hole[nan_fill]), method='nearest')
        grid[u_hole, v_hole] = fill_z
        
    u_dense, v_dense = np.nonzero(target_mask)
    z_dense = grid[u_dense, v_dense]
    dense_pts = np.column_stack([OX + u_dense * RES, OY + v_dense * RES, z_dense])
    
    return grid, dense_pts

# ==========================================
# 3. 3D 预览线程
# ==========================================
class RegionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.dirty3d = False
        self.extracted_pts = np.zeros((0, 3))
        self.seed_pts_3d = np.zeros((0, 3))

    def push_3d(self, extracted, seeds):
        with self.lock:
            self.extracted_pts = extracted
            self.seed_pts_3d = seeds
            self.dirty3d = True

class Viewer3D(threading.Thread):
    def __init__(self, global_pcd_xyz, state):
        super().__init__(daemon=True)
        self.global_pcd_xyz = global_pcd_xyz
        self.state = state

    def run(self):
        try:
            import open3d as o3d
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
            vis = o3d.visualization.Visualizer()
            vis.create_window(WIN3D, 1100, 800)
            
            pcd_bg = o3d.geometry.PointCloud()
            pcd_bg.points = o3d.utility.Vector3dVector(self.global_pcd_xyz)
            pcd_bg.paint_uniform_color([0.3, 0.3, 0.3])
            vis.add_geometry(pcd_bg)

            pcd_extracted = o3d.geometry.PointCloud()
            pcd_seeds = o3d.geometry.PointCloud()
            added = False

            while self.state.running:
                ex_pts, sd_pts = None, None
                if self.state.dirty3d:
                    with self.state.lock:
                        ex_pts = self.state.extracted_pts.copy()
                        sd_pts = self.state.seed_pts_3d.copy()
                        self.state.dirty3d = False
                
                if ex_pts is not None:
                    if added:
                        vis.remove_geometry(pcd_extracted, reset_bounding_box=False)
                        vis.remove_geometry(pcd_seeds, reset_bounding_box=False)
                        added = False
                        
                    if len(ex_pts) > 0:
                        pcd_extracted.points = o3d.utility.Vector3dVector(ex_pts)
                        pcd_extracted.paint_uniform_color([0.9, 0.1, 0.1]) 
                        
                        pcd_seeds.points = o3d.utility.Vector3dVector(sd_pts)
                        pcd_seeds.paint_uniform_color([0.0, 1.0, 1.0]) 
                        
                        vis.add_geometry(pcd_extracted, reset_bounding_box=False)
                        vis.add_geometry(pcd_seeds, reset_bounding_box=False)
                        vis.get_render_option().point_size = 4.0 
                        added = True

                if not vis.poll_events():
                    break
                vis.update_renderer()
                time.sleep(0.02)
            vis.destroy_window()
        except Exception as e:
            print(f"[3D] 预览线程退出: {e}")

# ==========================================
# 4. 2D 交互界面
# ==========================================
class BatchRegionTool2D:
    def __init__(self, global_pcd_points, state):
        self.global_pcd = global_pcd_points
        self.state = state
        self.current_floor_idx = 0
        self.render_scale = 1.0  # 【新增】动态缩放系数
        
        cv2.namedWindow(WIN2D, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN2D, self.on_mouse)
        
        # 加载第一张地图
        self.load_floor_data()

    def load_floor_data(self):
        floor_config = FLOORS_CONFIG[self.current_floor_idx]
        map_path = floor_config["map"]
        self.z_min = floor_config["z_min"]
        self.z_max = floor_config["z_max"]
        self.out_npy = floor_config["elevation_npy"]
        
        print(f"\n{'='*50}")
        print(f"[流水线] 正在加载地图 {self.current_floor_idx+1}/{len(FLOORS_CONFIG)}: {os.path.basename(map_path)}")
        print(f"[流水线] Z轴截断区间: [{self.z_min:.3f}m, {self.z_max:.3f}m]")
        
        self.img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            print(f"[错误] 无法加载地图 {map_path}，跳过此层。")
            self.next_floor()
            return
            
        self.H, self.W = self.img.shape
        
        self.white_mask_grid = np.zeros((self.W, self.H), dtype=bool)
        UU, VV = np.meshgrid(np.arange(self.W), np.arange(self.H), indexing='ij')
        self.white_mask_grid[UU, VV] = self.img[self.H - 1 - VV, UU] > 128
        
        t0 = time.time()
        z_mask = (self.global_pcd[:, 2] >= self.z_min) & (self.global_pcd[:, 2] <= self.z_max)
        sliced_pcd = self.global_pcd[z_mask]
        
        u_idx = np.round((sliced_pcd[:, 0] - OX) / RES).astype(int)
        v_idx = np.round((sliced_pcd[:, 1] - OY) / RES).astype(int)
        
        valid_mask = (u_idx >= 0) & (u_idx < self.W) & (v_idx >= 0) & (v_idx < self.H)
        sliced_pcd = sliced_pcd[valid_mask]
        u_idx, v_idx = u_idx[valid_mask], v_idx[valid_mask]
        
        white_mask = self.white_mask_grid[u_idx, v_idx]
        self.pcd_points = sliced_pcd[white_mask]
        print(f"[预处理] 切片完成，当前层有效 3D 点数: {len(self.pcd_points)} (耗时: {time.time()-t0:.2f}s)")
        
        if len(self.pcd_points) > 0:
            self.kdtree = cKDTree(self.pcd_points)
        else:
            self.kdtree = None
            
        self.seeds_uv = []           
        self.selected_idx = None     
        self.dragging_idx = None     
        self.current_dense_grid = None  
        
        self.state.push_3d(np.zeros((0, 3)), np.zeros((0, 3)))

    def trigger_recompute(self):
        if not self.seeds_uv or self.kdtree is None:
            self.current_dense_grid = None
            self.state.push_3d(np.zeros((0, 3)), np.zeros((0, 3)))
            return
            
        extracted_indices, valid_3d_seeds = run_global_region_growing(
            self.pcd_points, self.kdtree, self.seeds_uv, self.W, self.H
        )
        
        if extracted_indices:
            extracted_pts = self.pcd_points[extracted_indices]
            grid, dense_pts = generate_dense_surface(extracted_pts, self.W, self.H, self.white_mask_grid)
            self.current_dense_grid = grid
            
            dense_pts[:, 2] += ROBOT_HEIGHT
            if len(valid_3d_seeds) > 0:
                valid_3d_seeds[:, 2] += ROBOT_HEIGHT
                
            self.state.push_3d(dense_pts, valid_3d_seeds)
        else:
            self.current_dense_grid = None
            self.state.push_3d(np.zeros((0, 3)), np.zeros((0, 3)))

    def on_mouse(self, event, x, y, flags, param):
        # 【修改】使用动态缩放系数逆推真实图像坐标
        scale = getattr(self, 'render_scale', 1.0)
        new_H = int(self.H * scale)
        if y >= new_H: return 
            
        mx, my = x / scale, y / scale
        u, v = float(mx), float(self.H - 1 - my)
        
        closest_idx = None
        if self.seeds_uv:
            pts = np.array(self.seeds_uv)
            dists = np.hypot(pts[:, 0] - u, pts[:, 1] - v)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < 3.0: 
                closest_idx = min_idx

        if event == cv2.EVENT_LBUTTONDOWN:
            if closest_idx is not None:
                if self.selected_idx == closest_idx: self.selected_idx = None
                else: self.selected_idx = closest_idx
            else:
                px_x, px_y = int(mx), int(my)
                if 0 <= px_x < self.W and 0 <= px_y < self.H:
                    if self.img[px_y, px_x] > 128:
                        self.seeds_uv.append([u, v])
                        self.selected_idx = len(self.seeds_uv) - 1
                        self.trigger_recompute()
                        
        elif event == cv2.EVENT_MBUTTONDOWN:
            if closest_idx is not None:
                self.dragging_idx = closest_idx
                self.selected_idx = closest_idx
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_idx is not None:
                px_x, px_y = int(mx), int(my)
                if 0 <= px_x < self.W and 0 <= px_y < self.H and self.img[px_y, px_x] > 128:
                    self.seeds_uv[self.dragging_idx] = [u, v]
                    
        elif event == cv2.EVENT_MBUTTONUP:
            if self.dragging_idx is not None:
                self.dragging_idx = None
                self.trigger_recompute()

    def draw(self):
        # 【修改】限制 2D 窗口的最大显示尺寸，避免越界
        scale = min(MAX_2D_WIN_SIZE / float(self.W), MAX_2D_WIN_SIZE / float(self.H))
        scale = min(scale, 4.0) # 最大不超过 4 倍放大
        self.render_scale = scale
        
        new_W = max(1, int(self.W * scale))
        new_H = max(1, int(self.H * scale))
        
        frame = cv2.resize(self.img, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        for i, (u, v) in enumerate(self.seeds_uv):
            px_x = int(u * scale)
            px_y = int((self.H - 1 - v) * scale)
            
            if i == self.selected_idx:
                cv2.circle(frame, (px_x, px_y), 6, (0, 0, 255), -1, cv2.LINE_AA) 
                cv2.circle(frame, (px_x, px_y), 8, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px_x, px_y), 5, (0, 255, 255), -1, cv2.LINE_AA) 

        canvas_h = new_H + 60
        canvas_w = new_W
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[0:new_H, 0:new_W] = frame
        cv2.rectangle(canvas, (0, new_H), (canvas_w, canvas_h), (30, 30, 30), -1)

        current_map_name = os.path.basename(FLOORS_CONFIG[self.current_floor_idx]["map"])
        bar1 = f"Map {self.current_floor_idx+1}/{len(FLOORS_CONFIG)}: {current_map_name} | p: Save & Next Map | q: Quit"
        
        if self.selected_idx is not None:
            bar2 = f"Seed {self.selected_idx+1} selected. (LMB: Add/Select | MMB: Drag | x: Del)"
        else:
            bar2 = f"Total {len(self.seeds_uv)} seeds. (LMB: Add/Select | MMB: Drag | x: Del)"
            
        cv2.putText(canvas, bar1, (6, new_H + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, bar2, (6, new_H + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(WIN2D, canvas)

    def on_key(self, key):
        if key == ord("c"):
            self.selected_idx = None
            
        elif key in (8, 127, ord("x")): 
            if self.selected_idx is not None:
                self.seeds_uv.pop(self.selected_idx)
                self.selected_idx = None
                self.trigger_recompute()
                
        elif key == ord("p"): 
            self.save_and_next()
        elif key in (ord("q"), 27): 
            return False
        return True

    def save_and_next(self):
        # 【修改】保存时打印当前高度信息
        print("\n" + "="*50)
        print(f"[正在保存] 地图 {self.current_floor_idx + 1}/{len(FLOORS_CONFIG)}")
        print(f"  - 截断高度: z_min = {self.z_min:.3f} m, z_max = {self.z_max:.3f} m")

        if self.current_dense_grid is None:
            print("[警告] 当前地图未放置任何种子点，将保存为空的(NaN)高程图。")
            final_grid = np.full((self.W, self.H), np.nan)
        else:
            final_grid = self.current_dense_grid.copy()
            final_grid += ROBOT_HEIGHT
            
        np.save(self.out_npy, final_grid)
        
        z = final_grid[np.isfinite(final_grid)]
        preview_path = os.path.splitext(self.out_npy)[0] + "_preview.png"
        if len(z) > 0:
            z0, z1 = float(z.min()), float(z.max())
            prev = np.where(np.isfinite(final_grid), (final_grid - z0) / max(z1 - z0, 1e-6) * 255, 0).astype(np.uint8).T
            cv2.imwrite(preview_path, cv2.applyColorMap(prev, cv2.COLORMAP_VIRIDIS))
        else:
            cv2.imwrite(preview_path, np.zeros((self.H, self.W, 3), dtype=np.uint8))
            
        print(f"[保存成功] 已写入 NPY 高程图及预览 PNG: {self.out_npy}")
        print("="*50 + "\n")

        self.next_floor()

    def next_floor(self):
        if self.current_floor_idx < len(FLOORS_CONFIG) - 1:
            self.current_floor_idx += 1
            self.load_floor_data()
        else:
            print("\n" + "="*50)
            print("[完毕] 所有楼层地图均已处理并保存完毕！您现在可以按 'q' 退出程序。")
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
    
    # 【修改】全局点云降采样，大幅提高交互与算法运算速度
    print(f"[降采样] 正在应用 {VOXEL_SIZE}m 体素滤波...")
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    global_pcd_pts = np.asarray(pcd.points)

    state = RegionState()
    Viewer3D(global_pcd_pts, state).start()
    
    tool = BatchRegionTool2D(global_pcd_pts, state)
    tool.run()

if __name__ == "__main__":
    main()