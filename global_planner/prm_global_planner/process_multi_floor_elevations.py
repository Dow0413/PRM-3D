import numpy as np
import open3d as o3d
import cv2
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from scipy.interpolate import griddata  # 新增：用于平滑修复点云空洞

# ==========================================
# 1. 多楼层配置字典 (请根据实际情况修改)
# ==========================================
PCD_FILE_PATH = "/home/dow/maps/356_process_3.pcd"

# 楼层配置与 galileo_nav/config/global_prm.yaml 的 floors 保持一致：
# map/origin/resolution 与 yaml 对应；z_range 是为"表面提取"单独调的——
# 覆盖本层地面 + 通向上下层的楼梯爬升，排除其他层的地面。它刻意不照抄
# yaml 的 z_min/z_max（那是按"目标点落层判断"调的，两者目的不同，直接
# 复制会让共用地图的楼层采到别层地面）。origin/resolution 取自 expMap/35/356_1.yaml。
FLOOR_CONFIGS = {
    0: {
        "map_path": "/home/dow/DOW/PRM-3D/expMap/35/356_1.png",
        "z_range": [-1.4, 0.5],
        "resolution": 0.05,
        "origin_x": -15.471000,
        "origin_y": -7.330000,
        "output_npy": "/home/dow/DOW/PRM-3D/galileo_nav_stack/src/global_planner/prm_global_planner/npy/356_0.npy"
    },
    1: {
        "map_path": "/home/dow/DOW/PRM-3D/expMap/35/356_2.png",
        "z_range": [0.5, 1.45],
        "resolution": 0.05,
        "origin_x": -15.471000,
        "origin_y": -7.330000,
        "output_npy": "/home/dow/DOW/PRM-3D/galileo_nav_stack/src/global_planner/prm_global_planner/npy/356_1.npy"
    },
    2: {
        "map_path": "/home/dow/DOW/PRM-3D/expMap/35/356_3.png",
        "z_range": [1.45, 3.0],
        "resolution": 0.05,
        "origin_x": -15.471000,
        "origin_y": -7.330000,
        "output_npy": "/home/dow/DOW/PRM-3D/galileo_nav_stack/src/global_planner/prm_global_planner/npy/356_2.npy"
    },
    3: {
        "map_path": "/home/dow/DOW/PRM-3D/expMap/35/356_4.png",
        "z_range": [3.0, 4.3],
        "resolution": 0.05,
        "origin_x": -15.471000,
        "origin_y": -7.330000,
        "output_npy": "/home/dow/DOW/PRM-3D/galileo_nav_stack/src/global_planner/prm_global_planner/npy/356_3.npy"
    },
    4: {
        "map_path": "/home/dow/DOW/PRM-3D/expMap/35/356_5.png",
        "z_range": [4.3, 5.6],
        "resolution": 0.05,
        "origin_x": -15.471000,
        "origin_y": -7.330000,
        "output_npy": "/home/dow/DOW/PRM-3D/galileo_nav_stack/src/global_planner/prm_global_planner/npy/356_4.npy"
    },
    5: {
        "map_path": "/home/dow/DOW/PRM-3D/expMap/35/356_6.png",
        "z_range": [5.6, 8.1],
        "resolution": 0.05,
        "origin_x": -15.471000,
        "origin_y": -7.330000,
        "output_npy": "/home/dow/DOW/PRM-3D/galileo_nav_stack/src/global_planner/prm_global_planner/npy/356_5.npy"
    }
}

# ==========================================
# 2. 高程提取参数
# ==========================================
RADIUS_STEPS = [0.20, 0.40, 0.80, 1.50]
MIN_POINTS = 15                           
GROUND_PERCENTILE = 10                    
STAIR_RADIUS_STEPS = [0.12, 0.20, 0.40, 0.80]
STAIR_PERCENTILE = 50
SMOOTH_WINDOW = 9                         
BAND_PAD_UP = 0.8                         
STAIR_MARGIN_PX = 4.0                     
CONNECTIONS_JSON_PATH = "/home/dow/DOW/PRM-3D/connections35.json"


def load_stairwell_polygons(json_path):
    """读取 connections json，返回 {floor_id: [多边形(np.float32 Nx2, 图像坐标), ...]}"""
    import json
    import os

    result = {}
    if not json_path or not os.path.exists(json_path):
        print(f"警告: 找不到 connections json ({json_path})，楼梯井将不拓宽波段。")
        return result
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for conn in data.get("connections", []):
        poly = np.asarray(conn["polygon"], dtype=np.float32)
        for floor_id in (conn["fromFloor"], conn["toFloor"]):
            result.setdefault(floor_id, []).append(poly)
    return result


def process_multi_floor_elevations():
    STAIRWELL_POLYGONS = load_stairwell_polygons(CONNECTIONS_JSON_PATH)

    print(f"正在加载全局点云: {PCD_FILE_PATH} ...")
    pcd = o3d.io.read_point_cloud(PCD_FILE_PATH)
    global_points = np.asarray(pcd.points)
    print(f"点云加载完成，共 {len(global_points)} 个点。")

    for floor_id, config in FLOOR_CONFIGS.items():
        print(f"\n--- 开始处理 Floor {floor_id} ---")

        # 1. Z 轴直通滤波 (粗筛)
        z_min, z_max = config["z_range"]
        in_band = (
            (global_points[:, 2] >= z_min) & (global_points[:, 2] <= z_max)
        )
        floor_points = global_points[in_band]
        print(f"Z区间 [{z_min}, {z_max}] 过滤后剩余点数: {len(floor_points)}")
        padded_points = global_points[
            in_band
            | (
                (global_points[:, 2] > z_max)
                & (global_points[:, 2] <= z_max + BAND_PAD_UP)
            )
        ]

        # 2. 读取 2D 栅格地图作为 Mask
        map_img = cv2.imread(config["map_path"], cv2.IMREAD_GRAYSCALE)
        if map_img is None:
            print(f"警告: 找不到地图图片 {config['map_path']}，跳过该楼层。")
            continue

        height_px, width_px = map_img.shape
        elevation_matrix = np.full((width_px, height_px), np.nan)

        res = config["resolution"]
        ox = config["origin_x"]
        oy = config["origin_y"]

        # 3. 收集全部白色(可通行)像素
        white_rows, white_cols = np.nonzero(map_img > 250)
        vs = height_px - 1 - white_rows
        us = white_cols
        print(f"可通行(白色)像素数: {len(us)}")

        # 4. 标记楼梯井像素
        stair_pixel = np.zeros(len(us), dtype=bool)
        connections = STAIRWELL_POLYGONS.get(floor_id, [])
        if connections:
            for i, (u, row) in enumerate(zip(us, white_rows)):
                for poly in connections:
                    if cv2.pointPolygonTest(poly, (float(u), float(row)), True) >= -STAIR_MARGIN_PX:
                        stair_pixel[i] = True
                        break
            print(f"楼梯井多边形内像素: {int(stair_pixel.sum())}")

        # 5. 对每个可通行像素做 KD-tree 邻域搜索
        trees = {}
        if len(floor_points):
            trees[False] = (cKDTree(floor_points[:, :2]), floor_points, RADIUS_STEPS, GROUND_PERCENTILE)
        if len(padded_points):
            trees[True] = (cKDTree(padded_points[:, :2]), padded_points, STAIR_RADIUS_STEPS, STAIR_PERCENTILE)
        
        direct_count = 0
        for i, (u, v) in enumerate(zip(us, vs)):
            entry = trees.get(bool(stair_pixel[i]), trees.get(False))
            if entry is None:
                continue
            tree, pts, radii, percentile = entry
            x = ox + u * res
            y = oy + v * res
            for radius in radii:
                idx = tree.query_ball_point([x, y], r=radius)
                if len(idx) >= MIN_POINTS:
                    elevation_matrix[u, v] = np.percentile(pts[idx, 2], percentile)
                    direct_count += 1
                    break
        print(f"邻域搜索直接得到高程的像素: {direct_count}/{len(us)}")

        # 6. 核心优化：采用混合插值法处理因点云缺口、空洞导致缺失 Z 值的像素
        valid = np.isfinite(elevation_matrix[us, vs])
        if valid.any() and not valid.all():
            valid_us, valid_vs = us[valid], vs[valid]
            invalid_us, invalid_vs = us[~valid], vs[~valid]
            
            points_valid = np.column_stack([valid_us, valid_vs])
            values_valid = elevation_matrix[valid_us, valid_vs]
            points_invalid = np.column_stack([invalid_us, invalid_vs])

            # 步骤 6.1: 优先使用线性插值 (Linear)，在空洞内部生成连续平滑的过渡坡度
            filled_values = griddata(points_valid, values_valid, points_invalid, method='linear')

            # 步骤 6.2: 线性插值无法外插凸包外的边界点（会返回 NaN），对这些点使用最近邻 (Nearest) 兜底
            nan_mask = np.isnan(filled_values)
            if nan_mask.any():
                nearest_values = griddata(points_valid, values_valid, points_invalid[nan_mask], method='nearest')
                filled_values[nan_mask] = nearest_values

            # 回填修复的高程
            elevation_matrix[invalid_us, invalid_vs] = filled_values
            print(f"采用混合插值修复了 {(~valid).sum()} 个因点云缺失导致空洞的像素。")

        elif not valid.any() and len(us) > 0:
            # 防呆设计：如果极端情况下该楼层完全没有提取到有效高程点云
            default_z = (z_min + z_max) / 2.0
            elevation_matrix[us, vs] = default_z
            print(f"警告: 该楼层未能提取到任何有效高程！使用 Z 区间中值 {default_z} 进行全局兜底。")

        # 7. 平滑滤波
        if len(us):
            walkable = np.zeros((width_px, height_px))
            walkable[us, vs] = 1.0
            val = np.nan_to_num(elevation_matrix, nan=0.0) * walkable
            val_sum = uniform_filter(val, size=SMOOTH_WINDOW, mode="nearest")
            cnt_sum = uniform_filter(walkable, size=SMOOTH_WINDOW, mode="nearest")
            smoothed = val_sum / np.maximum(cnt_sum, 1e-6)
            elevation_matrix[us, vs] = smoothed[us, vs]

        # 8. 保存为 .npy 文件
        np.save(config["output_npy"], elevation_matrix)
        finite = int(np.isfinite(elevation_matrix[us, vs]).sum())
        print(f"Floor {floor_id} 提取完成！可通行像素高程覆盖率: {finite}/{len(us)}")
        print(f"矩阵已保存至: {config['output_npy']}")


if __name__ == "__main__":
    process_multi_floor_elevations()