import numpy as np
import open3d as o3d
import CSF  # <--- 注意：这里必须是大写的 CSF
import os

# ==========================================
# 1. 核心配置区：只依赖 3D 原始数据
# ==========================================
# 你的原始全局 PCD 文件路径
INPUT_PCD = "/home/dow/maps/map_0_1_crop_no_colors_new.pcd" 
OUTPUT_PCD = "/home/dow/DOW/PRM-3D/global_ground_only.pcd"

# 已经根据你的配置同步了 5 个楼层的 Z 轴切片区间
Z_SLICES = [
    [6.49, 7.17],   # index: 0 (lx_1)
    [2.82, 6.49],   # index: 1 (lx_2)
    [2.82, 3.58],   # index: 2 (lx_3)
    [-0.08, 2.82],  # index: 3 (lx_4)
    [-1.0, -0.08]   # index: 4 (lx_5)
]

def extract_global_ground_pcd():
    if not os.path.exists(INPUT_PCD):
        print(f"错误：找不到点云文件 {INPUT_PCD}")
        return

    print(f"正在加载原始全局点云: {INPUT_PCD} ...")
    pcd = o3d.io.read_point_cloud(INPUT_PCD)
    global_points = np.asarray(pcd.points)
    
    all_ground_points = []

    # 遍历每个切片，分别提取地面
    for i, z_range in enumerate(Z_SLICES):
        z_min, z_max = z_range
        print(f"\n--- 正在处理 Z 轴切片 {i}: [{z_min}, {z_max}] ---")
        
        # 💡 【核心修改 1：大重叠软切片】
        # 将容差从 0.05(5厘米) 暴增到 0.5(50厘米)！
        # 这意味着相邻切片之间会有整整 1.0 米的“公共重叠区”。
        # 即使你的 Z_SLICES 给得不准（偏差个二三十厘米），楼梯也会在这个巨大的缓冲带里被完整提取。
        # (向下多切 50cm 绝对安全，因为这还碰不到楼下的天花板)
        margin = 0.5 
        mask = (global_points[:, 2] >= (z_min - margin)) & (global_points[:, 2] <= (z_max + margin))
        slice_points = global_points[mask]
        
        if len(slice_points) == 0:
            print(f"切片 {i} 内未找到点云，跳过。")
            continue

        # 💡 【核心修改 2：布料物理特性调优】
        csf_alg = CSF.CSF()
        csf_alg.params.bSloopSmooth = True      
        
        # 将布料网格从 0.2 缩小到 0.1。更密集的网格能死死卡进楼梯台阶的缝隙，防止漏点。
        csf_alg.params.cloth_resolution = 0.1   
        
        # 将布料硬度从 3 降为 2。这让布料变得更柔软，在楼梯折角处不会发生僵硬的翘曲。
        csf_alg.params.rigidness = 2            
        csf_alg.params.time_step = 0.65
        
        csf_alg.setPointCloud(slice_points.tolist())
        ground_indices = CSF.VecInt()
        non_ground_indices = CSF.VecInt()
        csf_alg.do_filtering(ground_indices, non_ground_indices)
        
        # 提取出纯净地面点，并加入全局集合
        ground_points = slice_points[np.array(ground_indices)]
        all_ground_points.append(ground_points)
        print(f"该切片提取地面点: {len(ground_points)} 个。")

    # 3. 将所有楼层的地面点合并为一个完整的全局点云
    if not all_ground_points:
        print("提取失败，没有找到任何地面点。")
        return

    print("\n--- 正在合并全局地面点云 ---")
    merged_points = np.vstack(all_ground_points)

    # 4. 去重 (因为相邻 Z 切片在交界处可能会有重叠点)
    # 使用体素下采样 (Voxel Downsample) 来实现优雅的去重和均匀化
    temp_pcd = o3d.geometry.PointCloud()
    temp_pcd.points = o3d.utility.Vector3dVector(merged_points)
    
    # 体素大小设为 0.05 米，既能去重又能保证足够的高程精度
    final_pcd = temp_pcd.voxel_down_sample(voxel_size=0.05)

    # 5. 保存最终的单一全局 PCD
    o3d.io.write_point_cloud(OUTPUT_PCD, final_pcd)
    print(f"✅ 成功生成全局可通行地面点云！合并去重后共 {len(final_pcd.points)} 个点。")
    print(f"文件已保存至: {OUTPUT_PCD}")

if __name__ == "__main__":
    extract_global_ground_pcd()