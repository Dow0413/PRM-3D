import open3d as o3d
import numpy as np

def fill_point_cloud_holes_with_color(input_path, output_path, depth=9, voxel_size=0.05):
    print("1. 正在读取点云...")
    pcd = o3d.io.read_point_cloud(input_path)
    
    # 检查原始点云是否带有颜色
    has_colors = pcd.has_colors()
    if has_colors:
        print("-> 检测到点云包含颜色/语义信息，将进行颜色传递。")
    else:
        print("-> 警告：原点云没有 RGB 颜色数据，如果是自定义标量字段，Open3D 可能无法直接读取。")

    print("2. 正在估计和对齐法线...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    print("3. 正在运行泊松表面重建...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    print("4. 正在清理多余的重建表面...")
    densities = np.asarray(densities)
    # 导航地图多为平面，可以适当提高过滤阈值，避免生成天上地下的乱点
    density_threshold = np.quantile(densities, 0.05) 
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print("5. 正在重新采样修补点云...")
    num_points = int(len(pcd.points)) # 采样数量可以根据需要调整
    filled_pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=5)

    # ================= 核心修复：颜色/语义传递 =================
    if has_colors:
        print("6. 正在为修补的空洞分配最近邻的颜色/语义...")
        # 构建原始点云的 KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        
        # 准备一个数组存放新点的颜色
        filled_colors = np.zeros((len(filled_pcd.points), 3))
        original_colors = np.asarray(pcd.colors)
        
        # 遍历每一个新生成的点，寻找原点云中最近的 1 个点，拷贝其颜色
        for i, point in enumerate(filled_pcd.points):
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            filled_colors[i] = original_colors[idx[0]]
            
        # 将颜色赋值给修补的点云
        filled_pcd.colors = o3d.utility.Vector3dVector(filled_colors)
    # =========================================================

    print("7. 正在合并并进行体素下采样...")
    merged_pcd = pcd + filled_pcd
    
    # 体素下采样，合并的点云颜色会自动平均化（如果是严格的语义标签，此处会有微小混合，但不影响视觉判断）
    final_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)

    print(f"8. 正在保存处理后的点云至: {output_path}")
    o3d.io.write_point_cloud(output_path, final_pcd)
    
    print("处理完成！正在打开可视化窗口...")
    # 可视化对比：平移一下原点云方便左右对比
    pcd.translate([-pcd.get_max_bound()[0] * 1.5, 0, 0]) 
    o3d.visualization.draw_geometries([pcd, final_pcd], window_name="Semantic Hole Filling Result")

if __name__ == "__main__":
    INPUT_FILE = "/home/dow/maps/x30_pc_nav_traversable_semantic.pcd"   
    OUTPUT_FILE = "/home/dow/maps/traversable_semanticww.pcd"
    
    # 注意：导航地图通常尺度较大，voxel_size 建议设置大一点（比如 0.05m 或 0.1m）
    fill_point_cloud_holes_with_color(INPUT_FILE, OUTPUT_FILE, depth=9, voxel_size=0.05)