#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py —— 多楼层建图与预处理总调度管家 (独立进程退出版)
"""

import os
import sys
import subprocess
import yaml

CONFIG_FILE = "/home/dow/DOW/PRM-3D/PRM-tools/config/zjut.yaml"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"[错误] 找不到全局配置文件: {CONFIG_FILE}")
        sys.exit(1)
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def step1_slicer():
    print("\n[Step 1/3] 正在启动 3D 点云切片与 2D 栅格标注工具...")
    print("提示：绘制完成后按 'p' 保存各层，全部完成后按 'q' 退出窗口返回菜单。\n")
    subprocess.run(["python3", "tools/pcd_slicer_tool.py", "--config", CONFIG_FILE])

def step2_connections():
    print("\n[Step 2/3] 正在启动拓扑连接点标记工具...")
    print("提示：标记完成后按 's' 保存 JSON，按 'q' 退出窗口返回菜单。\n")
    subprocess.run(["python3", "tools/mark_connections.py", "--config", CONFIG_FILE])

def step3_elevation():
    print("\n[Step 3/3] 正在启动 3D 区域生长高程图提取工具...")
    print("提示：放置种子点并按 'p' 逐层保存后，按 'q' 退出窗口返回菜单。\n")
    subprocess.run(["python3", "tools/auto_region_growing.py", "--config", CONFIG_FILE])

def generate_nav_config():
    """根据最新生成的配置和产物，聚合导出最终可供导航直接使用的总 YAML 文件"""
    config = load_config()
    
    global_params = config.get('global', {})
    floors = config.get('floors', [])
    outputs = config.get('outputs', {})
    
    res = float(global_params.get('resolution', 0.05))
    origin_x = float(global_params.get('origin_x', 0.0))
    origin_y = float(global_params.get('origin_y', 0.0))
    connections_json = outputs.get('connections_json', 'connections.json')

    nav_config = {
        "floors": [],
        "connections_json": connections_json,
        "z_source": "npy",
        "map_resolution": res,
        "map_origin_x": origin_x,
        "map_origin_y": origin_y,
        "flip_y": True
    }

    for idx, floor in enumerate(floors):
        z_min = floor.get('z_min', -1.0)
        z_max = floor.get('z_max', 1.0)
        ground_z = round((z_min + z_max) / 2.0, 2) if (z_min is not None and z_max is not None) else 0.0
        
        floor_entry = {
            "index": floor.get('index', idx),
            "ground_z": ground_z,
            "z_min": z_min,
            "z_max": z_max,
            "map": floor.get('map_png', ''),
            "elevation_npy": floor.get('elevation_npy', '')
        }
        nav_config["floors"].append(floor_entry)

    output_yaml_path = outputs.get('nav_final_config', 'nav_multi_floor_config.yaml')
    
    os.makedirs(os.path.dirname(os.path.abspath(output_yaml_path)), exist_ok=True)
    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(nav_config, f, sort_keys=False, default_flow_style=False)
        
    print("\n" + "="*50)
    print(f"[成功] 已成功生成导航总配置文件: {output_yaml_path}")
    print("文件内容预览：")
    print("="*50)
    with open(output_yaml_path, 'r', encoding='utf-8') as f:
        print(f.read())
    print("="*50 + "\n")

def main():
    while True:
        print("\n================== PRM-3D 多楼层预处理流水线 ==================")
        print("1. 步骤一：运行 pcd_slicer_tool (生成 2D 栅格 PNG 并自动记录 z_min/z_max)")
        print("2. 步骤二：运行 mark_connections (标记相邻楼层交集，生成 connections.json)")
        print("3. 步骤三：运行 auto_region_growing (基于种子点与 z 区间提取 .npy 高程图)")
        print("4. 最终步：生成聚合导航配置文件 (输出目标 YAML)")
        print("0. 退出")
        print("================================================================")
        
        choice = input("请输入操作编号 (0-4): ").strip()
        
        if choice == '1':
            step1_slicer()
        elif choice == '2':
            step2_connections()
        elif choice == '3':
            step3_elevation()
        elif choice == '4':
            generate_nav_config()
        elif choice == '0':
            print("退出管家程序。")
            break
        else:
            print("[错误] 无效的输入，请重新选择。")

if __name__ == "__main__":
    main()