import sys
import json
import urllib.request
import numpy as np
import open3d as o3d
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox,
                             QGroupBox, QFileDialog, QMessageBox, QGridLayout)
from PyQt5.QtCore import Qt

BRIDGE_BASE_URL = 'http://localhost:8765'

class PCDViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCD 交互式点云查看器")
        self.setGeometry(100, 100, 1200, 800)
        
        # 数据存储
        self.point_cloud = None
        self.pcd_mesh = None
        self.node_widget = None
        self.current_node_center = (0.0, 0.0, 0.0)

        self.init_ui()

    def init_ui(self):
        # 主控部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左侧工具栏 ---
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)

        # 1. 文件操作区
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        self.btn_load = QPushButton("导入 PCD 文件")
        self.btn_load.clicked.connect(self.load_pcd)
        self.btn_load.setStyleSheet("padding: 8px; font-weight: bold;")
        file_layout.addWidget(self.btn_load)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # 2. 视角控制区 (新增)
        view_group = QGroupBox("视角控制 (快捷查看方位)")
        view_layout = QGridLayout()
        
        self.btn_top = QPushButton("上方 (Top)")
        self.btn_bottom = QPushButton("下方 (Bottom)")
        self.btn_front = QPushButton("前方 (Front)")
        self.btn_back = QPushButton("后方 (Back)")
        self.btn_left = QPushButton("左侧 (Left)")
        self.btn_right = QPushButton("右侧 (Right)")
        self.btn_iso = QPushButton("复位 (Iso)")

        # 绑定视角切换事件
        self.btn_top.clicked.connect(lambda: self.change_view('xy'))
        self.btn_bottom.clicked.connect(lambda: self.change_view('-xy'))
        self.btn_front.clicked.connect(lambda: self.change_view('xz'))
        self.btn_back.clicked.connect(lambda: self.change_view('-xz'))
        self.btn_right.clicked.connect(lambda: self.change_view('yz'))
        self.btn_left.clicked.connect(lambda: self.change_view('-yz'))
        self.btn_iso.clicked.connect(lambda: self.change_view('iso'))

        # 将按钮加入网格布局
        view_layout.addWidget(self.btn_top, 0, 0)
        view_layout.addWidget(self.btn_bottom, 0, 1)
        view_layout.addWidget(self.btn_front, 1, 0)
        view_layout.addWidget(self.btn_back, 1, 1)
        view_layout.addWidget(self.btn_left, 2, 0)
        view_layout.addWidget(self.btn_right, 2, 1)
        view_layout.addWidget(self.btn_iso, 3, 0, 1, 2) # 复位按钮横跨两列

        # 初始化时禁用视角按钮
        self.set_view_buttons_enabled(False)
        
        view_group.setLayout(view_layout)
        left_layout.addWidget(view_group)

        # 3. Z值过滤区
        z_group = QGroupBox("Z值区间高亮")
        z_layout = QVBoxLayout()
        
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min Z:"))
        self.spin_min_z = QDoubleSpinBox()
        self.spin_min_z.setRange(-10000.0, 10000.0)
        self.spin_min_z.setDecimals(2)
        min_layout.addWidget(self.spin_min_z)
        z_layout.addLayout(min_layout)

        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max Z:"))
        self.spin_max_z = QDoubleSpinBox()
        self.spin_max_z.setRange(-10000.0, 10000.0)
        self.spin_max_z.setDecimals(2)
        max_layout.addWidget(self.spin_max_z)
        z_layout.addLayout(max_layout)

        self.btn_apply_z = QPushButton("应用 Z 值区间")
        self.btn_apply_z.clicked.connect(self.apply_z_filter)
        self.btn_apply_z.setEnabled(False)
        z_layout.addWidget(self.btn_apply_z)
        
        z_group.setLayout(z_layout)
        left_layout.addWidget(z_group)

        # 4. Node 节点管理区
        node_group = QGroupBox("Node 节点 (拖拽球体实时更新)")
        node_layout = QVBoxLayout()
        
        self.btn_add_node = QPushButton("在地图中心添加 Node")
        self.btn_add_node.clicked.connect(self.add_node)
        self.btn_add_node.setEnabled(False)
        node_layout.addWidget(self.btn_add_node)

        self.lbl_node_x = QLabel("X: 0.000")
        self.lbl_node_y = QLabel("Y: 0.000")
        self.lbl_node_z = QLabel("Z: 0.000")
        self.lbl_node_x.setStyleSheet("font-family: monospace; font-size: 14px; color: blue;")
        self.lbl_node_y.setStyleSheet("font-family: monospace; font-size: 14px; color: green;")
        self.lbl_node_z.setStyleSheet("font-family: monospace; font-size: 14px; color: red;")
        
        node_layout.addWidget(self.lbl_node_x)
        node_layout.addWidget(self.lbl_node_y)
        node_layout.addWidget(self.lbl_node_z)

        self.btn_publish_goal = QPushButton("发送目标点 (Send as Goal)")
        self.btn_publish_goal.clicked.connect(self.publish_goal)
        self.btn_publish_goal.setEnabled(False)
        self.btn_publish_goal.setStyleSheet("padding: 8px; font-weight: bold; color: #fff; background-color: #1677ff;")
        node_layout.addWidget(self.btn_publish_goal)

        node_group.setLayout(node_layout)
        left_layout.addWidget(node_group)

        # --- 右侧 3D 视图区 ---
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#2b2b2b")
        self.plotter.add_axes()

        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.plotter.interactor)

    def set_view_buttons_enabled(self, state):
        """批量设置视角按钮的启用状态"""
        self.btn_top.setEnabled(state)
        self.btn_bottom.setEnabled(state)
        self.btn_front.setEnabled(state)
        self.btn_back.setEnabled(state)
        self.btn_left.setEnabled(state)
        self.btn_right.setEnabled(state)
        self.btn_iso.setEnabled(state)

    def change_view(self, view_name):
        """切换摄像机视角"""
        if self.point_cloud is None:
            return
            
        if view_name == 'iso':
            self.plotter.view_isometric()
        else:
            # PyVista 支持通过字符串直接设置正交视角，例如 'xy' (Z正向下看), '-xy' (Z负向上看)
            self.plotter.camera_position = view_name
            
        # 重置相机距离以确保点云完整显示在画面中
        self.plotter.reset_camera()

    def load_pcd(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 PCD 文件", "", "Point Cloud Files (*.pcd);;All Files (*)", options=options)
        
        if file_path:
            try:
                pcd = o3d.io.read_point_cloud(file_path)
                points = np.asarray(pcd.points)
                
                if len(points) == 0:
                    QMessageBox.warning(self, "警告", "PCD 文件中没有点云数据！")
                    return

                self.point_cloud = pv.PolyData(points)
                self.point_cloud["highlight"] = np.zeros(len(points))

                self.plotter.clear()
                self.plotter.add_axes()
                
                self.pcd_mesh = self.plotter.add_mesh(
                    self.point_cloud, 
                    scalars="highlight", 
                    cmap=["#a0a0a0", "#ffaa00"], 
                    clim=[0, 1],
                    point_size=2.0, 
                    show_scalar_bar=False,
                    render_points_as_spheres=True 
                )
                self.plotter.reset_camera()

                # 更新 UI 状态
                z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
                self.spin_min_z.setValue(z_min)
                self.spin_max_z.setValue(z_max)
                self.btn_apply_z.setEnabled(True)
                self.btn_add_node.setEnabled(True)
                self.set_view_buttons_enabled(True) # 启用视角按钮
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"读取文件失败:\n{str(e)}")

    def apply_z_filter(self):
        if self.point_cloud is None or self.pcd_mesh is None:
            return
            
        z_min = self.spin_min_z.value()
        z_max = self.spin_max_z.value()
        
        z_values = self.point_cloud.points[:, 2]
        mask = (z_values >= z_min) & (z_values <= z_max)
        new_scalars = mask.astype(int)
        
        self.point_cloud["highlight"] = new_scalars
        self.plotter.update_scalars(new_scalars, mesh=self.pcd_mesh, render=True)

    def add_node(self):
        if self.point_cloud is None:
            return

        if self.node_widget is not None:
            self.plotter.clear_sphere_widgets()
            
        center = self.point_cloud.center
        bounds = self.point_cloud.bounds
        radius = (bounds[1] - bounds[0]) * 0.02 
        if radius == 0: radius = 0.5
        
        self.node_widget = self.plotter.add_sphere_widget(
            callback=self.update_node_ui,
            center=center,
            radius=radius,
            color="red",
            test_callback=True
        )
        self.btn_publish_goal.setEnabled(True)
        self.update_node_ui(center)

    def update_node_ui(self, center):
        self.current_node_center = (center[0], center[1], center[2])
        self.lbl_node_x.setText(f"X: {center[0]:.3f}")
        self.lbl_node_y.setText(f"Y: {center[1]:.3f}")
        self.lbl_node_z.setText(f"Z: {center[2]:.3f}")

    def publish_goal(self):
        """将当前 Node 位置作为目标点发布到 ROS bridge"""
        x, y, z = self.current_node_center
        goal_pos = {"x": x, "y": y, "z": z}
        try:
            data = json.dumps(goal_pos).encode('utf-8')
            req = urllib.request.Request(
                f"{BRIDGE_BASE_URL}/publish_goal",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            QMessageBox.information(
                self, "目标点已发送",
                f"Goal 已发布 ({x:.2f}, {y:.2f}, {z:.2f})\n{result}"
            )
        except Exception as e:
            QMessageBox.warning(
                self, "发送失败",
                f"无法连接到 bridge ({BRIDGE_BASE_URL}/publish_goal):\n{str(e)}"
            )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = PCDViewer()
    window.show()
    sys.exit(app.exec_())
