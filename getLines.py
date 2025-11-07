import cv2
import numpy as np
import sys

class PointTracker:
    def __init__(self, image_path):
        self.image_path = image_path
        self.points = []
        self.load_image()
        
    def load_image(self):
        """加载并预处理图像"""
        self.original_img = cv2.imread(self.image_path)
        if self.original_img is None:
            print(f"无法加载图像: {self.image_path}")
            sys.exit(1)
        
        # 调整图像大小以适应1200x1200窗口
        h, w = self.original_img.shape[:2]
        scale = min(1200/max(h, w), 1.0)
        self.display_img = cv2.resize(self.original_img, None, fx=scale, fy=scale)
        self.scale_factor = scale
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 记录点击坐标
            self.points.append((x, y))
            print(f"添加点: ({x}, {y})")
            
    def keyboard_handler(self):
        """键盘处理函数"""
        key = cv2.waitKey(100) & 0xFF
        if key == 8 or key == 127:  # Backspace键
            if self.points:
                removed_point = self.points.pop()
                print(f"删除点: {removed_point}")
        elif key == 27:  # ESC键退出
            return True
        return False
    
    def calculate_min_x_distance(self):
        """计算相邻点之间的最小横坐标距离"""
        if len(self.points) < 2:
            return None
            
        min_distance = float('inf')
        for i in range(len(self.points) - 1):
            distance = abs(self.points[i+1][0] - self.points[i][0])
            min_distance = min(min_distance, distance)
            
        return min_distance
    
    def draw_points_and_lines(self, img):
        """在图像上绘制点和射线"""
        # 绘制所有点
        for i, point in enumerate(self.points):
            # 绘制点（红色）
            cv2.circle(img, point, 5, (0, 0, 255), -1)
            # 添加点的序号
            cv2.putText(img, str(i), (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 从该点向下引出射线（绿色）
            cv2.line(img, point, (point[0], img.shape[0]), (0, 255, 0), 1)
    
    def run(self):
        """主程序循环"""
        # 创建窗口
        cv2.namedWindow('Image Viewer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image Viewer', 1200, 1200)
        
        # 设置鼠标回调
        cv2.setMouseCallback('Image Viewer', self.mouse_callback)
        
        print("程序说明:")
        print("- 左键点击添加点")
        print("- 按Backspace键删除最后一个点")
        print("- 按ESC键退出程序")
        print("-" * 40)
        
        while True:
            # 复制图像以避免重复绘制
            display_copy = self.display_img.copy()
            
            # 绘制点和射线
            self.draw_points_and_lines(display_copy)
            
            # 计算并显示最小横坐标距离
            min_distance = self.calculate_min_x_distance()
            if min_distance is not None:
                print(f"相邻点间最小横坐标距离: {min_distance:.2f}")
            
            # 显示当前点数
            cv2.putText(display_copy, f'Points: {len(self.points)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 显示最小距离信息
            if min_distance is not None:
                cv2.putText(display_copy, f'Min X distance: {min_distance:.2f}', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 显示图像
            cv2.imshow('Image Viewer', display_copy)
            
            # 处理键盘输入
            if self.keyboard_handler():
                break
                
        cv2.destroyAllWindows()

def main():
    # 指定图片路径
    # image_path = input("请输入图片路径: ").strip()
    image_path = "/home/tzyh_subsys/WorkSpace/kSNPP/expMap/GAME.png"
    # 创建并运行点追踪器
    tracker = PointTracker(image_path)
    tracker.run()

if __name__ == "__main__":
    main()