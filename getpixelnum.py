import cv2
import numpy as np

def count_black_and_white(image_path):
    # 读取图像，cv2 默认读取为 BGR 格式
    img = cv2.imread(image_path)
    
    if img is None:
        print("错误：无法读取图片，请检查路径是否正确。")
        return
    
    # 纯白色的 BGR 值为 [255, 255, 255]
    # 使用 np.all 在最后一个维度 (axis=-1) 上检查 RGB 三个通道是否全部匹配
    white_pixels = np.sum(np.all(img == [255, 255, 255], axis=-1))
    
    # 纯黑色的 BGR 值为 [0, 0, 0]
    black_pixels = np.sum(np.all(img == [0, 0, 0], axis=-1))
    
    print(f"统计结果:")
    print(f" -> 白色像素点数量: {white_pixels}")
    print(f" -> 黑色像素点数量: {black_pixels}")
    
    return black_pixels, white_pixels

# 使用示例（请将 'your_image.jpg' 替换为你的图片路径）
image_file = '/home/dow/DOW/PRM-3D/expMap/galileo_5.png'
count_black_and_white(image_file)