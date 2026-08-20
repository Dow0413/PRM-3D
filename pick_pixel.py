#!/usr/bin/env python3
"""
鼠标点击图像获取像素坐标。
- 左键点击：打印坐标和像素值
- 右键点击：复制最后坐标到剪贴板（方便粘贴到 C++ 代码）
- 按 q 退出，按 s 保存标记图
"""

import sys
import cv2
import numpy as np


class PixelPicker:
    def __init__(self, img_path):
        self.img_orig = cv2.imread(img_path)
        if self.img_orig is None:
            print(f"Error: cannot read {img_path}")
            sys.exit(1)
        self.img_gray = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2GRAY)

        self.img_disp = self.img_orig.copy()
        self.window = "Pixel Picker"
        self.clicks = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        print(f"Image: {img_path}")
        print(f"Size:  {self.img_orig.shape[1]} x {self.img_orig.shape[0]}")
        print()
        print("Left click  — print pixel coordinates & value")
        print("Right click — copy last coordinate (for C++ code)")
        print("s          — save marked image")
        print("q / ESC    — quit")
        print()

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1200, 900)
        cv2.setMouseCallback(self.window, self._on_mouse)
        self._redraw()

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            b, g, r = self.img_orig[y, x]
            gray = self.img_gray[y, x]
            self.clicks.append((x, y))

            cv2.circle(self.img_disp, (x, y), 4, (0, 0, 255), -1)
            cv2.circle(self.img_disp, (x, y), 6, (255, 255, 255), 1)
            label = f"{len(self.clicks)}:({x},{y})"
            cv2.putText(self.img_disp, label, (x + 12, y - 12),
                        self.font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            self._redraw()

            print(f"#{len(self.clicks):>2} | "
                  f"({x:>5}, {y:>5}) | "
                  f"BGR=({b:>3},{g:>3},{r:>3}) Gray={gray:>3} | "
                  f"{{{(float(x)):.1f}, {(float(y)):.1f}, 0}}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.clicks:
                x, y = self.clicks[-1]
                code = f"{{{float(x):.1f}, {float(y):.1f}, 0}}"
                try:
                    import pyperclip
                    pyperclip.copy(code)
                    print(f"  -> copied: {code}")
                except ImportError:
                    print(f"  -> {code}  (install pyperclip for auto-copy)")

    def _redraw(self):
        h, w = self.img_disp.shape[:2]
        # 坐标刻度线
        overlay = self.img_disp.copy()
        step = max(50, min(w, h) // 20)
        for gx in range(0, w, step):
            cv2.line(overlay, (gx, 0), (gx, h), (40, 40, 40), 1)
        for gy in range(0, h, step):
            cv2.line(overlay, (0, gy), (w, gy), (40, 40, 40), 1)
        self.img_disp = cv2.addWeighted(overlay, 0.7, self.img_disp, 0.3, 0)
        cv2.imshow(self.window, self.img_disp)

    def run(self):
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                out = f"marked_{len(self.clicks)}pts.png"
                cv2.imwrite(out, self.img_disp)
                print(f"Saved: {out}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pick_pixel.py <image_path>")
        print("Example: python pick_pixel.py expMap/floor0.png")
        sys.exit(1)
    PixelPicker(sys.argv[1]).run()
