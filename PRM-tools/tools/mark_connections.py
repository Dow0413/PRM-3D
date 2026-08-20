#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_connections.py —— PRM-3D 跨楼层/跨地图「拓扑连接点」标记工具 (配置版)

用法
----
  python3 tools/mark_connections.py --config map_pipeline.yaml --expand 8 --cost 15
"""

import argparse
import json
import os
import sys
import yaml

import cv2
import numpy as np

# 配色 (BGR)
C_BG     = (40, 40, 40)     # 背景 / 障碍
C_MAP_I  = (0, 170, 0)      # 第 i 张图独有自由区（绿）
C_MAP_J  = (0, 0, 170)      # 第 i+1 张图独有自由区（红）
C_CAND   = (0, 220, 220)    # 候选过渡区（未选中，黄）
C_SELECT = (240, 220, 0)    # 已选中（青）
C_MANUAL = (255, 0, 255)    # 手动补点（品红）

def free_mask(gray):
    """原始可行走掩膜：灰度>128 视为白色/可行走。"""
    return gray > 128


def inflated_free(gray, expand):
    """膨胀后的可行走掩膜：障碍(<=128)向外膨胀 expand 像素。"""
    if expand <= 0:
        return free_mask(gray)
    obstacle = (gray <= 128).astype(np.uint8)
    ks = int(2 * expand + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    dilated = cv2.dilate(obstacle, kernel)
    return dilated == 0


def snap(x, y, valid, max_r=80):
    """把 (x, y) 吸附到 valid 掩膜里最近的可行走像素；失败返回 None。"""
    H, W = valid.shape
    ix, iy = int(round(x)), int(round(y))
    if 0 <= ix < W and 0 <= iy < H and valid[iy, ix]:
        return ix, iy
    for r in range(1, max_r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                xx, yy = ix + dx, iy + dy
                if 0 <= xx < W and 0 <= yy < H and valid[yy, xx]:
                    return xx, yy
    return None


def detect_candidates(free_i, free_j, min_area):
    """交集连通域 = 候选过渡区。返回 (labels, [ {label, area, cx, cy}, ... ])。"""
    ov = (free_i & free_j).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(ov, connectivity=8)
    cands = []
    for k in range(1, n):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = cents[k]
        cands.append({"label": k, "area": area, "cx": float(cx), "cy": float(cy)})
    return labels, cands


def polygon_from_label(labels, lab, epsilon=1.5):
    """从一个连通域 label 提取外轮廓多边形。"""
    mask = (labels == lab).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    if epsilon > 0 and len(c) > 4:
        c = cv2.approxPolyDP(c, epsilon, True)
    poly = [[int(p[0][0]), int(p[0][1])] for p in c]
    if len(poly) > 1 and poly[0] != poly[-1]:
        poly.append(poly[0])
    return poly


class ConnectionMarker:
    def __init__(self, maps, expand, cost, min_area, out_path, eps=1.5):
        self.grays = maps
        self.expand = expand
        self.cost = cost
        self.min_area = min_area
        self.out_path = out_path
        self.eps = eps

        self.n = len(maps)
        self.raw = [free_mask(g) for g in maps]
        self.infl = [inflated_free(g, expand) for g in maps]

        self.pairs = [(i, i + 1) for i in range(self.n - 1)]
        self.labels = []       
        self.cands = []        
        self.cand_map = []     
        for (i, j) in self.pairs:
            labs, cs = detect_candidates(self.raw[i], self.raw[j], min_area)
            self.labels.append(labs)
            self.cands.append(cs)
            self.cand_map.append({c["label"]: c for c in cs})

        self.sel = [set() for _ in self.pairs]     
        self.manual = [[] for _ in self.pairs]     
        self.pair_idx = 0
        self.mode = "select"                       
        self.modified = False
        self.dirty = True

    def _text(self, img, txt, pos, scale=0.6):
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)

    def render(self):
        i, j = self.pairs[self.pair_idx]
        fi, fj = self.raw[i], self.raw[j]
        ov = fi & fj
        h, w = fi.shape
        img = np.full((h, w, 3), C_BG, np.uint8)
        img[fi] = C_MAP_I
        img[fj] = C_MAP_J
        img[ov] = C_CAND

        labels = self.labels[self.pair_idx]
        for lab in self.sel[self.pair_idx]:
            img[labels == lab] = C_SELECT

        for c in self.cands[self.pair_idx]:
            cx, cy = int(c["cx"]), int(c["cy"])
            cv2.circle(img, (cx, cy), 3, (0, 0, 0), -1)
            self._text(img, str(c["label"]), (cx + 8, cy - 8), 0.7)

        for (x, y) in self.manual[self.pair_idx]:
            cv2.drawMarker(img, (x, y), C_MANUAL, cv2.MARKER_STAR, 14, 2)

        last = "  (最后一对，按 s 保存)" if self.pair_idx == len(self.pairs) - 1 else ""
        hud = [
            "Pair %d/%d   floor%d <-> floor%d%s"
            % (self.pair_idx + 1, len(self.pairs), i, j, last),
            "mode: %s   candidates: %d   selected: %d   manual: %d   cost: %g"
            % (self.mode.upper(), len(self.cands[self.pair_idx]),
               len(self.sel[self.pair_idx]), len(self.manual[self.pair_idx]), self.cost),
            "[click]=%s   [m]mode  [p]next  [b]back  [c]clear  [s]save  [q]quit"
            % ("select candidate" if self.mode == "select" else "add manual point"),
        ]
        for k, line in enumerate(hud):
            self._text(img, line, (8, 18 + k * 22), 0.55)
        return img

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        p = self.pair_idx
        i, j = self.pairs[p]
        h, w = self.raw[i].shape
        if not (0 <= x < w and 0 <= y < h):
            return

        if self.mode == "select":
            lab = int(self.labels[p][y, x])
            if lab <= 0:
                print("[pair%d] clicked empty area (not a candidate)" % (p + 1))
                return
            if lab in self.sel[p]:
                self.sel[p].discard(lab)
            else:
                self.sel[p].add(lab)
            self.modified = self.dirty = True
            c = self.cand_map[p][lab]
            print("[pair%d] candidate %d -> %s" % (p + 1, lab, "SELECTED" if lab in self.sel[p] else "deselected"))
        else:  
            valid = self.infl[i] & self.infl[j]
            s = snap(x, y, valid)
            if s is None:
                print("[pair%d] manual point skipped: no walkable pixel nearby" % (p + 1))
                return
            self.manual[p].append(s)
            self.modified = self.dirty = True
            print("[pair%d] manual point added at %s" % (p + 1, s))

    def connections(self):
        conns = []
        for p, (i, j) in enumerate(self.pairs):
            valid = self.infl[i] & self.infl[j]
            labels = self.labels[p]
            for lab in sorted(self.sel[p]):
                c = self.cand_map[p][lab]
                poly = polygon_from_label(labels, lab, self.eps)
                s = snap(c["cx"], c["cy"], valid)
                if s is None:
                    continue
                conns.append({
                    "fromFloor": i,
                    "toFloor": j,
                    "cost": float(self.cost),
                    "area": c["area"],
                    "polygon": poly,                       
                    "centroid": [float(s[0]), float(s[1])],  
                })
            for (x, y) in self.manual[p]:
                conns.append({
                    "fromFloor": i,
                    "toFloor": j,
                    "cost": float(self.cost),
                    "area": 0,
                    "polygon": [],                         
                    "centroid": [float(x), float(y)],
                })
        return conns

    def save(self):
        doc = {
            "numFloors": self.n,
            "defaultCost": float(self.cost),
            "expandRadius": float(self.expand),
            "connections": self.connections(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print("saved %d connections -> %s" % (len(doc["connections"]), self.out_path))
        self.modified = False

    def run(self):
        if self.n < 2:
            print("need at least 2 maps")
            return
        WIN = "mark_connections (PRM-3D)"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self.on_mouse)

        print("提示：先点窗口激活，再用按键。%d 对相邻地图待标记。\n" % len(self.pairs))
        while True:
            if self.dirty:
                cv2.imshow(WIN, self.render())
                self.dirty = False
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                if self.modified:
                    print("WARNING: 有未保存的改动。")
                break
            # if key in (27, ord("q")):
            #     break
            elif key == ord("m"):
                self.mode = "manual" if self.mode == "select" else "select"
                self.dirty = True
                print("mode ->", self.mode)
            elif key == ord("p"):
                if self.pair_idx < len(self.pairs) - 1:
                    self.pair_idx += 1
                    self.dirty = True
                    print("-> pair %d/%d" % (self.pair_idx + 1, len(self.pairs)))
                else:
                    print("已在最后一对，按 s 保存 JSON")
            elif key == ord("b"):
                self.pair_idx = max(0, self.pair_idx - 1)
                self.dirty = True
                print("-> pair %d/%d" % (self.pair_idx + 1, len(self.pairs)))
            elif key == ord("c"):
                self.sel[self.pair_idx].clear()
                self.manual[self.pair_idx] = []
                self.modified = self.dirty = True
                print("[pair%d] cleared" % (self.pair_idx + 1))
            elif key == ord("s"):
                self.save()
                self.dirty = True

        cv2.destroyAllWindows()


def headless(marker):
    print("maps=%d  expand=%g  min_area=%d\n" % (marker.n, marker.expand, marker.min_area))
    for p, (i, j) in enumerate(marker.pairs):
        print("=== pair %d: map[%d] <-> map[%d]  (%d candidates) ==="
              % (p + 1, i, j, len(marker.cands[p])))
        valid = marker.infl[i] & marker.infl[j]
        for c in marker.cands[p]:
            s = snap(c["cx"], c["cy"], valid)
            poly = polygon_from_label(marker.labels[p], c["label"], marker.eps)
            print("  cand %d: area=%5d  centroid=(%6.1f,%6.1f)  snap->%s  polygon_verts=%d"
                  % (c["label"], c["area"], c["cx"], c["cy"], s, len(poly)))
    print()


def main():
    ap = argparse.ArgumentParser(description="PRM-3D 拓扑连接点标记工具")
    ap.add_argument("--config", type=str, default="map_pipeline.yaml", help="全局 YAML 配置文件路径")
    ap.add_argument("--expand", type=float, default=8.0, help="障碍膨胀半径（像素）")
    ap.add_argument("--cost", type=float, default=15.0, help="每条跨层连接的默认代价")
    ap.add_argument("--min-area", type=int, default=8, help="候选连通域最小面积")
    ap.add_argument("--eps", type=float, default=1.5, help="多边形轮廓简化阈值")
    ap.add_argument("--headless", action="store_true", help="只打印候选并退出")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"配置文件不存在: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    floors = config_data.get('floors', [])
    if len(floors) < 2:
        sys.exit("配置文件中定义的 floors 数量少于 2 层，无法进行跨层连接。")

    # 从 YAML 动态读取所有地图路径与输出路径
    map_paths = [floor['map_png'] for floor in floors]
    out_path = config_data.get('outputs', {}).get('connections_json', 'connections.json')

    grays = []
    for p in map_paths:
        if not os.path.exists(p):
            sys.exit(f"无法加载地图文件: {p}")
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"解析地图失败: {p}")
        grays.append(img)

    marker = ConnectionMarker(grays, args.expand, args.cost, args.min_area, out_path, args.eps)
    if args.headless:
        headless(marker)
    else:
        marker.run()


if __name__ == "__main__":
    main()