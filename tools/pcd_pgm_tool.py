"""Interactive PCD / PGM to PGM tooling — single-file version."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import io
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk


# ─────────────────────────── 常量 ───────────────────────────

TRAVERSABLE_LABEL = 1
BLOCKED_LABEL = 2
MAX_GRID_CELLS = 40_000_000

LABEL_TO_VALUE = {
    "traversable": TRAVERSABLE_LABEL,
    "blocked": BLOCKED_LABEL,
    "erase": 0,
}

LABEL_TO_TEXT = {
    "traversable": "可通行",
    "blocked": "不可通行",
    "erase": "擦除",
}

LABEL_TO_COLOR = {
    "traversable": "#28aa5f",
    "blocked": "#d23c32",
    "erase": "#f5a623",
}


# ─────────────────────────── 数据类 ───────────────────────────

@dataclass
class PCDHeader:
    fields: List[str]
    sizes: List[int]
    types: List[str]
    counts: List[int]
    width: int
    height: int
    points: int
    data: str
    data_offset: int


@dataclass
class GridMetadata:
    origin_x: float
    origin_y: float
    resolution: float
    width: int
    height: int
    extent_locked: bool


@dataclass
class MapResult:
    image: np.ndarray
    point_counts: np.ndarray
    metadata: GridMetadata
    filtered_points: int
    z_min: float
    z_max: float


@dataclass
class PreprocessParams:
    rotation_deg: float = 0.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    crop_percent: float = 0.0
    keep_box: tuple[float, float, float, float] | None = None
    keep_largest_component: bool = False
    component_resolution: float = 0.05
    component_min_cells: int = 3
    component_min_points_per_cell: int = 1

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class PolygonAnnotation:
    label: str
    points_world: list[tuple[float, float]]


# ─────────────────────────── PCD I/O ───────────────────────────

_DTYPE_MAP: Dict[Tuple[str, int], str] = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("I", 1): "<i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
    ("I", 8): "<i8",
    ("U", 1): "<u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("U", 8): "<u8",
}


def load_pcd_xyz(path: str | Path) -> np.ndarray:
    """Load x/y/z coordinates from a PCD file as an (N, 3) float32 array."""
    file_path = Path(path)
    header = _read_pcd_header(file_path)
    required_fields = {"x", "y", "z"}
    if not required_fields.issubset(set(header.fields)):
        raise ValueError(f"PCD 缺少必要字段 x/y/z: {file_path}")
    if header.points == 0:
        return np.empty((0, 3), dtype=np.float32)
    if header.data == "ascii":
        xyz = _load_ascii_xyz(file_path, header)
    elif header.data == "binary":
        xyz = _load_binary_xyz(file_path, header)
    elif header.data == "binary_compressed":
        raise NotImplementedError(
            "暂不支持 DATA binary_compressed 格式，请先转换成 ascii 或 binary 再使用。"
        )
    else:
        raise ValueError(f"不支持的 PCD DATA 类型: {header.data}")
    finite_mask = np.isfinite(xyz).all(axis=1)
    return np.ascontiguousarray(xyz[finite_mask], dtype=np.float32)


def save_pcd_xyz(path: str | Path, xyz: np.ndarray) -> Path:
    """Save an (N, 3) XYZ array as an ASCII PCD file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("要保存的点云必须是形如 (N, 3) 的数组。")
    finite_mask = np.isfinite(points).all(axis=1)
    points = np.ascontiguousarray(points[finite_mask], dtype=np.float32)
    header_lines = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {points.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {points.shape[0]}\n"
        "DATA ascii\n"
    )
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(header_lines)
        np.savetxt(handle, points, fmt="%.6f %.6f %.6f")
    return output_path


def _read_pcd_header(path: Path) -> PCDHeader:
    header_items: Dict[str, str] = {}
    with path.open("rb") as handle:
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"未找到 DATA 头，无法解析 PCD: {path}")
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue
            key, *rest = line.split()
            upper_key = key.upper()
            header_items[upper_key] = " ".join(rest)
            if upper_key == "DATA":
                data_offset = handle.tell()
                break
    fields = header_items.get("FIELDS", "").split()
    sizes = _parse_int_list(header_items.get("SIZE", ""))
    types = header_items.get("TYPE", "").split()
    counts = _parse_int_list(header_items.get("COUNT", "")) or [1] * len(fields)
    width = int(header_items.get("WIDTH", "0"))
    height = int(header_items.get("HEIGHT", "1"))
    points = int(header_items.get("POINTS", str(width * height)))
    data = header_items.get("DATA", "").strip().lower()
    if not (fields and sizes and types):
        raise ValueError(f"PCD 头信息不完整: {path}")
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError(f"PCD 字段定义数量不一致: {path}")
    return PCDHeader(
        fields=fields, sizes=sizes, types=types, counts=counts,
        width=width, height=height, points=points, data=data, data_offset=data_offset,
    )


def _parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(item) for item in value.split()]


def _load_ascii_xyz(path: Path, header: PCDHeader) -> np.ndarray:
    xyz_columns = [header.fields.index(name) for name in ("x", "y", "z")]
    with path.open("rb") as handle:
        handle.seek(header.data_offset)
        text_stream = io.TextIOWrapper(handle, encoding="utf-8", newline="")
        xyz = np.loadtxt(text_stream, dtype=np.float32, usecols=xyz_columns, ndmin=2)
    return xyz


def _load_binary_xyz(path: Path, header: PCDHeader) -> np.ndarray:
    dtype = _build_structured_dtype(header)
    with path.open("rb") as handle:
        handle.seek(header.data_offset)
        cloud = np.fromfile(handle, dtype=dtype, count=header.points)
    xyz = np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(np.float32, copy=False)
    return xyz


def _build_structured_dtype(header: PCDHeader) -> np.dtype:
    entries = []
    for field, size, field_type, count in zip(
        header.fields, header.sizes, header.types, header.counts
    ):
        dtype_code = _DTYPE_MAP.get((field_type.upper(), size))
        if dtype_code is None:
            raise ValueError(f"不支持的字段类型: TYPE={field_type} SIZE={size}")
        if count == 1:
            entries.append((field, dtype_code))
        else:
            entries.append((field, dtype_code, (count,)))
    return np.dtype(entries)


# ─────────────────────────── PGM I/O ───────────────────────────

def load_pgm_to_map_result(path: str | Path) -> MapResult:
    """Load a PGM file (optionally with companion YAML / JSON) into a MapResult.

    This allows the user to open an existing PGM map and draw annotations on it
    without needing a source PCD file.
    """
    file_path = Path(path)

    # Load the PGM image
    img = Image.open(file_path).convert("L")
    image_array = np.asarray(img, dtype=np.uint8)
    # Keep image as-is; coordinate conversion formulas (image_y = height - world_y/resolution - 1)
    # already handle the world↔image mapping correctly.

    height, width = image_array.shape

    # Try to read companion YAML
    yaml_path = file_path.with_suffix(".yaml")
    yaml_meta = _try_read_pgm_yaml(yaml_path) if yaml_path.exists() else None

    # Try to read companion JSON (regions.json)
    json_path = file_path.with_name(file_path.stem + "_regions.json")
    json_meta = _try_read_pgm_json(json_path) if json_path.exists() else None

    if yaml_meta:
        resolution = yaml_meta.get("resolution", 0.05)
        origin = yaml_meta.get("origin", [0.0, 0.0, 0.0])
        origin_x, origin_y = float(origin[0]), float(origin[1])
    elif json_meta:
        resolution = json_meta.get("resolution", 0.05)
        origin = json_meta.get("origin", [0.0, 0.0, 0.0])
        origin_x, origin_y = float(origin[0]), float(origin[1])
    else:
        resolution = 0.05
        origin_x = 0.0
        origin_y = 0.0

    metadata = GridMetadata(
        origin_x=origin_x,
        origin_y=origin_y,
        resolution=resolution,
        width=width,
        height=height,
        extent_locked=True,
    )

    # Build an empty point-counts array (PGM-only mode has no point cloud)
    point_counts = np.zeros((height, width), dtype=np.uint32)

    return MapResult(
        image=image_array,
        point_counts=point_counts,
        metadata=metadata,
        filtered_points=0,
        z_min=0.0,
        z_max=0.0,
    )


def _try_read_pgm_yaml(path: Path) -> dict | None:
    """Minimal YAML parser for the simple PGM YAML files."""
    try:
        content = path.read_text(encoding="utf-8")
        result: dict = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                inner = value[1:-1]
                nums = [float(x.strip()) for x in inner.split(",") if x.strip()]
                result[key] = nums
            else:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
        return result
    except Exception:
        return None


def _try_read_pgm_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ─────────────────────────── 预处理 ───────────────────────────

def preprocess_point_cloud(
    points: np.ndarray,
    params: PreprocessParams,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Apply XY rotation/origin shift and optional outlier removal to a point cloud."""
    source = np.asarray(points, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("点云数据必须是形如 (N, 3) 的数组。")
    finite_mask = np.isfinite(source).all(axis=1)
    working = np.ascontiguousarray(source[finite_mask], dtype=np.float32)
    if working.size == 0:
        raise ValueError("点云中没有可用的有限值坐标。")

    raw_count = int(source.shape[0])
    finite_count = int(working.shape[0])
    rotation_center = _rotation_center_xy(working)
    transformed = transform_points_with_params(working, params, rotation_center)
    keep_mask = np.ones(transformed.shape[0], dtype=bool)

    box_removed = 0
    normalized_box = _normalize_keep_box(params.keep_box)
    if normalized_box is not None:
        box_mask = _build_keep_box_mask(transformed, normalized_box)
        box_removed = int(np.count_nonzero(keep_mask & ~box_mask))
        keep_mask &= box_mask

    crop_percent = float(params.crop_percent)
    crop_removed = 0
    if crop_percent > 0:
        crop_mask = _build_percentile_crop_mask(transformed, crop_percent)
        crop_removed = int(np.count_nonzero(keep_mask & ~crop_mask))
        keep_mask &= crop_mask

    component_removed = 0
    component_info: Dict[str, object] = {}
    if bool(params.keep_largest_component):
        if not np.any(keep_mask):
            raise ValueError("百分位裁剪后没有剩余点，无法继续保留最大主体区域。")
        component_mask, component_info = _build_largest_component_mask(
            transformed[keep_mask],
            resolution=float(params.component_resolution),
            min_cells=int(params.component_min_cells),
            min_points_per_cell=int(params.component_min_points_per_cell),
        )
        merged_component_mask = np.zeros_like(keep_mask)
        merged_component_mask[np.flatnonzero(keep_mask)] = component_mask
        component_removed = int(np.count_nonzero(keep_mask & ~merged_component_mask))
        keep_mask &= merged_component_mask

    processed = np.ascontiguousarray(transformed[keep_mask], dtype=np.float32)
    if processed.size == 0:
        raise ValueError("预处理后没有剩余点，请重新框选保留区域、降低裁剪比例或关闭去杂点。")

    info: Dict[str, object] = {
        "params": params.to_dict(),
        "rotation_center": [float(rotation_center[0]), float(rotation_center[1])],
        "raw_points": raw_count,
        "finite_points": finite_count,
        "kept_points": int(processed.shape[0]),
        "removed_points": int(finite_count - processed.shape[0]),
        "box_removed_points": box_removed,
        "keep_box": list(normalized_box) if normalized_box is not None else None,
        "crop_removed_points": crop_removed,
        "component_removed_points": component_removed,
        "output_bounds": _bounds_dict(processed),
    }
    info.update(component_info)
    return processed, info


def transform_points_with_params(
    points: np.ndarray,
    params: PreprocessParams,
    rotation_center: Tuple[float, float] | np.ndarray,
) -> np.ndarray:
    transformed = np.asarray(points, dtype=np.float32).copy()
    center = np.asarray(rotation_center, dtype=np.float64)
    theta = np.deg2rad(float(params.rotation_deg))
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    xy = transformed[:, :2].astype(np.float64, copy=False)
    shifted = xy - center
    rotated_x = shifted[:, 0] * cos_theta - shifted[:, 1] * sin_theta + center[0]
    rotated_y = shifted[:, 0] * sin_theta + shifted[:, 1] * cos_theta + center[1]
    transformed[:, 0] = rotated_x.astype(np.float32) - np.float32(params.origin_x)
    transformed[:, 1] = rotated_y.astype(np.float32) - np.float32(params.origin_y)
    return np.ascontiguousarray(transformed, dtype=np.float32)


def remap_xy_between_params(
    xy_points: np.ndarray,
    old_params: PreprocessParams,
    new_params: PreprocessParams,
    rotation_center: Tuple[float, float] | np.ndarray,
) -> np.ndarray:
    """Move annotation points from one processed coordinate system to another."""
    xy = np.asarray(xy_points, dtype=np.float64)
    if xy.size == 0:
        return xy.reshape((-1, 2)).astype(np.float32)
    center = np.asarray(rotation_center, dtype=np.float64)
    old_theta = np.deg2rad(float(old_params.rotation_deg))
    old_cos = float(np.cos(old_theta))
    old_sin = float(np.sin(old_theta))
    new_theta = np.deg2rad(float(new_params.rotation_deg))
    new_cos = float(np.cos(new_theta))
    new_sin = float(np.sin(new_theta))
    old_rotated = xy + np.array([old_params.origin_x, old_params.origin_y], dtype=np.float64)
    shifted_old = old_rotated - center
    raw_x = shifted_old[:, 0] * old_cos + shifted_old[:, 1] * old_sin + center[0]
    raw_y = -shifted_old[:, 0] * old_sin + shifted_old[:, 1] * old_cos + center[1]
    shifted_raw = np.column_stack((raw_x, raw_y)) - center
    new_x = shifted_raw[:, 0] * new_cos - shifted_raw[:, 1] * new_sin + center[0]
    new_y = shifted_raw[:, 0] * new_sin + shifted_raw[:, 1] * new_cos + center[1]
    new_xy = np.column_stack((new_x - new_params.origin_x, new_y - new_params.origin_y))
    return new_xy.astype(np.float32)


def _rotation_center_xy(points: np.ndarray) -> tuple[float, float]:
    return (
        float((np.min(points[:, 0]) + np.max(points[:, 0])) / 2.0),
        float((np.min(points[:, 1]) + np.max(points[:, 1])) / 2.0),
    )


def _normalize_keep_box(
    keep_box: tuple[float, float, float, float] | list[float] | None,
) -> tuple[float, float, float, float] | None:
    if keep_box is None:
        return None
    if len(keep_box) != 4:
        raise ValueError("框选保留区域必须包含 min_x, max_x, min_y, max_y 四个值。")
    a, b, c, d = [float(value) for value in keep_box]
    min_x, max_x = sorted((a, b))
    min_y, max_y = sorted((c, d))
    if not np.isfinite([min_x, max_x, min_y, max_y]).all():
        raise ValueError("框选保留区域包含无效坐标。")
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("框选保留区域太小，请重新拖拽一个矩形。")
    return (min_x, max_x, min_y, max_y)


def _build_keep_box_mask(
    points: np.ndarray,
    keep_box: tuple[float, float, float, float],
) -> np.ndarray:
    min_x, max_x, min_y, max_y = keep_box
    return (
        (points[:, 0] >= min_x) & (points[:, 0] <= max_x)
        & (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
    )


def _build_percentile_crop_mask(points: np.ndarray, crop_percent: float) -> np.ndarray:
    pct = min(max(float(crop_percent), 0.0), 45.0)
    if pct <= 0:
        return np.ones(points.shape[0], dtype=bool)
    lower = pct / 100.0
    upper = 1.0 - lower
    x_min, x_max = np.quantile(points[:, 0], [lower, upper])
    y_min, y_max = np.quantile(points[:, 1], [lower, upper])
    return (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    )


def _build_largest_component_mask(
    points: np.ndarray,
    resolution: float,
    min_cells: int,
    min_points_per_cell: int,
) -> tuple[np.ndarray, Dict[str, object]]:
    if resolution <= 0:
        raise ValueError("主体区域检测分辨率必须大于 0。")
    if min_points_per_cell < 1:
        min_points_per_cell = 1
    bounds = (
        float(np.min(points[:, 0])), float(np.max(points[:, 0])),
        float(np.min(points[:, 1])), float(np.max(points[:, 1])),
    )
    width = max(1, int(np.floor((bounds[1] - bounds[0]) / resolution)) + 1)
    height = max(1, int(np.floor((bounds[3] - bounds[2]) / resolution)) + 1)
    cell_count = int(width * height)
    if cell_count > 40_000_000:
        raise ValueError(
            f"主体区域检测需要 {cell_count:,} 个栅格，请调大地图分辨率或先使用百分位裁剪。"
        )
    x_indices = np.floor((points[:, 0] - bounds[0]) / resolution).astype(np.int64)
    y_indices = np.floor((points[:, 1] - bounds[2]) / resolution).astype(np.int64)
    linear = x_indices + y_indices * width
    counts = np.bincount(linear, minlength=cell_count)
    occupied_cells = np.flatnonzero(counts >= min_points_per_cell).astype(np.int64)
    if occupied_cells.size == 0:
        return np.zeros(points.shape[0], dtype=bool), {
            "component_count": 0, "largest_component_cells": 0,
            "occupied_cells_before_component_filter": 0,
        }
    largest = _largest_component_cells(occupied_cells, width)
    largest_cells = int(len(largest))
    if largest_cells < max(1, int(min_cells)):
        keep_all = np.ones(points.shape[0], dtype=bool)
        return keep_all, {
            "component_count": 1, "largest_component_cells": largest_cells,
            "occupied_cells_before_component_filter": int(occupied_cells.size),
            "component_filter_skipped": True,
        }
    largest_array = np.fromiter(largest, dtype=np.int64)
    keep_mask = np.isin(linear, largest_array)
    return keep_mask, {
        "component_count": None, "largest_component_cells": largest_cells,
        "occupied_cells_before_component_filter": int(occupied_cells.size),
        "component_filter_skipped": False,
    }


def _largest_component_cells(occupied_cells: np.ndarray, width: int) -> set[int]:
    remaining = set(int(cell) for cell in occupied_cells.tolist())
    largest: set[int] = set()
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            cell = stack.pop()
            x = cell % width
            neighbors = []
            if x > 0:
                neighbors.append(cell - 1)
            if x < width - 1:
                neighbors.append(cell + 1)
            neighbors.append(cell - width)
            neighbors.append(cell + width)
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        if len(component) > len(largest):
            largest = component
    return largest


def _bounds_dict(points: np.ndarray) -> Dict[str, float]:
    return {
        "min_x": float(np.min(points[:, 0])),
        "max_x": float(np.max(points[:, 0])),
        "min_y": float(np.min(points[:, 1])),
        "max_y": float(np.max(points[:, 1])),
        "min_z": float(np.min(points[:, 2])),
        "max_z": float(np.max(points[:, 2])),
    }


# ─────────────────────────── 地图核心 ───────────────────────────

class OccupancyMapBuilder:
    def __init__(self, xyz: np.ndarray):
        points = np.asarray(xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("点云数据必须是形如 (N, 3) 的数组。")
        if points.size == 0:
            raise ValueError("点云为空，无法生成地图。")
        finite_mask = np.isfinite(points).all(axis=1)
        self.points = np.ascontiguousarray(points[finite_mask], dtype=np.float32)
        if self.points.size == 0:
            raise ValueError("点云中没有可用的有限值坐标。")
        self.global_bounds = _compute_bounds(self.points)
        self.z_range = (
            float(np.min(self.points[:, 2])),
            float(np.max(self.points[:, 2])),
        )

    def build_map(
        self,
        z_min: float,
        z_max: float,
        resolution: float = 0.05,
        min_points_per_cell: int = 1,
        fixed_extent: bool = True,
    ) -> MapResult:
        if resolution <= 0:
            raise ValueError("分辨率必须大于 0。")
        if min_points_per_cell < 1:
            raise ValueError("单元格最少点数必须大于等于 1。")
        z_low, z_high = sorted((float(z_min), float(z_max)))
        z_mask = (self.points[:, 2] >= z_low) & (self.points[:, 2] <= z_high)
        filtered = self.points[z_mask]
        if fixed_extent or filtered.size == 0:
            bounds = self.global_bounds
        else:
            bounds = _compute_bounds(filtered)
        width = max(1, int(np.floor((bounds[1] - bounds[0]) / resolution)) + 1)
        height = max(1, int(np.floor((bounds[3] - bounds[2]) / resolution)) + 1)
        cell_count = width * height
        if cell_count > MAX_GRID_CELLS:
            raise ValueError(
                f"当前分辨率会生成 {cell_count:,} 个栅格，预览过大，请调高分辨率后重试。"
            )
        counts_flat = np.zeros(cell_count, dtype=np.uint32)
        if filtered.size:
            x_indices = np.floor((filtered[:, 0] - bounds[0]) / resolution).astype(np.int64)
            y_indices = np.floor((filtered[:, 1] - bounds[2]) / resolution).astype(np.int64)
            valid = (
                (x_indices >= 0) & (x_indices < width)
                & (y_indices >= 0) & (y_indices < height)
            )
            if np.any(valid):
                linear = x_indices[valid] + y_indices[valid] * width
                bincounts = np.bincount(linear, minlength=cell_count)
                counts_flat[: bincounts.shape[0]] = bincounts.astype(np.uint32, copy=False)
        counts_bottom = counts_flat.reshape((height, width))
        occupied_bottom = counts_bottom >= min_points_per_cell
        image_bottom = np.where(occupied_bottom, 0, 255).astype(np.uint8)
        return MapResult(
            image=np.flipud(image_bottom),
            point_counts=np.flipud(counts_bottom),
            metadata=GridMetadata(
                origin_x=float(bounds[0]), origin_y=float(bounds[2]),
                resolution=float(resolution), width=width, height=height,
                extent_locked=bool(fixed_extent),
            ),
            filtered_points=int(filtered.shape[0]),
            z_min=z_low, z_max=z_high,
        )


def resize_label_mask(mask: Optional[np.ndarray], target_size: Tuple[int, int]) -> np.ndarray:
    width, height = target_size
    if mask is None:
        return np.zeros((height, width), dtype=np.uint8)
    if mask.shape == (height, width):
        return mask.astype(np.uint8, copy=False)
    resized = Image.fromarray(mask.astype(np.uint8), mode="L").resize(
        (width, height), resample=Image.NEAREST
    )
    return np.asarray(resized, dtype=np.uint8)


def compose_preview_image(base_image: np.ndarray, label_mask: Optional[np.ndarray]) -> np.ndarray:
    base = np.asarray(base_image, dtype=np.uint8)
    rgb = np.repeat(base[:, :, None], 3, axis=2).astype(np.float32)
    if label_mask is None:
        return rgb.astype(np.uint8)
    overlay = np.zeros_like(rgb, dtype=np.float32)
    alpha = np.zeros(base.shape, dtype=np.float32)
    traversable = label_mask == TRAVERSABLE_LABEL
    blocked = label_mask == BLOCKED_LABEL
    overlay[traversable] = (40, 170, 95)
    overlay[blocked] = (210, 60, 50)
    alpha[traversable] = 0.45
    alpha[blocked] = 0.45
    blended = rgb * (1.0 - alpha[:, :, None]) + overlay * alpha[:, :, None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def normalize_label_mask_input(mask: np.ndarray, source_name: str = "") -> np.ndarray:
    data = np.asarray(mask, dtype=np.uint8)
    unique_values = set(np.unique(data).tolist())
    if unique_values.issubset({0, 1, 2}):
        return data
    if unique_values.issubset({0, 255}):
        normalized = np.zeros_like(data, dtype=np.uint8)
        lower_name = source_name.lower()
        if "blocked" in lower_name:
            normalized[data > 0] = BLOCKED_LABEL
        else:
            normalized[data > 0] = TRAVERSABLE_LABEL
        return normalized
    raise ValueError(
        "无法识别标注掩膜。请加载 *_regions_mask.png、*_traversable_mask.png 或 *_blocked_mask.png。"
    )


def apply_polygon_to_mask(
    mask: np.ndarray,
    polygon_points: list[tuple[int, int]],
    label_value: int,
) -> np.ndarray:
    if len(polygon_points) < 3:
        raise ValueError("至少需要 3 个顶点才能形成多边形。")
    mask_image = Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L")
    drawer = ImageDraw.Draw(mask_image)
    drawer.polygon([tuple(map(int, point)) for point in polygon_points], fill=int(label_value))
    return np.asarray(mask_image, dtype=np.uint8)


def save_export_bundle(
    output_prefix: str | Path,
    map_result: MapResult,
    label_mask: Optional[np.ndarray] = None,
    session_data: Optional[Dict[str, object]] = None,
    export_selection: Optional[Dict[str, bool]] = None,
) -> Dict[str, Path]:
    """Save export bundle, optionally selecting which files to export."""
    prefix_path = Path(output_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    if export_selection is None:
        export_selection = {
            "pgm": True, "yaml": True, "preview": True,
            "regions_mask": True, "traversable_mask": True,
            "blocked_mask": True, "regions_meta": True,
            "session": session_data is not None,
        }

    outputs: Dict[str, Path] = {}
    label_mask = resize_label_mask(label_mask, (map_result.metadata.width, map_result.metadata.height))

    if export_selection.get("pgm"):
        pgm_path = prefix_path.with_suffix(".pgm")
        Image.fromarray(map_result.image, mode="L").save(pgm_path)
        outputs["pgm"] = pgm_path

    if export_selection.get("yaml"):
        yaml_path = prefix_path.with_suffix(".yaml")
        yaml_content = _build_yaml_content(
            pgm_path.name if "pgm" in outputs else prefix_path.with_suffix(".pgm").name,
            map_result.metadata,
        )
        yaml_path.write_text(yaml_content, encoding="utf-8")
        outputs["yaml"] = yaml_path

    if export_selection.get("preview"):
        preview_path = prefix_path.with_name(prefix_path.name + "_preview.png")
        preview_image = compose_preview_image(map_result.image, label_mask)
        Image.fromarray(preview_image, mode="RGB").save(preview_path)
        outputs["preview"] = preview_path

    if export_selection.get("regions_mask"):
        regions_mask_path = prefix_path.with_name(prefix_path.name + "_regions_mask.png")
        Image.fromarray(label_mask, mode="L").save(regions_mask_path)
        outputs["regions_mask"] = regions_mask_path

    if export_selection.get("traversable_mask"):
        traversable_path = prefix_path.with_name(prefix_path.name + "_traversable_mask.png")
        Image.fromarray((label_mask == TRAVERSABLE_LABEL).astype(np.uint8) * 255, mode="L").save(
            traversable_path
        )
        outputs["traversable_mask"] = traversable_path

    if export_selection.get("blocked_mask"):
        blocked_path = prefix_path.with_name(prefix_path.name + "_blocked_mask.png")
        Image.fromarray((label_mask == BLOCKED_LABEL).astype(np.uint8) * 255, mode="L").save(
            blocked_path
        )
        outputs["blocked_mask"] = blocked_path

    if export_selection.get("regions_meta"):
        regions_meta_path = prefix_path.with_name(prefix_path.name + "_regions.json")
        regions_meta = {
            "version": 1,
            "image": (
                outputs.get("regions_mask", prefix_path.with_name(prefix_path.name + "_regions_mask.png")
                ).name
                if export_selection.get("regions_mask")
                else (prefix_path.name + "_regions_mask.png")
            ),
            "labels": {"0": "none", "1": "traversable", "2": "blocked"},
            "origin": [map_result.metadata.origin_x, map_result.metadata.origin_y, 0.0],
            "resolution": map_result.metadata.resolution,
            "width": map_result.metadata.width,
            "height": map_result.metadata.height,
            "row_order": "top_to_bottom",
            "world_coordinate_rule": {
                "x": "origin_x + col * resolution",
                "y": "origin_y + (height - 1 - row) * resolution",
            },
        }
        regions_meta_path.write_text(
            json.dumps(regions_meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        outputs["regions_meta"] = regions_meta_path

    if export_selection.get("session") and session_data:
        session_path = prefix_path.with_name(prefix_path.name + "_session.json")
        session_path.write_text(
            json.dumps(session_data, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        outputs["session"] = session_path

    return outputs


def _compute_bounds(points: np.ndarray) -> Tuple[float, float, float, float]:
    x_values = points[:, 0]
    y_values = points[:, 1]
    return (
        float(np.min(x_values)), float(np.max(x_values)),
        float(np.min(y_values)), float(np.max(y_values)),
    )


def _build_yaml_content(image_name: str, metadata: GridMetadata) -> str:
    return (
        f"image: {image_name}\n"
        f"resolution: {metadata.resolution:.6f}\n"
        f"origin: [{metadata.origin_x:.6f}, {metadata.origin_y:.6f}, 0.000000]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
    )


# ─────────────────────────── GUI 应用程序 ───────────────────────────

class PCDPGMToolApp:
    def __init__(self, root: tk.Tk, initial_path: str | None = None):
        self.root = root
        self.root.title("PCD PGM 标注工具")
        self.root.geometry("1420x900")
        self.root.minsize(1150, 760)

        self.point_cloud_path: Path | None = None
        self.raw_points: np.ndarray | None = None
        self.processed_points: np.ndarray | None = None
        self.preprocess_params = PreprocessParams()
        self.preprocess_info: dict[str, object] = {}
        self.selecting_keep_region = False
        self.keep_region_box: tuple[float, float, float, float] | None = None
        self.keep_region_drag_start_world: tuple[float, float] | None = None
        self.keep_region_drag_current_world: tuple[float, float] | None = None
        self.builder: OccupancyMapBuilder | None = None
        self.current_map = None
        self.is_pgm_mode = False  # True when loaded from PGM without PCD

        self.base_label_mask: np.ndarray | None = None
        self.label_mask: np.ndarray | None = None
        self.annotation_polygons: list[PolygonAnnotation] = []
        self.active_polygon_points_world: list[tuple[float, float]] = []
        self.active_polygon_label: str | None = None
        self.hover_world_point: tuple[float, float] | None = None

        self.preview_pil_image: Image.Image | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preview_layout: dict[str, int] | None = None
        self.preview_zoom = 1.0
        self.preview_min_zoom = 0.25
        self.preview_max_zoom = 8.0
        self.preview_pan_x = 0.0
        self.preview_pan_y = 0.0
        self.preview_pan_start: tuple[int, int] | None = None
        self.preview_pan_origin: tuple[float, float] | None = None
        self.controls_canvas: tk.Canvas | None = None
        self.controls_window_id: int | None = None

        self.update_job: str | None = None
        self._rotation_debounce_job: str | None = None
        self.render_busy = False
        self.render_serial = 0
        self.pending_render_request: tuple[int, dict[str, float | int | bool]] | None = None

        self.file_var = tk.StringVar(value="未加载文件")
        self.status_var = tk.StringVar(value="请选择一个 PCD 或 PGM 文件开始。")
        self.polygon_summary_var = tk.StringVar(value="当前没有已提交的多边形。")
        self.z_min_var = tk.DoubleVar(value=0.0)
        self.z_max_var = tk.DoubleVar(value=1.0)
        self.resolution_var = tk.DoubleVar(value=0.05)
        self.min_points_var = tk.IntVar(value=1)
        self.fixed_extent_var = tk.BooleanVar(value=True)
        self.draw_mode_var = tk.StringVar(value="traversable")
        self.rotation_deg_var = tk.DoubleVar(value=0.0)
        self.origin_x_var = tk.DoubleVar(value=0.0)
        self.origin_y_var = tk.DoubleVar(value=0.0)
        self.crop_percent_var = tk.DoubleVar(value=0.0)
        self.keep_largest_component_var = tk.BooleanVar(value=False)
        self.component_min_cells_var = tk.IntVar(value=3)
        self.keep_region_summary_var = tk.StringVar(value="框选保留区域：未设置")
        self.export_processed_pcd_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._draw_placeholder("请选择一个 PCD 或 PGM 文件开始。")

        self.root.bind("<Return>", lambda _event: self._commit_active_polygon())
        self.root.bind("<BackSpace>", lambda _event: self._undo_last_polygon_point())
        self.root.bind("<Escape>", lambda _event: self._cancel_active_polygon())
        self.root.bind("<Control-s>", lambda _event: self._export_bundle())
        self.root.bind("<Control-S>", lambda _event: self._export_bundle())
        self.root.bind("<Control-0>", lambda _event: self._reset_preview_view())

        if initial_path:
            p = Path(initial_path)
            if p.suffix.lower() == ".pgm":
                self.load_pgm_async(p)
            else:
                self.load_pcd_async(p)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        sidebar = ttk.Frame(self.root)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(0, weight=1)

        self.controls_canvas = tk.Canvas(sidebar, highlightthickness=0, width=390)
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")
        controls_scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=self.controls_canvas.yview)
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        preview = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        preview.grid(row=0, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.controls_canvas, padding=12)
        self.controls_window_id = self.controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", self._on_controls_frame_configure)
        self.controls_canvas.bind("<Configure>", self._on_controls_canvas_configure)
        self.controls_canvas.bind("<Enter>", self._bind_controls_mousewheel)
        self.controls_canvas.bind("<Leave>", self._unbind_controls_mousewheel)

        self._build_controls(controls)
        self._build_preview(preview)
        self._build_quick_actions()

    def _build_controls(self, parent: ttk.Frame) -> None:
        io_frame = ttk.LabelFrame(parent, text="文件", padding=10)
        io_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        io_frame.columnconfigure(0, weight=1)

        ttk.Button(io_frame, text="打开 PCD", command=self._choose_pcd_file).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(io_frame, text="打开 PGM", command=self._choose_pgm_file).grid(
            row=1, column=0, sticky="ew", pady=(4, 0)
        )
        ttk.Label(io_frame, textvariable=self.file_var, wraplength=320).grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Button(io_frame, text="加载已有标注", command=self._load_annotation_mask).grid(
            row=3, column=0, sticky="ew", pady=(10, 0)
        )

        map_frame = ttk.LabelFrame(parent, text="地图参数", padding=10)
        map_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        map_frame.columnconfigure(1, weight=1)
        map_frame.columnconfigure(2, weight=1)

        ttk.Label(map_frame, text="Z 最小值").grid(row=0, column=0, sticky="w")
        self.z_min_entry = ttk.Entry(map_frame, textvariable=self.z_min_var, width=10)
        self.z_min_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.z_min_scale = ttk.Scale(
            map_frame, orient="horizontal", from_=-10.0, to=10.0,
            variable=self.z_min_var, command=self._on_scale_change,
        )
        self.z_min_scale.grid(row=0, column=2, sticky="ew")

        ttk.Label(map_frame, text="Z 最大值").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.z_max_entry = ttk.Entry(map_frame, textvariable=self.z_max_var, width=10)
        self.z_max_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        self.z_max_scale = ttk.Scale(
            map_frame, orient="horizontal", from_=-10.0, to=10.0,
            variable=self.z_max_var, command=self._on_scale_change,
        )
        self.z_max_scale.grid(row=1, column=2, sticky="ew", pady=(8, 0))

        ttk.Label(map_frame, text="地图分辨率").grid(row=2, column=0, sticky="w", pady=(12, 0))
        resolution_spin = ttk.Spinbox(
            map_frame, from_=0.01, to=1.0, increment=0.01,
            textvariable=self.resolution_var, width=10,
            command=self.schedule_map_update,
        )
        resolution_spin.grid(row=2, column=1, sticky="w", pady=(12, 0))

        ttk.Label(map_frame, text="单格最少点数").grid(row=3, column=0, sticky="w", pady=(8, 0))
        min_points_spin = ttk.Spinbox(
            map_frame, from_=1, to=50, increment=1,
            textvariable=self.min_points_var, width=10,
            command=self.schedule_map_update,
        )
        min_points_spin.grid(row=3, column=1, sticky="w", pady=(8, 0))

        ttk.Checkbutton(
            map_frame, text="锁定全局范围，调阈值时保持多边形位置不漂移",
            variable=self.fixed_extent_var, command=self.schedule_map_update,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 0))

        preprocess_frame = ttk.LabelFrame(parent, text="点云预处理（影响 modified.pcd 和 PGM）", padding=10)
        preprocess_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        preprocess_frame.columnconfigure(1, weight=1)

        ttk.Label(preprocess_frame, text="旋转角度 °").grid(row=0, column=0, sticky="w")
        self.rotation_entry = ttk.Entry(preprocess_frame, textvariable=self.rotation_deg_var, width=10)
        self.rotation_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.rotation_scale = ttk.Scale(
            preprocess_frame, orient="horizontal", from_=-180.0, to=180.0,
            variable=self.rotation_deg_var, command=self._on_rotation_change,
        )
        self.rotation_scale.grid(row=0, column=2, sticky="ew", padx=(8, 0))
        preprocess_frame.columnconfigure(2, weight=1)

        ttk.Label(preprocess_frame, text="新原点 X").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.origin_x_entry = ttk.Entry(preprocess_frame, textvariable=self.origin_x_var, width=10)
        self.origin_x_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(preprocess_frame, text="新原点 Y").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.origin_y_entry = ttk.Entry(preprocess_frame, textvariable=self.origin_y_var, width=10)
        self.origin_y_entry.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Button(
            preprocess_frame, text="框选保留主体区域",
            command=self._begin_select_keep_region,
        ).grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(
            preprocess_frame, text="清除框选保留区域",
            command=self._clear_keep_region,
        ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(
            preprocess_frame, textvariable=self.keep_region_summary_var, wraplength=320,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Checkbutton(
            preprocess_frame, text="删除外围杂点：只保留最大主体区域",
            variable=self.keep_largest_component_var,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(10, 0))

        ttk.Label(preprocess_frame, text="主体最小格数").grid(row=7, column=0, sticky="w", pady=(8, 0))
        self.component_min_cells_entry = ttk.Spinbox(
            preprocess_frame, from_=1, to=100000, increment=1,
            textvariable=self.component_min_cells_var, width=10,
        )
        self.component_min_cells_entry.grid(row=7, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(preprocess_frame, text="外围百分位裁剪 %").grid(row=8, column=0, sticky="w", pady=(8, 0))
        self.crop_percent_entry = ttk.Entry(preprocess_frame, textvariable=self.crop_percent_var, width=10)
        self.crop_percent_entry.grid(row=8, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Checkbutton(
            preprocess_frame, text="导出修改后的 PCD（仅 PCD 模式有效）",
            variable=self.export_processed_pcd_var, state="disabled",
        ).grid(row=9, column=0, columnspan=3, sticky="w", pady=(10, 0))

        ttk.Button(
            preprocess_frame, text="应用预处理并刷新",
            command=self._apply_preprocess_async,
        ).grid(row=10, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        ttk.Button(preprocess_frame, text="重置预处理", command=self._reset_preprocess).grid(
            row=11, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )
        ttk.Label(
            preprocess_frame,
            text="旋转角度会自动应用并刷新，其他参数需点击「应用预处理并刷新」。",
            wraplength=320,
        ).grid(row=12, column=0, columnspan=3, sticky="w", pady=(10, 0))

        draw_frame = ttk.LabelFrame(parent, text="多边形标注", padding=10)
        draw_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        draw_frame.columnconfigure(0, weight=1)

        ttk.Radiobutton(draw_frame, text="可通行", value="traversable", variable=self.draw_mode_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Radiobutton(draw_frame, text="不可通行", value="blocked", variable=self.draw_mode_var).grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Radiobutton(draw_frame, text="擦除区域", value="erase", variable=self.draw_mode_var).grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )

        ttk.Label(draw_frame, textvariable=self.polygon_summary_var, wraplength=320).grid(
            row=3, column=0, sticky="w", pady=(12, 0)
        )

        self.polygon_listbox = tk.Listbox(draw_frame, height=7, exportselection=False)
        self.polygon_listbox.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.polygon_listbox.bind("<<ListboxSelect>>", lambda _event: self._render_canvas())

        ttk.Button(draw_frame, text="完成当前多边形", command=self._commit_active_polygon).grid(
            row=5, column=0, sticky="ew", pady=(10, 0)
        )
        ttk.Button(draw_frame, text="撤销上一个顶点", command=self._undo_last_polygon_point).grid(
            row=6, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(draw_frame, text="取消当前多边形", command=self._cancel_active_polygon).grid(
            row=7, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(draw_frame, text="继续编辑选中多边形", command=self._edit_selected_polygon).grid(
            row=8, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(draw_frame, text="删除选中多边形", command=self._delete_selected_polygon).grid(
            row=9, column=0, sticky="ew", pady=(8, 0)
        )

        ttk.Label(
            draw_frame,
            text=(
                "左键逐点加顶点，右键或 Enter 完成当前多边形。"
                "点云预处理里可进入框选模式，拖拽矩形选择要保留的主体点云区域。"
                "鼠标滚轮可缩放预览，中键拖动可平移，Ctrl+0 或中键双击恢复适配窗口。"
                "Backspace 撤销一个点，Esc 取消当前多边形。"
                "选中列表里的多边形后，可以继续补点再重新提交。"
            ),
            wraplength=320,
        ).grid(row=10, column=0, sticky="w", pady=(10, 0))

        action_frame = ttk.LabelFrame(parent, text="操作", padding=10)
        action_frame.grid(row=4, column=0, sticky="ew")
        action_frame.columnconfigure(0, weight=1)

        ttk.Button(action_frame, text="刷新预览", command=self.request_map_render).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(action_frame, text="清空全部标注", command=self._clear_annotations).grid(
            row=1, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(action_frame, text="导出文件...", command=self._export_bundle).grid(
            row=2, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Label(action_frame, textvariable=self.status_var, wraplength=320).grid(
            row=3, column=0, sticky="w", pady=(12, 0)
        )

        for widget in (self.z_min_entry, self.z_max_entry, resolution_spin, min_points_spin):
            widget.bind("<Return>", lambda _event: self.schedule_map_update())
            widget.bind("<FocusOut>", lambda _event: self.schedule_map_update())

    def _build_preview(self, parent: ttk.Frame) -> None:
        self.canvas = tk.Canvas(parent, bg="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._render_canvas())
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_left_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_left_release)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind("<Button-4>", self._on_canvas_mousewheel)
        self.canvas.bind("<Button-5>", self._on_canvas_mousewheel)
        self.canvas.bind("<ButtonPress-2>", self._on_canvas_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_canvas_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_canvas_pan_end)
        self.canvas.bind("<Double-Button-2>", lambda _event: self._reset_preview_view())

    def _build_quick_actions(self) -> None:
        footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        footer.grid(row=1, column=0, columnspan=2, sticky="ew")
        footer.columnconfigure(0, weight=0)
        footer.columnconfigure(1, weight=0)
        footer.columnconfigure(2, weight=0)
        footer.columnconfigure(3, weight=0)
        footer.columnconfigure(4, weight=1)

        ttk.Button(footer, text="打开 PCD", command=self._choose_pcd_file).grid(row=0, column=0, sticky="w")
        ttk.Button(footer, text="打开 PGM", command=self._choose_pgm_file).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        ttk.Button(footer, text="刷新预览", command=self.request_map_render).grid(
            row=0, column=2, sticky="w", padx=(8, 0)
        )
        ttk.Button(footer, text="导出文件...", command=self._export_bundle).grid(
            row=0, column=3, sticky="w", padx=(8, 0)
        )
        ttk.Label(footer, text="`Ctrl+S` 导出，`Ctrl+0` 恢复预览缩放",).grid(
            row=0, column=4, sticky="e", padx=(12, 0)
        )

    def _on_controls_frame_configure(self, _event: tk.Event) -> None:
        if self.controls_canvas is None:
            return
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _on_controls_canvas_configure(self, event: tk.Event) -> None:
        if self.controls_canvas is None or self.controls_window_id is None:
            return
        self.controls_canvas.itemconfigure(self.controls_window_id, width=event.width)

    def _bind_controls_mousewheel(self, _event: tk.Event) -> None:
        self.root.bind_all("<MouseWheel>", self._on_controls_mousewheel)
        self.root.bind_all("<Button-4>", self._on_controls_mousewheel)
        self.root.bind_all("<Button-5>", self._on_controls_mousewheel)

    def _unbind_controls_mousewheel(self, _event: tk.Event) -> None:
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_controls_mousewheel(self, event: tk.Event) -> None:
        if self.controls_canvas is None:
            return
        if getattr(event, "num", None) == 4:
            self.controls_canvas.yview_scroll(-1, "units")
            return
        if getattr(event, "num", None) == 5:
            self.controls_canvas.yview_scroll(1, "units")
            return
        delta = int(-event.delta / 120) if getattr(event, "delta", 0) else 0
        if delta != 0:
            self.controls_canvas.yview_scroll(delta, "units")

    def _collect_preprocess_params(self) -> PreprocessParams:
        try:
            rotation_deg = float(self.rotation_deg_var.get())
            origin_x = float(self.origin_x_var.get())
            origin_y = float(self.origin_y_var.get())
            crop_percent = float(self.crop_percent_var.get())
            component_resolution = float(self.resolution_var.get())
            component_min_cells = int(self.component_min_cells_var.get())
        except (tk.TclError, ValueError) as exc:
            raise ValueError("请检查旋转角度、原点、裁剪比例和主体格数是否为有效数字。") from exc
        return PreprocessParams(
            rotation_deg=rotation_deg, origin_x=origin_x, origin_y=origin_y,
            crop_percent=max(0.0, min(crop_percent, 45.0)),
            keep_box=self.keep_region_box,
            keep_largest_component=bool(self.keep_largest_component_var.get()),
            component_resolution=max(component_resolution, 1e-6),
            component_min_cells=max(1, component_min_cells),
            component_min_points_per_cell=1,
        )

    def _begin_select_keep_region(self) -> None:
        if self.current_map is None:
            messagebox.showinfo("提示", "请先加载文件并生成预览。")
            return
        if self.is_pgm_mode:
            messagebox.showinfo("提示", "PGM 模式下不支持预处理操作。")
            return
        self.selecting_keep_region = True
        self.keep_region_drag_start_world = None
        self.keep_region_drag_current_world = None
        self.canvas.configure(cursor="crosshair")
        self.status_var.set("请在右侧预览图上按住左键拖拽矩形；松开后只保留矩形内的 PCD 点。")
        self._draw_polygon_overlay()

    def _clear_keep_region(self) -> None:
        self.keep_region_box = None
        self.keep_region_drag_start_world = None
        self.keep_region_drag_current_world = None
        self.selecting_keep_region = False
        self._update_keep_region_summary()
        if self.current_map is not None and not self.is_pgm_mode:
            self.canvas.configure(cursor="")
            self.status_var.set("已清除框选保留区域，正在重新生成完整处理点云。")
            self._apply_preprocess_async()

    def _update_keep_region_summary(self) -> None:
        if self.keep_region_box is None:
            self.keep_region_summary_var.set("框选保留区域：未设置")
            return
        min_x, max_x, min_y, max_y = self.keep_region_box
        self.keep_region_summary_var.set(
            f"框选保留区域：X {min_x:.3f} ~ {max_x:.3f}，Y {min_y:.3f} ~ {max_y:.3f}"
        )

    def _reset_preprocess(self) -> None:
        if self.raw_points is None:
            return
        if self.is_pgm_mode:
            return
        self.rotation_deg_var.set(0.0)
        self.origin_x_var.set(0.0)
        self.origin_y_var.set(0.0)
        self.crop_percent_var.set(0.0)
        self.keep_region_box = None
        self.keep_region_drag_start_world = None
        self.keep_region_drag_current_world = None
        self.selecting_keep_region = False
        self._update_keep_region_summary()
        self.keep_largest_component_var.set(False)
        self.component_min_cells_var.set(3)
        self._apply_preprocess_async()

    def _apply_preprocess_async(self) -> None:
        if self.raw_points is None:
            messagebox.showinfo("提示", "请先打开 PCD 文件。")
            return
        if self.is_pgm_mode:
            messagebox.showinfo("提示", "PGM 模式下不支持预处理操作。")
            return
        try:
            new_params = self._collect_preprocess_params()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        old_params = self.preprocess_params
        if self.keep_region_box is not None:
            remapped_box = self._remap_keep_region_box_between_preprocess(old_params, new_params)
            if remapped_box is not None:
                self.keep_region_box = remapped_box
                new_params.keep_box = remapped_box
                self._update_keep_region_summary()
        raw_points = self.raw_points
        self.status_var.set("正在后台应用点云预处理...")

        def worker() -> None:
            try:
                processed, preprocess_info = preprocess_point_cloud(raw_points, new_params)
                builder = OccupancyMapBuilder(processed)
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._on_preprocess_failed(exc))
                return
            self.root.after(
                0,
                lambda: self._on_preprocess_success(
                    old_params, new_params, processed, preprocess_info, builder
                ),
            )
        threading.Thread(target=worker, daemon=True).start()

    def _on_preprocess_failed(self, exc: Exception) -> None:
        self.status_var.set(f"点云预处理失败: {exc}")
        messagebox.showerror("点云预处理失败", str(exc))

    def _on_preprocess_success(
        self,
        old_params: PreprocessParams,
        new_params: PreprocessParams,
        processed_points: np.ndarray,
        preprocess_info: dict[str, object],
        builder: OccupancyMapBuilder,
    ) -> None:
        self._remap_annotations_between_preprocess(old_params, new_params)
        self.base_label_mask = None
        self.processed_points = processed_points
        self.preprocess_params = new_params
        self.keep_region_box = new_params.keep_box
        self._update_keep_region_summary()
        self.preprocess_info = preprocess_info
        self.builder = builder
        self.current_map = None
        self.preview_pil_image = None
        self.preview_layout = None
        self.selecting_keep_region = False
        self.keep_region_drag_start_world = None
        self.keep_region_drag_current_world = None
        self.canvas.configure(cursor="")
        self._reset_preview_view(redraw=False)
        z_min, z_max = builder.z_range
        current_z_min = min(max(float(self.z_min_var.get()), z_min), z_max)
        current_z_max = min(max(float(self.z_max_var.get()), z_min), z_max)
        if current_z_min > current_z_max:
            current_z_min, current_z_max = z_min, z_max
        self.z_min_scale.configure(from_=z_min, to=z_max)
        self.z_max_scale.configure(from_=z_min, to=z_max)
        self.z_min_var.set(current_z_min)
        self.z_max_var.set(current_z_max)
        removed = int(preprocess_info.get("removed_points", 0))
        kept = int(preprocess_info.get("kept_points", processed_points.shape[0]))
        self.status_var.set(f"点云预处理完成：保留 {kept:,} 个点，删除 {removed:,} 个点。")
        self._refresh_polygon_list()
        self.request_map_render()

    def _remap_annotations_between_preprocess(
        self, old_params: PreprocessParams, new_params: PreprocessParams,
    ) -> None:
        if self.raw_points is None:
            return
        if (old_params.rotation_deg == new_params.rotation_deg
                and old_params.origin_x == new_params.origin_x
                and old_params.origin_y == new_params.origin_y):
            return
        rotation_center = _rotation_center_xy(self.raw_points)
        def remap(points_world: list[tuple[float, float]]) -> list[tuple[float, float]]:
            if not points_world:
                return []
            xy = np.asarray(points_world, dtype=np.float32)
            mapped = remap_xy_between_params(xy, old_params, new_params, rotation_center)
            return [(float(x), float(y)) for x, y in mapped]
        for polygon in self.annotation_polygons:
            polygon.points_world = remap(polygon.points_world)
        self.active_polygon_points_world = remap(self.active_polygon_points_world)
        if self.hover_world_point is not None:
            mapped_hover = remap([self.hover_world_point])
            self.hover_world_point = mapped_hover[0] if mapped_hover else None

    def _remap_keep_region_box_between_preprocess(
        self, old_params: PreprocessParams, new_params: PreprocessParams,
    ) -> tuple[float, float, float, float] | None:
        if self.raw_points is None or self.keep_region_box is None:
            return self.keep_region_box
        if (old_params.rotation_deg == new_params.rotation_deg
                and old_params.origin_x == new_params.origin_x
                and old_params.origin_y == new_params.origin_y):
            return self.keep_region_box
        min_x, max_x, min_y, max_y = self.keep_region_box
        corners = np.asarray(
            [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)], dtype=np.float32,
        )
        rotation_center = _rotation_center_xy(self.raw_points)
        mapped = remap_xy_between_params(corners, old_params, new_params, rotation_center)
        return (
            float(np.min(mapped[:, 0])), float(np.max(mapped[:, 0])),
            float(np.min(mapped[:, 1])), float(np.max(mapped[:, 1])),
        )

    def _choose_pcd_file(self) -> None:
        file_name = filedialog.askopenfilename(
            title="选择 PCD 文件",
            filetypes=[("PCD files", "*.pcd"), ("All files", "*.*")],
        )
        if file_name:
            self.load_pcd_async(Path(file_name))

    def _choose_pgm_file(self) -> None:
        file_name = filedialog.askopenfilename(
            title="选择 PGM 文件",
            filetypes=[("PGM files", "*.pgm"), ("All files", "*.*")],
        )
        if file_name:
            self.load_pgm_async(Path(file_name))

    def load_pcd_async(self, file_path: Path) -> None:
        self.file_var.set(str(file_path))
        self.status_var.set("正在读取点云，请稍候...")
        self.is_pgm_mode = False

        def worker() -> None:
            try:
                points = load_pcd_xyz(file_path)
                params = PreprocessParams()
                processed, preprocess_info = preprocess_point_cloud(points, params)
                builder = OccupancyMapBuilder(processed)
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._on_load_failed(exc))
                return
            self.root.after(
                0,
                lambda: self._on_load_success(
                    file_path, points, processed, params, preprocess_info, builder,
                ),
            )
        threading.Thread(target=worker, daemon=True).start()

    def load_pgm_async(self, file_path: Path) -> None:
        """Load a PGM file directly into map mode for annotation."""
        self.file_var.set(str(file_path))
        self.status_var.set("正在读取 PGM 地图，请稍候...")
        self.is_pgm_mode = True

        def worker() -> None:
            try:
                map_result = load_pgm_to_map_result(file_path)
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._on_load_failed(exc))
                return
            self.root.after(0, lambda: self._on_pgm_load_success(file_path, map_result))

        threading.Thread(target=worker, daemon=True).start()

    def _on_load_failed(self, exc: Exception) -> None:
        self.status_var.set("读取失败。")
        messagebox.showerror("读取失败", str(exc))

    def _on_load_success(
        self,
        file_path: Path,
        raw_points: np.ndarray,
        processed_points: np.ndarray,
        preprocess_params: PreprocessParams,
        preprocess_info: dict[str, object],
        builder: OccupancyMapBuilder,
    ) -> None:
        self.point_cloud_path = file_path
        self.raw_points = raw_points
        self.processed_points = processed_points
        self.preprocess_params = preprocess_params
        self.preprocess_info = preprocess_info
        self.builder = builder
        self.is_pgm_mode = False
        self.current_map = None
        self.preview_pil_image = None
        self.preview_layout = None
        self._reset_preview_view(redraw=False)
        self.selecting_keep_region = False
        self.keep_region_drag_start_world = None
        self.keep_region_drag_current_world = None
        self.rotation_deg_var.set(preprocess_params.rotation_deg)
        self.origin_x_var.set(preprocess_params.origin_x)
        self.origin_y_var.set(preprocess_params.origin_y)
        self.crop_percent_var.set(preprocess_params.crop_percent)
        self.keep_region_box = preprocess_params.keep_box
        self._update_keep_region_summary()
        self.keep_largest_component_var.set(preprocess_params.keep_largest_component)
        self.component_min_cells_var.set(preprocess_params.component_min_cells)
        self.base_label_mask = None
        self.label_mask = None
        self.annotation_polygons = []
        self.active_polygon_points_world = []
        self.active_polygon_label = None
        self.hover_world_point = None
        self._refresh_polygon_list()
        z_min, z_max = builder.z_range
        self.z_min_scale.configure(from_=z_min, to=z_max)
        self.z_max_scale.configure(from_=z_min, to=z_max)
        self.z_min_var.set(z_min)
        self.z_max_var.set(z_max)
        self.status_var.set(
            f"已加载 {raw_points.shape[0]:,} 个原始点，当前处理后 {builder.points.shape[0]:,} 个点，"
            f"Z 范围 {z_min:.3f} 到 {z_max:.3f}。"
        )
        self.request_map_render()

    def _on_pgm_load_success(self, file_path: Path, map_result: MapResult) -> None:
        """Set up the app state for PGM-only mode (no point cloud)."""
        self.point_cloud_path = file_path
        self.raw_points = None
        self.processed_points = None
        self.preprocess_params = PreprocessParams()
        self.preprocess_info = {}
        self.builder = None
        self.current_map = map_result
        self.is_pgm_mode = True
        self._refresh_preview_image()
        self.selecting_keep_region = False
        self.keep_region_drag_start_world = None
        self.keep_region_drag_current_world = None
        self.base_label_mask = None
        self.label_mask = None
        self.annotation_polygons = []
        self.active_polygon_points_world = []
        self.active_polygon_label = None
        self.hover_world_point = None
        self._refresh_polygon_list()

        meta = map_result.metadata
        self.z_min_var.set(0.0)
        self.z_max_var.set(0.0)
        self.resolution_var.set(meta.resolution)
        self.fixed_extent_var.set(True)

        self.status_var.set(
            f"已加载 PGM 地图: {meta.width} x {meta.height} 像素，分辨率 {meta.resolution:.4f}。"
            f" 可直接在地图上绘制通行区域标注。"
        )
        self._render_canvas()

    def _on_scale_change(self, _value: str) -> None:
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
            self.update_job = None
        self.request_map_render()

    def _on_rotation_change(self, _value: str) -> None:
        """旋转角度变化时自动触发预处理刷新（PCD 模式）。"""
        if self.raw_points is None or self.is_pgm_mode:
            return
        if self._rotation_debounce_job is not None:
            self.root.after_cancel(self._rotation_debounce_job)
        self._rotation_debounce_job = self.root.after(200, self._apply_preprocess_async)

    def schedule_map_update(self) -> None:
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(180, self.request_map_render)

    def _collect_render_params(self) -> dict[str, float | int | bool]:
        return {
            "z_min": float(self.z_min_var.get()),
            "z_max": float(self.z_max_var.get()),
            "resolution": float(self.resolution_var.get()),
            "min_points_per_cell": int(self.min_points_var.get()),
            "fixed_extent": bool(self.fixed_extent_var.get()),
        }

    def request_map_render(self) -> None:
        self.update_job = None
        if self.builder is None:
            # In PGM mode, just re-render the existing map
            if self.is_pgm_mode and self.current_map is not None:
                self._refresh_preview_image()
            return
        try:
            params = self._collect_render_params()
        except (tk.TclError, ValueError):
            self.status_var.set("参数暂时无效，请检查 Z 阈值、分辨率和点数设置。")
            return
        self.render_serial += 1
        self.pending_render_request = (self.render_serial, params)
        if self.render_busy:
            self.status_var.set("正在刷新预览，最新参数会在当前任务结束后自动显示。")
            return
        self._start_pending_render()

    def _start_pending_render(self) -> None:
        if self.builder is None or self.pending_render_request is None:
            return
        serial, params = self.pending_render_request
        self.pending_render_request = None
        self.render_busy = True
        self.status_var.set("正在后台刷新预览...")
        def worker() -> None:
            try:
                result = self.builder.build_map(**params)
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._on_render_failed(serial, exc))
                return
            self.root.after(0, lambda: self._on_render_success(serial, result))
        threading.Thread(target=worker, daemon=True).start()

    def _on_render_failed(self, serial: int, exc: Exception) -> None:
        self.render_busy = False
        if serial == self.render_serial:
            self.status_var.set(f"地图生成失败: {exc}")
        if self.pending_render_request is not None:
            self._start_pending_render()

    def _on_render_success(self, serial: int, result) -> None:
        self.render_busy = False
        if serial != self.render_serial:
            if self.pending_render_request is not None:
                self._start_pending_render()
            return
        self.current_map = result
        self._refresh_preview_image()
        self.status_var.set(
            f"预览已更新: {result.metadata.width} x {result.metadata.height} 像素，"
            f"阈值内点数 {result.filtered_points:,}。"
        )
        if self.pending_render_request is not None:
            self._start_pending_render()

    def _refresh_preview_image(self) -> None:
        if self.current_map is None:
            self.preview_pil_image = None
            self._draw_placeholder("请选择一个 PCD 或 PGM 文件开始。")
            return
        self.label_mask = self._build_annotation_mask()
        preview = compose_preview_image(self.current_map.image, self.label_mask)
        self.preview_pil_image = Image.fromarray(preview, mode="RGB")
        self._render_canvas()

    def _build_annotation_mask(self) -> np.ndarray:
        if self.current_map is None:
            return np.zeros((1, 1), dtype=np.uint8)
        mask = resize_label_mask(
            self.base_label_mask,
            (self.current_map.metadata.width, self.current_map.metadata.height),
        )
        for polygon in self.annotation_polygons:
            image_points = []
            for world_point in polygon.points_world:
                image_point = self._world_to_image_point(world_point)
                if image_point is None:
                    continue
                image_points.append(
                    (int(round(image_point[0])), int(round(image_point[1])))
                )
            if len(image_points) >= 3:
                mask = apply_polygon_to_mask(mask, image_points, LABEL_TO_VALUE[polygon.label])
        return mask

    def _render_canvas(self) -> None:
        self.canvas.delete("all")
        if self.preview_pil_image is None:
            self._draw_placeholder("请选择一个 PCD 或 PGM 文件开始。")
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        img_width, img_height = self.preview_pil_image.size
        base_scale = min(canvas_width / img_width, canvas_height / img_height)
        base_scale = max(base_scale, 0.01)
        scale = max(base_scale * self.preview_zoom, 0.01)
        display_width = max(1, int(round(img_width * scale)))
        display_height = max(1, int(round(img_height * scale)))
        self._clamp_preview_pan(canvas_width, canvas_height, display_width, display_height)
        offset_x = int(round((canvas_width - display_width) / 2 + self.preview_pan_x))
        offset_y = int(round((canvas_height - display_height) / 2 + self.preview_pan_y))
        resized = self.preview_pil_image.resize(
            (display_width, display_height), resample=Image.NEAREST,
        )
        self.preview_photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.preview_photo)
        self.canvas.create_rectangle(
            offset_x, offset_y, offset_x + display_width, offset_y + display_height,
            outline="#444444",
        )
        self.preview_layout = {
            "offset_x": offset_x, "offset_y": offset_y,
            "display_width": display_width, "display_height": display_height,
            "image_width": img_width, "image_height": img_height,
            "base_scale_ppm": int(round(base_scale * 1_000_000)),
            "zoom_ppm": int(round(self.preview_zoom * 1_000_000)),
        }
        self._draw_polygon_overlay()

    def _clamp_preview_pan(
        self, canvas_width: int, canvas_height: int,
        display_width: int, display_height: int,
    ) -> None:
        base_offset_x = (canvas_width - display_width) / 2
        base_offset_y = (canvas_height - display_height) / 2
        if display_width <= canvas_width:
            self.preview_pan_x = 0.0
        else:
            min_pan_x = (canvas_width - display_width) - base_offset_x
            max_pan_x = -base_offset_x
            self.preview_pan_x = min(max(self.preview_pan_x, min_pan_x), max_pan_x)
        if display_height <= canvas_height:
            self.preview_pan_y = 0.0
        else:
            min_pan_y = (canvas_height - display_height) - base_offset_y
            max_pan_y = -base_offset_y
            self.preview_pan_y = min(max(self.preview_pan_y, min_pan_y), max_pan_y)

    def _reset_preview_view(self, redraw: bool = True) -> None:
        self.preview_zoom = 1.0
        self.preview_pan_x = 0.0
        self.preview_pan_y = 0.0
        self.preview_pan_start = None
        self.preview_pan_origin = None
        if redraw and self.preview_pil_image is not None:
            self._render_canvas()
            self.status_var.set("预览缩放已恢复为适配窗口。")

    def _canvas_xy_to_image_point(
        self, x_coord: int, y_coord: int, *, clamp: bool = False,
    ) -> tuple[float, float] | None:
        if self.preview_layout is None:
            return None
        left = self.preview_layout["offset_x"]
        top = self.preview_layout["offset_y"]
        width = self.preview_layout["display_width"]
        height = self.preview_layout["display_height"]
        image_width = self.preview_layout["image_width"]
        image_height = self.preview_layout["image_height"]
        if width <= 0 or height <= 0:
            return None
        if not clamp and not (left <= x_coord < left + width and top <= y_coord < top + height):
            return None
        image_x = (x_coord - left) * image_width / width
        image_y = (y_coord - top) * image_height / height
        if clamp:
            image_x = min(max(image_x, 0.0), max(image_width - 1.0, 0.0))
            image_y = min(max(image_y, 0.0), max(image_height - 1.0, 0.0))
        return (image_x, image_y)

    def _on_canvas_mousewheel(self, event: tk.Event) -> str | None:
        if self.preview_pil_image is None or self.preview_layout is None:
            return None
        if getattr(event, "num", None) == 4:
            zoom_factor = 1.2
        elif getattr(event, "num", None) == 5:
            zoom_factor = 1 / 1.2
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return "break"
            zoom_factor = 1.2 if delta > 0 else 1 / 1.2
        old_zoom = self.preview_zoom
        new_zoom = min(max(old_zoom * zoom_factor, self.preview_min_zoom), self.preview_max_zoom)
        if abs(new_zoom - old_zoom) < 1e-6:
            return "break"
        image_point = self._canvas_xy_to_image_point(event.x, event.y, clamp=True)
        if image_point is None:
            return "break"
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        img_width, img_height = self.preview_pil_image.size
        base_scale = max(min(canvas_width / img_width, canvas_height / img_height), 0.01)
        self.preview_zoom = new_zoom
        new_scale = base_scale * self.preview_zoom
        display_width = max(1, int(round(img_width * new_scale)))
        display_height = max(1, int(round(img_height * new_scale)))
        new_offset_x = event.x - image_point[0] * display_width / img_width
        new_offset_y = event.y - image_point[1] * display_height / img_height
        self.preview_pan_x = new_offset_x - (canvas_width - display_width) / 2
        self.preview_pan_y = new_offset_y - (canvas_height - display_height) / 2
        self._clamp_preview_pan(canvas_width, canvas_height, display_width, display_height)
        self._render_canvas()
        self.status_var.set(f"预览缩放：{self.preview_zoom * 100:.0f}%。中键拖动可平移，Ctrl+0 可恢复。")
        return "break"

    def _on_canvas_pan_start(self, event: tk.Event) -> None:
        if self.preview_pil_image is None:
            return
        self.preview_pan_start = (event.x, event.y)
        self.preview_pan_origin = (self.preview_pan_x, self.preview_pan_y)
        self.canvas.configure(cursor="fleur")

    def _on_canvas_pan_move(self, event: tk.Event) -> None:
        if self.preview_pil_image is None or self.preview_pan_start is None or self.preview_pan_origin is None:
            return
        start_x, start_y = self.preview_pan_start
        origin_x, origin_y = self.preview_pan_origin
        self.preview_pan_x = origin_x + event.x - start_x
        self.preview_pan_y = origin_y + event.y - start_y
        self._render_canvas()

    def _on_canvas_pan_end(self, _event: tk.Event) -> None:
        self.preview_pan_start = None
        self.preview_pan_origin = None
        self.canvas.configure(cursor="")

    def _draw_polygon_overlay(self) -> None:
        self.canvas.delete("overlay")
        if self.preview_layout is None:
            return
        self._draw_keep_region_overlay()
        selected_index = self._get_selected_polygon_index()
        for polygon_index, polygon in enumerate(self.annotation_polygons):
            canvas_points = self._polygon_world_to_canvas_points(polygon.points_world)
            if len(canvas_points) < 2:
                continue
            flat_points = [value for point in canvas_points for value in point]
            line_width = 2 if polygon_index == selected_index else 1
            self.canvas.create_line(
                *flat_points, flat_points[0], flat_points[1],
                fill=LABEL_TO_COLOR[polygon.label], width=line_width, tags="overlay",
            )
            if polygon_index == selected_index:
                for x_coord, y_coord in canvas_points:
                    self.canvas.create_oval(
                        x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2,
                        fill=LABEL_TO_COLOR[polygon.label],
                        outline=LABEL_TO_COLOR[polygon.label], tags="overlay",
                    )
        active_points = self._polygon_world_to_canvas_points(self.active_polygon_points_world)
        if active_points:
            active_label = self.active_polygon_label or self.draw_mode_var.get()
            line_color = LABEL_TO_COLOR[active_label]
            for x_coord, y_coord in active_points:
                self.canvas.create_oval(
                    x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2,
                    fill=line_color, outline=line_color, tags="overlay",
                )
            if len(active_points) >= 2:
                flat_points = [value for point in active_points for value in point]
                self.canvas.create_line(*flat_points, fill=line_color, width=1, tags="overlay")
            if self.hover_world_point is not None:
                hover_canvas = self._world_to_canvas_point(self.hover_world_point)
                if hover_canvas is not None:
                    last_point = active_points[-1]
                    self.canvas.create_line(
                        last_point[0], last_point[1], hover_canvas[0], hover_canvas[1],
                        fill=line_color, width=1, dash=(3, 2), tags="overlay",
                    )
                    if len(active_points) >= 2:
                        first_point = active_points[0]
                        self.canvas.create_line(
                            hover_canvas[0], hover_canvas[1], first_point[0], first_point[1],
                            fill=line_color, width=1, dash=(3, 2), tags="overlay",
                        )

    def _draw_keep_region_overlay(self) -> None:
        box = self.keep_region_box
        if self.selecting_keep_region and self.keep_region_drag_start_world is not None:
            current = self.keep_region_drag_current_world or self.keep_region_drag_start_world
            sx, sy = self.keep_region_drag_start_world
            cx, cy = current
            box = (min(sx, cx), max(sx, cx), min(sy, cy), max(sy, cy))
        if box is None:
            return
        min_x, max_x, min_y, max_y = box
        corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        canvas_points = self._polygon_world_to_canvas_points(corners)
        if len(canvas_points) < 2:
            return
        flat_points = [value for point in canvas_points for value in point]
        self.canvas.create_line(
            *flat_points, flat_points[0], flat_points[1],
            fill="#3fa7ff", width=2, dash=(6, 4), tags="overlay",
        )
        if self.selecting_keep_region:
            label_x = min(point[0] for point in canvas_points)
            label_y = min(point[1] for point in canvas_points)
            self.canvas.create_text(
                label_x + 6, label_y + 6, anchor="nw", text="保留区域",
                fill="#9ed0ff", tags="overlay",
            )

    def _draw_placeholder(self, text: str) -> None:
        self.canvas.delete("all")
        self.preview_layout = None
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            text=text, fill="#d0d0d0",
            font=("Microsoft YaHei UI", 14),
        )

    def _canvas_to_image_point(self, event: tk.Event) -> tuple[float, float] | None:
        return self._canvas_xy_to_image_point(event.x, event.y)

    def _image_to_world_point(self, image_point: tuple[float, float]) -> tuple[float, float] | None:
        if self.current_map is None:
            return None
        x_pixel, y_pixel = image_point
        meta = self.current_map.metadata
        world_x = meta.origin_x + (x_pixel + 0.5) * meta.resolution
        world_y = meta.origin_y + (meta.height - y_pixel - 0.5) * meta.resolution
        return (world_x, world_y)

    def _world_to_image_point(self, world_point: tuple[float, float]) -> tuple[float, float] | None:
        if self.current_map is None:
            return None
        world_x, world_y = world_point
        meta = self.current_map.metadata
        image_x = (world_x - meta.origin_x) / meta.resolution
        image_y = meta.height - ((world_y - meta.origin_y) / meta.resolution) - 1.0
        return (image_x, image_y)

    def _world_to_canvas_point(self, world_point: tuple[float, float]) -> tuple[int, int] | None:
        if self.preview_layout is None:
            return None
        image_point = self._world_to_image_point(world_point)
        if image_point is None:
            return None
        x_pixel, y_pixel = image_point
        scale_x = self.preview_layout["display_width"] / self.preview_layout["image_width"]
        scale_y = self.preview_layout["display_height"] / self.preview_layout["image_height"]
        return (
            self.preview_layout["offset_x"] + int(round((x_pixel + 0.5) * scale_x)),
            self.preview_layout["offset_y"] + int(round((y_pixel + 0.5) * scale_y)),
        )

    def _polygon_world_to_canvas_points(
        self, points_world: list[tuple[float, float]],
    ) -> list[tuple[int, int]]:
        canvas_points: list[tuple[int, int]] = []
        for world_point in points_world:
            canvas_point = self._world_to_canvas_point(world_point)
            if canvas_point is not None:
                canvas_points.append(canvas_point)
        return canvas_points

    def _on_canvas_left_press(self, event: tk.Event) -> None:
        if self.current_map is None:
            return
        if self.selecting_keep_region:
            world_point = self._event_to_world_point(event)
            if world_point is None:
                return
            self.keep_region_drag_start_world = world_point
            self.keep_region_drag_current_world = world_point
            self.status_var.set("正在框选保留区域，松开左键后会应用到点云预处理。")
            self._draw_polygon_overlay()
            return
        self._on_canvas_left_click(event)

    def _on_canvas_left_drag(self, event: tk.Event) -> None:
        if not self.selecting_keep_region or self.keep_region_drag_start_world is None:
            return
        world_point = self._event_to_world_point(event, clamp=True)
        if world_point is None:
            return
        self.keep_region_drag_current_world = world_point
        self._draw_polygon_overlay()

    def _on_canvas_left_release(self, event: tk.Event) -> None:
        if not self.selecting_keep_region:
            return
        if self.keep_region_drag_start_world is None:
            return
        world_point = self._event_to_world_point(event, clamp=True)
        if world_point is None:
            world_point = self.keep_region_drag_current_world
        if world_point is None:
            self.keep_region_drag_start_world = None
            self.keep_region_drag_current_world = None
            self._draw_polygon_overlay()
            return
        start_x, start_y = self.keep_region_drag_start_world
        end_x, end_y = world_point
        min_x, max_x = sorted((float(start_x), float(end_x)))
        min_y, max_y = sorted((float(start_y), float(end_y)))
        min_size = max(float(self.resolution_var.get()), 1e-6)
        if (max_x - min_x) < min_size or (max_y - min_y) < min_size:
            self.status_var.set("框选区域太小，请重新拖拽一个更大的矩形。")
            self.keep_region_drag_start_world = None
            self.keep_region_drag_current_world = None
            self._draw_polygon_overlay()
            return
        self.keep_region_box = (min_x, max_x, min_y, max_y)
        self.keep_region_drag_start_world = None
        self.keep_region_drag_current_world = None
        self.selecting_keep_region = False
        self.canvas.configure(cursor="")
        self._update_keep_region_summary()
        self.status_var.set(
            f"已设置框选保留区域：X {min_x:.3f}~{max_x:.3f}, "
            f"Y {min_y:.3f}~{max_y:.3f}。正在重新生成点云和地图。"
        )
        self._apply_preprocess_async()

    def _event_to_world_point(self, event: tk.Event, *, clamp: bool = False) -> tuple[float, float] | None:
        image_point = self._canvas_xy_to_image_point(event.x, event.y, clamp=clamp)
        if image_point is None:
            return None
        return self._image_to_world_point(image_point)

    def _on_canvas_left_click(self, event: tk.Event) -> None:
        if self.current_map is None:
            return
        image_point = self._canvas_to_image_point(event)
        if image_point is None:
            return
        world_point = self._image_to_world_point(image_point)
        if world_point is None:
            return
        if not self.active_polygon_points_world:
            self.active_polygon_label = self.draw_mode_var.get()
        self.active_polygon_points_world.append(world_point)
        self.hover_world_point = world_point
        self.status_var.set(
            f"当前多边形已添加 {len(self.active_polygon_points_world)} 个顶点。"
            "右键或 Enter 可完成，调整 Z 后也可以继续补点。"
        )
        self._draw_polygon_overlay()

    def _on_canvas_right_click(self, _event: tk.Event) -> None:
        self._commit_active_polygon()

    def _on_canvas_motion(self, event: tk.Event) -> None:
        if self.selecting_keep_region:
            return
        if not self.active_polygon_points_world:
            return
        image_point = self._canvas_to_image_point(event)
        if image_point is None:
            self.hover_world_point = None
        else:
            self.hover_world_point = self._image_to_world_point(image_point)
        self._draw_polygon_overlay()

    def _on_canvas_leave(self, _event: tk.Event) -> None:
        self.hover_world_point = None
        if self.active_polygon_points_world:
            self._draw_polygon_overlay()

    def _commit_active_polygon(self) -> None:
        if len(self.active_polygon_points_world) < 3:
            if self.active_polygon_points_world:
                self.status_var.set("至少需要 3 个顶点才能形成多边形。")
            return
        polygon_label = self.active_polygon_label or self.draw_mode_var.get()
        self.annotation_polygons.append(
            PolygonAnnotation(label=polygon_label, points_world=list(self.active_polygon_points_world))
        )
        polygon_text = LABEL_TO_TEXT[polygon_label]
        self.active_polygon_points_world = []
        self.active_polygon_label = None
        self.hover_world_point = None
        self._refresh_polygon_list(select_index=len(self.annotation_polygons) - 1)
        self._refresh_preview_image()
        self.status_var.set(
            f"已提交一个 {polygon_text} 多边形。你可以继续画下一个，或者选中列表里的多边形继续补点。"
        )

    def _undo_last_polygon_point(self) -> None:
        if not self.active_polygon_points_world:
            return
        self.active_polygon_points_world.pop()
        if self.active_polygon_points_world:
            self.status_var.set(
                f"已撤销一个顶点，当前多边形还剩 {len(self.active_polygon_points_world)} 个顶点。"
            )
        else:
            self.active_polygon_label = None
            self.hover_world_point = None
            self.status_var.set("当前多边形已清空。")
        self._draw_polygon_overlay()

    def _cancel_active_polygon(self) -> None:
        if self.selecting_keep_region:
            self.selecting_keep_region = False
            self.keep_region_drag_start_world = None
            self.keep_region_drag_current_world = None
            self.canvas.configure(cursor="")
            self._draw_polygon_overlay()
            self.status_var.set("已取消框选保留区域。")
            return
        if not self.active_polygon_points_world:
            return
        self.active_polygon_points_world = []
        self.active_polygon_label = None
        self.hover_world_point = None
        self.status_var.set("已取消当前多边形。")
        self._draw_polygon_overlay()

    def _edit_selected_polygon(self) -> None:
        polygon_index = self._get_selected_polygon_index()
        if polygon_index is None:
            messagebox.showinfo("提示", "请先在列表里选中一个多边形。")
            return
        if self.active_polygon_points_world:
            replace = messagebox.askyesno(
                "存在未完成多边形",
                "当前还有一个未完成的多边形，是否放弃它并切换到选中的多边形继续编辑？",
            )
            if not replace:
                return
        polygon = self.annotation_polygons.pop(polygon_index)
        self.active_polygon_points_world = list(polygon.points_world)
        self.active_polygon_label = polygon.label
        self.draw_mode_var.set(polygon.label)
        self.hover_world_point = None
        self._refresh_polygon_list()
        self._refresh_preview_image()
        self.status_var.set(
            f"已载入一个 {LABEL_TO_TEXT[polygon.label]} 多边形，"
            "你可以继续补点、撤销点，完成后再次提交。"
        )

    def _delete_selected_polygon(self) -> None:
        polygon_index = self._get_selected_polygon_index()
        if polygon_index is None:
            messagebox.showinfo("提示", "请先在列表里选中一个多边形。")
            return
        polygon = self.annotation_polygons.pop(polygon_index)
        self._refresh_polygon_list()
        self._refresh_preview_image()
        self.status_var.set(f"已删除一个 {LABEL_TO_TEXT[polygon.label]} 多边形。")

    def _get_selected_polygon_index(self) -> int | None:
        selection = self.polygon_listbox.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if index < 0 or index >= len(self.annotation_polygons):
            return None
        return index

    def _refresh_polygon_list(self, select_index: int | None = None) -> None:
        previous_index = self._get_selected_polygon_index()
        self.polygon_listbox.delete(0, tk.END)
        for index, polygon in enumerate(self.annotation_polygons, start=1):
            self.polygon_listbox.insert(
                tk.END,
                f"{index}. {LABEL_TO_TEXT[polygon.label]} ({len(polygon.points_world)} 点)",
            )
        total_count = len(self.annotation_polygons)
        active_count = len(self.active_polygon_points_world)
        if total_count == 0:
            if active_count:
                self.polygon_summary_var.set(
                    f"已提交 0 个多边形，当前有 1 个未完成多边形，已打 {active_count} 个点。"
                )
            else:
                self.polygon_summary_var.set("当前没有已提交的多边形。")
        else:
            summary = f"已提交 {total_count} 个多边形。"
            if active_count:
                summary += f" 当前未完成多边形已有 {active_count} 个点。"
            self.polygon_summary_var.set(summary)
        target_index = select_index
        if target_index is None:
            target_index = previous_index
        if target_index is not None and 0 <= target_index < len(self.annotation_polygons):
            self.polygon_listbox.selection_set(target_index)

    def _clear_annotations(self) -> None:
        if self.current_map is None:
            return
        self.base_label_mask = None
        self.label_mask = np.zeros(
            (self.current_map.metadata.height, self.current_map.metadata.width), dtype=np.uint8
        )
        self.annotation_polygons = []
        self.active_polygon_points_world = []
        self.active_polygon_label = None
        self.hover_world_point = None
        self._refresh_polygon_list()
        self._refresh_preview_image()
        self.status_var.set("全部标注已清空。")

    def _load_annotation_mask(self) -> None:
        if self.current_map is None:
            messagebox.showinfo("提示", "请先加载文件并生成地图。")
            return
        file_name = filedialog.askopenfilename(
            title="加载标注 PNG",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if not file_name:
            return
        try:
            loaded = Image.open(file_name).convert("L")
            loaded_mask = np.asarray(loaded, dtype=np.uint8)
            normalized = normalize_label_mask_input(loaded_mask, source_name=file_name)
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))
            return
        self.base_label_mask = normalized
        self.annotation_polygons = []
        self.active_polygon_points_world = []
        self.active_polygon_label = None
        self.hover_world_point = None
        self._refresh_polygon_list()
        self._refresh_preview_image()
        self.status_var.set("已有标注已作为底图载入。之后可以继续叠加新的多边形。")

    def _export_bundle(self) -> None:
        if self.current_map is None:
            messagebox.showinfo("提示", "请先生成地图。")
            return
        if self.active_polygon_points_world:
            proceed = messagebox.askyesno(
                "未完成的多边形",
                "当前还有未完成的多边形。是否先自动提交后再导出？",
            )
            if not proceed:
                return
            self._commit_active_polygon()
            if self.active_polygon_points_world:
                return

        # ── Show selective export dialog ──
        dialog = ExportSelectDialog(self.root)
        if not dialog.result:
            return
        export_selection = dialog.result

        suggested_name = "map_output"
        if self.point_cloud_path is not None:
            suggested_name = self.point_cloud_path.stem

        file_name = filedialog.asksaveasfilename(
            title="导出地图与标注",
            defaultextension=".pgm",
            initialfile=f"{suggested_name}.pgm",
            filetypes=[("PGM files", "*.pgm"), ("All files", "*.*")],
        )
        if not file_name:
            return

        output_prefix = Path(file_name).with_suffix("")

        # Build session data
        modified_pcd_name = f"{output_prefix.name}_modified.pcd"
        session = {
            "source_pcd": str(self.point_cloud_path) if self.point_cloud_path else None,
            "modified_pcd": modified_pcd_name,
            "z_min": self.current_map.z_min,
            "z_max": self.current_map.z_max,
            "resolution": self.current_map.metadata.resolution,
            "min_points_per_cell": int(self.min_points_var.get()),
            "fixed_extent": bool(self.fixed_extent_var.get()),
            "preprocess": self.preprocess_info,
            "polygon_count": len(self.annotation_polygons),
            "polygons": [
                {"label": polygon.label, "points_world": polygon.points_world}
                for polygon in self.annotation_polygons
            ],
        }

        try:
            # Export selected files
            outputs = save_export_bundle(
                output_prefix=output_prefix,
                map_result=self.current_map,
                label_mask=self.label_mask,
                session_data=session if export_selection.get("session") else None,
                export_selection=export_selection,
            )

            # Export modified PCD if selected and we have point cloud data
            if export_selection.get("modified_pcd") and self.processed_points is not None:
                modified_pcd_path = output_prefix.with_name(modified_pcd_name)
                outputs["modified_pcd"] = save_pcd_xyz(modified_pcd_path, self.processed_points)
            elif export_selection.get("modified_pcd") and self.is_pgm_mode:
                # In PGM mode, warn that modified PCD is not available
                messagebox.showwarning(
                    "提示", "PGM 模式下没有点云数据，modified.pcd 不会导出。"
                )

            # Export transform JSON if selected
            if export_selection.get("transform"):
                transform_name = f"{output_prefix.name}_transform.json"
                transform_path = output_prefix.with_name(transform_name)
                transform_payload = {
                    "source_pcd": str(self.point_cloud_path) if self.point_cloud_path else None,
                    "modified_pcd": modified_pcd_name,
                    "map_pgm": output_prefix.with_suffix(".pgm").name,
                    "map_yaml": output_prefix.with_suffix(".yaml").name,
                    "same_source_for_pgm_and_modified_pcd": True,
                    "note": "PGM preview/export and modified.pcd are generated from the same processed point cloud coordinates.",
                    "preprocess": self.preprocess_info,
                }
                transform_path.write_text(
                    json.dumps(transform_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                outputs["transform"] = transform_path
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            return

        created = [str(path) for path in outputs.values() if path is not None]
        self.status_var.set("导出完成。")
        if created:
            messagebox.showinfo("导出完成", "已生成以下文件:\n\n" + "\n".join(created))
        else:
            messagebox.showinfo("导出完成", "没有文件被导出。")


class ExportSelectDialog:
    """A dialog with checkboxes to let the user select which files to export."""

    EXPORT_ITEMS = [
        ("pgm",             "地图 PGM"),
        ("yaml",            "ROS YAML"),
        ("preview",         "预览 PNG"),
        ("regions_mask",    "区域标注掩膜 (regions_mask.png)"),
        ("traversable_mask","可通行掩膜 (traversable_mask.png)"),
        ("blocked_mask",    "不可通行掩膜 (blocked_mask.png)"),
        ("regions_meta",    "区域元数据 (regions.json)"),
        ("session",         "会话数据 (session.json)"),
        ("modified_pcd",    "修改后的点云 (modified.pcd)"),
        ("transform",       "变换信息 (transform.json)"),
    ]

    def __init__(self, parent: tk.Tk):
        self.result: Dict[str, bool] | None = None

        top = tk.Toplevel(parent)
        top.title("选择导出文件")
        top.resizable(False, False)
        top.transient(parent)
        top.grab_set()

        # Center on parent
        top.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        tw, th = 420, 430
        x = px + (pw - tw) // 2
        y = py + (ph - th) // 2
        top.geometry(f"{tw}x{th}+{x}+{y}")

        ttk.Label(top, text="勾选需要导出的文件：", font=("", 10, "bold")).pack(
            padx=16, pady=(16, 8), anchor="w"
        )

        # Checkbox variables
        self.vars: Dict[str, tk.BooleanVar] = {}
        checkframe = ttk.Frame(top, padding=(16, 4))
        checkframe.pack(fill="x")

        for key, label in self.EXPORT_ITEMS:
            var = tk.BooleanVar(value=True)
            self.vars[key] = var
            ttk.Checkbutton(
                checkframe, text=label, variable=var,
            ).pack(anchor="w", pady=2)

        # Buttons
        btnframe = ttk.Frame(top)
        btnframe.pack(fill="x", padx=16, pady=(8, 16))

        def on_ok() -> None:
            self.result = {key: var.get() for key, var in self.vars.items()}
            top.destroy()

        def on_cancel() -> None:
            self.result = None
            top.destroy()

        ttk.Button(btnframe, text="全部勾选", command=lambda: self._set_all(True)).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(btnframe, text="全部取消", command=lambda: self._set_all(False)).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(btnframe, text="确定", command=on_ok).pack(side="right", padx=(8, 0))
        ttk.Button(btnframe, text="取消", command=on_cancel).pack(side="right")

        # Wait for dialog to close
        parent.wait_window(top)

    def _set_all(self, value: bool) -> None:
        for var in self.vars.values():
            var.set(value)


# ─────────────────────────── 启动入口 ───────────────────────────

def launch_app(initial_path: str | None = None) -> None:
    root = tk.Tk()
    PCDPGMToolApp(root, initial_path=initial_path)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="PCD / PGM 转 PGM 预览与标注工具")
    parser.add_argument("--gui", action="store_true", help="启动图形界面")
    parser.add_argument("--input", help="输入的 PCD 或 PGM 文件路径")
    parser.add_argument("--output", help="导出的输出前缀，例如 output/map")
    parser.add_argument("--z-min", type=float, help="Z 轴最小阈值")
    parser.add_argument("--z-max", type=float, help="Z 轴最大阈值")
    parser.add_argument("--resolution", type=float, default=0.05, help="地图分辨率，默认 0.05")
    parser.add_argument(
        "--min-points-per-cell", type=int, default=1,
        help="投影到同一个栅格后，至少需要多少个点才算占据",
    )
    parser.add_argument(
        "--freehand-mask",
        help="可选，导出时一并带上的标注 PNG，像素值使用 0/1/2 或 0/255。",
    )
    parser.add_argument(
        "--unlock-extent", action="store_true",
        help="不锁定全局 XY 范围，阈值变化时地图边界会跟随变化。",
    )
    parser.add_argument("--rotation-deg", type=float, default=0.0, help="点云 XY 平面旋转角度，单位度。")
    parser.add_argument("--origin-x", type=float, default=0.0, help="旋转后从点云 X 坐标中减去的原点 X。")
    parser.add_argument("--origin-y", type=float, default=0.0, help="旋转后从点云 Y 坐标中减去的原点 Y。")
    parser.add_argument(
        "--crop-percent", type=float, default=0.0,
        help="按 X/Y 百分位裁剪外围点，例如 0.5 表示每侧裁剪 0.5%%。",
    )
    parser.add_argument(
        "--keep-box", nargs=4, type=float, metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"),
        help="只保留处理后坐标位于该 XY 矩形内的点，用于命令行框选裁剪。",
    )
    parser.add_argument(
        "--keep-largest-component", action="store_true",
        help="删除外围孤立杂点，只保留 XY 投影上的最大主体连通区域。",
    )
    parser.add_argument(
        "--component-min-cells", type=int, default=3,
        help="最大主体区域至少需要的栅格数量，默认 3。",
    )
    args = parser.parse_args()

    should_launch_gui = args.gui or (args.input is None and args.output is None)
    if should_launch_gui:
        launch_app(initial_path=args.input)
        return

    # Determine if input is PGM or PCD
    input_path = Path(args.input)
    if input_path.suffix.lower() == ".pgm":
        # PGM mode: load map and export with annotations
        map_result = load_pgm_to_map_result(input_path)
        label_mask = None
        if args.freehand_mask:
            loaded = np.asarray(Image.open(args.freehand_mask).convert("L"), dtype=np.uint8)
            label_mask = normalize_label_mask_input(loaded, source_name=args.freehand_mask)
        output_prefix = Path(args.output)
        outputs = save_export_bundle(
            output_prefix=output_prefix, map_result=map_result, label_mask=label_mask,
            session_data={
                "source_pgm": str(input_path),
                "resolution": map_result.metadata.resolution,
            },
        )
        print("导出完成:")
        for name, path in outputs.items():
            if path is not None:
                print(f"- {name}: {path}")
        return

    # PCD mode: original logic
    if not args.output:
        parser.error("命令行导出模式下必须同时提供 --input 和 --output。")

    raw_points = load_pcd_xyz(args.input)
    preprocess_params = PreprocessParams(
        rotation_deg=args.rotation_deg,
        origin_x=args.origin_x,
        origin_y=args.origin_y,
        crop_percent=args.crop_percent,
        keep_box=tuple(args.keep_box) if args.keep_box else None,
        keep_largest_component=args.keep_largest_component,
        component_resolution=args.resolution,
        component_min_cells=args.component_min_cells,
        component_min_points_per_cell=1,
    )
    processed_points, preprocess_info = preprocess_point_cloud(raw_points, preprocess_params)
    builder = OccupancyMapBuilder(processed_points)
    z_min, z_max = builder.z_range
    chosen_z_min = z_min if args.z_min is None else args.z_min
    chosen_z_max = z_max if args.z_max is None else args.z_max

    map_result = builder.build_map(
        z_min=chosen_z_min, z_max=chosen_z_max,
        resolution=args.resolution,
        min_points_per_cell=args.min_points_per_cell,
        fixed_extent=not args.unlock_extent,
    )

    label_mask = None
    if args.freehand_mask:
        loaded = np.asarray(Image.open(args.freehand_mask).convert("L"), dtype=np.uint8)
        label_mask = normalize_label_mask_input(loaded, source_name=args.freehand_mask)

    output_prefix = Path(args.output)
    modified_pcd_name = f"{output_prefix.name}_modified.pcd"
    outputs = save_export_bundle(
        output_prefix=output_prefix, map_result=map_result, label_mask=label_mask,
        session_data={
            "source_pcd": str(Path(args.input)),
            "modified_pcd": modified_pcd_name,
            "z_min": chosen_z_min,
            "z_max": chosen_z_max,
            "resolution": args.resolution,
            "min_points_per_cell": args.min_points_per_cell,
            "fixed_extent": not args.unlock_extent,
            "preprocess": preprocess_info,
        },
    )
    outputs["modified_pcd"] = save_pcd_xyz(output_prefix.with_name(modified_pcd_name), processed_points)

    transform_path = output_prefix.with_name(f"{output_prefix.name}_transform.json")
    transform_path.write_text(
        json.dumps(
            {
                "source_pcd": str(Path(args.input)),
                "modified_pcd": modified_pcd_name,
                "map_pgm": output_prefix.with_suffix(".pgm").name,
                "map_yaml": output_prefix.with_suffix(".yaml").name,
                "same_source_for_pgm_and_modified_pcd": True,
                "note": "PGM and modified.pcd are generated from the same processed point cloud coordinates.",
                "preprocess": preprocess_info,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    outputs["transform"] = transform_path

    print("导出完成:")
    for name, path in outputs.items():
        if path is not None:
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
