"""Helpers for host-independent filesystem paths in launch parameters."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


PACKAGE_URI_PREFIX = "package://"


def resolve_path(path_value):
    """Resolve ~, environment variables, and package://pkg/relative/path URIs."""
    if not isinstance(path_value, str):
        return path_value

    expanded = os.path.expanduser(os.path.expandvars(path_value.strip()))
    if not expanded.startswith(PACKAGE_URI_PREFIX):
        return expanded

    remainder = expanded[len(PACKAGE_URI_PREFIX) :]
    package_name, separator, relative_path = remainder.partition("/")
    if not package_name or not separator:
        return expanded

    return str(Path(get_package_share_directory(package_name)) / relative_path)


def resolve_path_params(params, keys):
    """Return a copy of params with selected string path parameters resolved."""
    resolved = dict(params)
    for key in keys:
        value = resolved.get(key)
        if isinstance(value, str) and value.strip():
            resolved[key] = resolve_path(value)
    return resolved


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def flatten_floors(params):
    """Flatten a user-facing ``floors`` list-of-dicts into parallel arrays.

    The user edits a nested list in global_prm.yaml::

        floors:
          - index: 0
            z_min: 7.0
            z_max: 9.0
            map: /path/to/floor.png
            elevation_npy: /path/to/elevation.npy   # optional

    ROS 2 parameters cannot carry a nested list-of-dicts, so this turns it into
    the flat arrays consumed by prm_planner_node: ``floor_maps``,
    ``floor_z_mins``, ``floor_z_maxs``, ``floor_elevation_npys``. ``index`` is
    used for ordering (falling back to list position); ``map`` and
    ``elevation_npy`` paths are resolved like any other path parameter.
    """
    if not isinstance(params, dict):
        return params

    floors = params.get("floors")
    if not isinstance(floors, list) or not floors:
        return dict(params)

    entries = []
    for position, entry in enumerate(floors):
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("index", position))
        except (TypeError, ValueError):
            index = position
        entries.append((index, entry))
    entries.sort(key=lambda item: item[0])

    out = dict(params)
    out.pop("floors", None)
    out["floor_maps"] = [
        resolve_path(str(entry.get("map", "")).strip()) for _, entry in entries
    ]
    out["floor_z_mins"] = [_as_float(entry.get("z_min", 0.0)) for _, entry in entries]
    out["floor_z_maxs"] = [_as_float(entry.get("z_max", 0.0)) for _, entry in entries]
    # Optional per-floor walking-surface height; NaN ("nan") lets the planner
    # estimate it from the PCD at startup.
    out["floor_ground_zs"] = [
        _as_float(entry.get("ground_z", float("nan"))) for _, entry in entries
    ]
    out["floor_elevation_npys"] = [
        resolve_path(str(entry.get("elevation_npy", "")).strip()) for _, entry in entries
    ]
    return out
