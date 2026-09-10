// PRM-3D multi-floor global planner node.
//
// Subscribes to odometry and /goal_pose, maps metric (x,y,z) to (pixel, floor_id)
// using a configurable resolution/origin and per-floor Z ranges, plans through the
// fused multi-floor PRM roadmap, and publishes nav_msgs/Path to /global_path.
//
// The 2D PRM path is then lifted to 3D. Path Z has two sources (z_source):
//   "npy" — per-floor 2.5D elevation grids (floors[].elevation_npy), produced
//     offline by process_multi_floor_elevations.py from the same PCD the 2D
//     maps were sliced from: per-cell ground Z over each floor's walkable
//     pixels, stairwell bands widened, holes interpolated, lightly smoothed.
//     Each path point is a bilinear lookup in its floor's grid — no runtime
//     PCD access, and the grids are inspectable as images before ever
//     planning.
//   "pcd" — direct per-point query of the global PCD map (pcd_map_path)
//     inside a tight window around the floor's walking-surface height
//     (floor_ground_zs override or auto-estimated), keeping the densest Z
//     cluster and a low quantile of it.
// In both cases points with no support fall back to the PCD query (npy mode),
// then inherit the previous valid Z, then the floor's ground / z_min; and the
// Z series is finished by a median filter + slope clamp (see postProcessZ).
//
// API usage mirrors test/sim2d.cpp in the PRM-3D repo.

#include "PRMmap.h"
#include "PRMmultifloor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <prm_interfaces/srv/replan_plan.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

namespace
{

geometry_msgs::msg::Point makePoint(double x, double y, double z)
{
  geometry_msgs::msg::Point p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

std_msgs::msg::ColorRGBA makeColor(float r, float g, float b, float a)
{
  std_msgs::msg::ColorRGBA c;
  c.r = r;
  c.g = g;
  c.b = b;
  c.a = a;
  return c;
}

// A point of the global PCD map (map frame, metres).
struct PcdPoint
{
  float x;
  float y;
  float z;
};

// Hash-grid index over the PCD map for ground-Z lookups: the XY plane is tiled
// into square cells and each cell keeps the points that fall inside it.
struct PcdZIndex
{
  double cell = 0.25;   // cell size (m)
  double x0 = 0.0;      // grid origin = point-cloud XY minimum minus one cell
  double y0 = 0.0;
  int nx = 0;
  int ny = 0;
  std::vector<std::vector<PcdPoint>> cells;
  bool valid = false;
};

// Minimal PCD reader for the x/y/z fields. Supports DATA ascii and DATA
// binary (binary_compressed is not supported). Returns false on any
// unsupported layout.
bool readPcdXyz(const std::string & path, std::vector<PcdPoint> & points)
{
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    return false;
  }

  std::vector<std::string> fields;
  std::vector<int> sizes;
  std::vector<char> types;
  std::vector<int> counts;
  long point_count = 0;
  std::string data_mode;

  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream ss(line);
    std::string key;
    ss >> key;
    if (key == "FIELDS") {
      std::string v;
      while (ss >> v) {
        fields.push_back(v);
      }
    } else if (key == "SIZE") {
      int v;
      while (ss >> v) {
        sizes.push_back(v);
      }
    } else if (key == "TYPE") {
      char v;
      while (ss >> v) {
        types.push_back(v);
      }
    } else if (key == "COUNT") {
      int v;
      while (ss >> v) {
        counts.push_back(v);
      }
    } else if (key == "POINTS") {
      ss >> point_count;
    } else if (key == "DATA") {
      ss >> data_mode;
      break;
    }
  }

  if (data_mode.empty() || point_count <= 0 || fields.size() != sizes.size() ||
      fields.size() != types.size()) {
    return false;
  }
  if (counts.size() != fields.size()) {
    counts.assign(fields.size(), 1);
  }

  int ix = -1;
  int iy = -1;
  int iz = -1;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (fields[i] == "x") {
      ix = static_cast<int>(i);
    } else if (fields[i] == "y") {
      iy = static_cast<int>(i);
    } else if (fields[i] == "z") {
      iz = static_cast<int>(i);
    }
  }
  if (ix < 0 || iy < 0 || iz < 0) {
    return false;
  }

  points.clear();
  points.reserve(static_cast<size_t>(point_count));

  if (data_mode == "ascii") {
    std::string row;
    while (std::getline(f, row)) {
      if (row.empty()) {
        continue;
      }
      std::istringstream rs(row);
      std::vector<double> vals;
      double v;
      while (rs >> v) {
        vals.push_back(v);
      }
      if (vals.size() < fields.size()) {
        continue;
      }
      points.push_back(PcdPoint{
        static_cast<float>(vals[static_cast<size_t>(ix)]),
        static_cast<float>(vals[static_cast<size_t>(iy)]),
        static_cast<float>(vals[static_cast<size_t>(iz)])});
    }
    return !points.empty();
  }

  if (data_mode != "binary") {
    return false;
  }

  // Byte layout of one binary point record.
  std::vector<size_t> offset(fields.size(), 0);
  size_t point_step = 0;
  for (size_t i = 0; i < fields.size(); ++i) {
    offset[i] = point_step;
    point_step += static_cast<size_t>(sizes[i]) * static_cast<size_t>(counts[i]);
  }

  auto readField = [](const std::vector<char> & buf, size_t off, int size, char type) -> double {
    switch (size) {
      case 4:
        if (type == 'F') {
          return *reinterpret_cast<const float *>(&buf[off]);
        }
        if (type == 'I') {
          return static_cast<double>(*reinterpret_cast<const int32_t *>(&buf[off]));
        }
        return static_cast<double>(*reinterpret_cast<const uint32_t *>(&buf[off]));
      case 8:
        if (type == 'F') {
          return *reinterpret_cast<const double *>(&buf[off]);
        }
        return static_cast<double>(*reinterpret_cast<const int64_t *>(&buf[off]));
      case 2:
        if (type == 'I') {
          return static_cast<double>(*reinterpret_cast<const int16_t *>(&buf[off]));
        }
        return static_cast<double>(*reinterpret_cast<const uint16_t *>(&buf[off]));
      case 1:
        return static_cast<double>(static_cast<unsigned char>(buf[off]));
      default:
        return 0.0;
    }
  };

  std::vector<char> buf(point_step);
  for (long p = 0; p < point_count; ++p) {
    f.read(buf.data(), static_cast<std::streamsize>(point_step));
    if (f.gcount() != static_cast<std::streamsize>(point_step)) {
      break;
    }
    points.push_back(PcdPoint{
      static_cast<float>(readField(buf, offset[static_cast<size_t>(ix)],
        sizes[static_cast<size_t>(ix)], types[static_cast<size_t>(ix)])),
      static_cast<float>(readField(buf, offset[static_cast<size_t>(iy)],
        sizes[static_cast<size_t>(iy)], types[static_cast<size_t>(iy)])),
      static_cast<float>(readField(buf, offset[static_cast<size_t>(iz)],
        sizes[static_cast<size_t>(iz)], types[static_cast<size_t>(iz)]))});
  }
  return !points.empty();
}

// Bucket the points into the hash grid.
bool buildPcdZIndex(const std::vector<PcdPoint> & points, double cell, PcdZIndex & idx)
{
  if (points.empty() || cell <= 0.0) {
    return false;
  }
  double min_x = points[0].x;
  double max_x = points[0].x;
  double min_y = points[0].y;
  double max_y = points[0].y;
  for (const PcdPoint & p : points) {
    min_x = std::min(min_x, static_cast<double>(p.x));
    max_x = std::max(max_x, static_cast<double>(p.x));
    min_y = std::min(min_y, static_cast<double>(p.y));
    max_y = std::max(max_y, static_cast<double>(p.y));
  }
  idx = PcdZIndex{};
  idx.cell = cell;
  idx.x0 = min_x - cell;
  idx.y0 = min_y - cell;
  idx.nx = static_cast<int>((max_x - idx.x0) / cell) + 2;
  idx.ny = static_cast<int>((max_y - idx.y0) / cell) + 2;
  idx.cells.assign(static_cast<size_t>(idx.nx) * static_cast<size_t>(idx.ny), {});
  for (const PcdPoint & p : points) {
    const int cx = static_cast<int>((p.x - idx.x0) / cell);
    const int cy = static_cast<int>((p.y - idx.y0) / cell);
    if (cx < 0 || cx >= idx.nx || cy < 0 || cy >= idx.ny) {
      continue;
    }
    idx.cells[static_cast<size_t>(cy) * idx.nx + cx].push_back(p);
  }
  idx.valid = true;
  return true;
}

// Ground-height Z within [z_min, z_max] among points closer than `radius` to
// (x, y). Two robustness stages: first only the densest cluster of Z values
// (width cluster_tol) is kept — the surface the neighbourhood is actually made
// of — then the `quantile` quantile inside that cluster is taken (0 = cluster
// minimum, 0.5 = median, 1 = cluster maximum, ~0.25 ≈ ground). Isolated stray
// points below the floor and separate surfaces (ceiling, another floor sharing
// the band) are discarded by the cluster stage. NaN when no point qualifies.
double queryQuantileZ(
  const PcdZIndex & idx, double x, double y, double radius, double z_min,
  double z_max, double quantile, double cluster_tol)
{
  if (!idx.valid) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const int cx0 = static_cast<int>(std::floor((x - radius - idx.x0) / idx.cell));
  const int cx1 = static_cast<int>(std::floor((x + radius - idx.x0) / idx.cell));
  const int cy0 = static_cast<int>(std::floor((y - radius - idx.y0) / idx.cell));
  const int cy1 = static_cast<int>(std::floor((y + radius - idx.y0) / idx.cell));
  const double r2 = radius * radius;
  std::vector<double> zs;
  for (int cy = std::max(0, cy0); cy <= std::min(idx.ny - 1, cy1); ++cy) {
    for (int cx = std::max(0, cx0); cx <= std::min(idx.nx - 1, cx1); ++cx) {
      for (const PcdPoint & p : idx.cells[static_cast<size_t>(cy) * idx.nx + cx]) {
        const double dx = p.x - x;
        const double dy = p.y - y;
        if (dx * dx + dy * dy > r2) {
          continue;
        }
        if (p.z >= z_min && p.z <= z_max) {
          zs.push_back(p.z);
        }
      }
    }
  }
  if (zs.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(zs.begin(), zs.end());

  // Densest-window cluster: two pointers over the sorted Z values, window
  // width cluster_tol; the window holding the most points is the dominant
  // surface. Ties resolve to the lowest window.
  if (cluster_tol > 0.0 && zs.size() > 1) {
    size_t best_lo = 0;
    size_t best_cnt = 0;
    size_t lo = 0;
    for (size_t hi = 0; hi < zs.size(); ++hi) {
      while (zs[hi] - zs[lo] > cluster_tol) {
        ++lo;
      }
      const size_t cnt = hi - lo + 1;
      if (cnt > best_cnt) {
        best_cnt = cnt;
        best_lo = lo;
      }
    }
    zs.assign(zs.begin() + static_cast<std::ptrdiff_t>(best_lo),
      zs.begin() + static_cast<std::ptrdiff_t>(best_lo + best_cnt));
  }

  const double q = std::clamp(quantile, 0.0, 1.0);
  const size_t pick = static_cast<size_t>(
    std::lround(q * static_cast<double>(zs.size() - 1)));
  return zs[pick];
}

// A per-floor 2.5D elevation grid, produced offline from the same PCD the 2D
// maps were sliced from by process_multi_floor_elevations.py (per-cell ground
// Z over the floor's walkable pixels, holes interpolated, lightly smoothed).
// Indexed [u, v] with u = (x - origin_x)/res and v = (y - origin_y)/res —
// note there is NO y flip here, unlike the map PNGs whose rows grow downward
// (the script writes v = height - 1 - row). NaN outside the walkable area.
struct ElevationGrid
{
  int width = 0;          // u axis size (map columns)
  int height = 0;         // v axis size (map rows)
  std::vector<double> z;  // [u * height + v]
  bool valid = false;
};

// Minimal .npy reader: version 1/2 headers, dtype <f4/<f8, 2-D, C order —
// exactly what np.save() produces for the float matrices above. Returns false
// on any other layout.
bool readNpyGrid(const std::string & path, ElevationGrid & grid)
{
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    return false;
  }
  static const char MAGIC[6] = {(char)0x93, 'N', 'U', 'M', 'P', 'Y'};
  char magic[6];
  f.read(magic, 6);
  if (f.gcount() != 6 || std::memcmp(magic, MAGIC, 6) != 0) {
    return false;
  }
  char version[2];
  f.read(version, 2);
  if (f.gcount() != 2 || version[0] < 1 || version[0] > 2) {
    return false;
  }
  uint32_t hlen = 0;
  if (version[0] == 1) {
    uint16_t h16 = 0;
    f.read(reinterpret_cast<char *>(&h16), 2);
    if (f.gcount() != 2) {
      return false;
    }
    hlen = h16;
  } else {
    uint32_t h32 = 0;
    f.read(reinterpret_cast<char *>(&h32), 4);
    if (f.gcount() != 4) {
      return false;
    }
    hlen = h32;
  }
  std::string header(hlen, '\0');
  f.read(&header[0], static_cast<std::streamsize>(hlen));
  if (f.gcount() != static_cast<std::streamsize>(hlen)) {
    return false;
  }

  // descr: the first '...' quoted string after the "descr" key — skip the
  // key's own closing quote first ('descr': '<f8')
  std::string descr;
  size_t pos = header.find("descr");
  if (pos != std::string::npos) {
    size_t q1 = header.find('\'', pos);
    if (q1 != std::string::npos) {
      q1 = header.find('\'', q1 + 1);  // value's opening quote
    }
    const size_t q2 = (q1 == std::string::npos) ? std::string::npos : header.find('\'', q1 + 1);
    if (q2 != std::string::npos) {
      descr = header.substr(q1 + 1, q2 - q1 - 1);
    }
  }
  if (descr != "<f4" && descr != "<f8") {
    return false;  // little-endian float32/64 only
  }

  // fortran_order must be False (row-major)
  pos = header.find("fortran_order");
  if (pos == std::string::npos || header.find("False", pos) == std::string::npos) {
    return false;
  }

  // shape: integers inside the parentheses after "shape"
  std::vector<long long> shape;
  pos = header.find("shape");
  if (pos == std::string::npos) {
    return false;
  }
  const size_t p1 = header.find('(', pos);
  const size_t p2 = (p1 == std::string::npos) ? std::string::npos : header.find(')', p1);
  if (p1 == std::string::npos || p2 == std::string::npos) {
    return false;
  }
  size_t i = p1 + 1;
  while (i < p2) {
    if (std::isdigit(static_cast<unsigned char>(header[i]))) {
      long long v = 0;
      while (i < p2 && std::isdigit(static_cast<unsigned char>(header[i]))) {
        v = v * 10 + (header[i] - '0');
        ++i;
      }
      shape.push_back(v);
    } else {
      ++i;
    }
  }
  if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
    return false;
  }

  const size_t count = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]);
  const size_t elem = (descr == "<f8") ? 8u : 4u;
  std::vector<char> raw(count * elem);
  f.read(raw.data(), static_cast<std::streamsize>(raw.size()));
  if (f.gcount() != static_cast<std::streamsize>(raw.size())) {
    return false;
  }

  grid.width = static_cast<int>(shape[0]);
  grid.height = static_cast<int>(shape[1]);
  grid.z.resize(count);
  for (size_t k = 0; k < count; ++k) {
    double v = std::numeric_limits<double>::quiet_NaN();
    if (descr == "<f8") {
      std::memcpy(&v, &raw[k * 8], 8);
    } else {
      float fv = 0.0F;
      std::memcpy(&fv, &raw[k * 4], 4);
      v = fv;
    }
    grid.z[k] = v;
  }
  grid.valid = true;
  return true;
}

// Elevation at grid coordinates (u, v): bilinear over the four surrounding
// cells, NaN cells dropped and the weights renormalised. When no corner is
// finite (path point off the walkable area) the nearest finite cell within a
// 2-cell box answers; NaN when even that finds nothing.
double lookupElevation(const ElevationGrid & g, double u, double v)
{
  static constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
  if (!g.valid) {
    return kNan;
  }
  const auto at = [&g](int uu, int vv) -> double {
    if (uu < 0 || uu >= g.width || vv < 0 || vv >= g.height) {
      return kNan;
    }
    return g.z[static_cast<size_t>(uu) * g.height + vv];
  };

  const int u0 = static_cast<int>(std::floor(u));
  const int v0 = static_cast<int>(std::floor(v));
  const double du = u - u0;
  const double dv = v - v0;
  const double w[4] = {(1.0 - du) * (1.0 - dv), du * (1.0 - dv), (1.0 - du) * dv, du * dv};
  const int us[4] = {u0, u0 + 1, u0, u0 + 1};
  const int vs[4] = {v0, v0, v0 + 1, v0 + 1};
  double sum = 0.0;
  double wsum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const double val = at(us[k], vs[k]);
    if (std::isfinite(val)) {
      sum += w[k] * val;
      wsum += w[k];
    }
  }
  if (wsum > 1e-6) {
    return sum / wsum;
  }

  double best_d = std::numeric_limits<double>::max();
  double best = kNan;
  for (int r = 1; r <= 2 && !std::isfinite(best); ++r) {
    for (int uu = u0 - r; uu <= u0 + r + 1; ++uu) {
      for (int vv = v0 - r; vv <= v0 + r + 1; ++vv) {
        const double val = at(uu, vv);
        if (!std::isfinite(val)) {
          continue;
        }
        const double dx = uu - u;
        const double dy = vv - v;
        const double d = dx * dx + dy * dy;
        if (d < best_d) {
          best_d = d;
          best = val;
        }
      }
    }
  }
  return best;
}

}  // namespace

class PrmPlannerNode : public rclcpp::Node
{
public:
  PrmPlannerNode()
  : Node("prm_planner_node")
  {
    // --- I/O & behaviour ---
    frame_id_ = declare_parameter<std::string>("frame_id", "map");
    output_path_topic_ = declare_parameter<std::string>("output_path_topic", "/global_path");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/Odometry");
    goal_topic_ = declare_parameter<std::string>("goal_topic", "/goal_pose");
    clicked_point_topic_ = declare_parameter<std::string>("clicked_point_topic", "/clicked_point");
    start_from_odom_ = declare_parameter<bool>("start_from_odom", true);
    path_height_offset_ = declare_parameter<double>("path_height_offset", 0.0);

    // --- maps (flattened from the floors: list by the launch files) ---
    floor_maps_ = declare_parameter<std::vector<std::string>>("floor_maps", {});
    floor_z_mins_ = declare_parameter<std::vector<double>>("floor_z_mins", {});
    floor_z_maxs_ = declare_parameter<std::vector<double>>("floor_z_maxs", {});
    // Optional per-floor walking-surface height (floors[].ground_z, flattened
    // by the launch files). Entries that are omitted/NaN are estimated from
    // the PCD at startup. Anchors path-Z queries to the floor's own ground so
    // other floors' surfaces in the (wide) z band cannot be picked up.
    floor_ground_zs_ = declare_parameter<std::vector<double>>("floor_ground_zs", {});
    // Per-floor 2.5D elevation grids (floors[].elevation_npy, flattened by the
    // launch files) — see the file header. Used when z_source is "npy".
    floor_elevation_npys_ = declare_parameter<std::vector<std::string>>("floor_elevation_npys", {});
    connections_json_ = declare_parameter<std::string>("connections_json", "");

    // --- path Z source: "npy" (offline elevation grids) or "pcd" (direct
    // per-point PCD queries). Floors without a loadable grid fall back to the
    // PCD source, then to the floor's ground / z_min. ---
    z_source_ = declare_parameter<std::string>("z_source", "pcd");

    // --- 3D map for path Z ---
    pcd_map_path_ = declare_parameter<std::string>("pcd_map_path", "");
    pcd_search_radius_ = declare_parameter<double>("pcd_search_radius", 0.2);
    pcd_z_quantile_ = declare_parameter<double>("pcd_z_quantile", 0.25);
    pcd_z_cluster_tol_ = declare_parameter<double>("pcd_z_cluster_tol", 0.15);
    pcd_ground_tol_below_ = declare_parameter<double>("pcd_ground_tol_below", 0.6);
    pcd_ground_tol_above_ = declare_parameter<double>("pcd_ground_tol_above", 0.9);

    // --- pixel <-> metric transform ---
    map_resolution_ = declare_parameter<double>("map_resolution", 0.05);
    map_origin_x_ = declare_parameter<double>("map_origin_x", -23.373531);
    map_origin_y_ = declare_parameter<double>("map_origin_y", -20.128895);
    flip_y_ = declare_parameter<bool>("flip_y", true);

    // --- PRM-3D parameters ---
    expansion_radius_ = declare_parameter<double>("expansion_radius", 4.0);
    // 动态障碍入图膨胀 (px)：markObstacle 写入 本体+该值 的黑色圆盘，与静态
    // 黑区（膨胀 expansion_radius）同待遇；重规划不重建膨胀+PRM 采样，入图时
    // 一次膨胀到位。默认(-1)对齐 expansion_radius；窄通道绕不过去（未找到
    // 路径）可单独调小。
    dynamic_inflation_px_ = declare_parameter<double>("dynamic_inflation_px", -1.0);
    prm_k_nodes_ = declare_parameter<int>("prm_k_nodes", 500);
    prm_r_nei_ = declare_parameter<double>("prm_r_nei", 25.0);
    fuse_k_neighbors_ = declare_parameter<int>("fuse_k_neighbors", 10);
    fuse_cross_radius_ = declare_parameter<double>("fuse_cross_radius", 8.0);

    // --- path Z post-processing ---
    z_median_filter_window_ = declare_parameter<int>("z_median_filter_window", 7);
    z_max_slope_ = declare_parameter<double>("z_max_slope", 1.5);

    // ---- dynamic-obstacle replanning (safeplanner-2 服务架构) ----
    // 全局端是纯服务端：障碍点云由 local_replan 过滤/降采样后随重规划请求
    // 原子送达（本节点不再自己订阅点云——单一障碍来源，杜绝「触发器看得见、
    // 全局标记不到」的双源不同步）。感知参数全部在 local_replan.yaml。
    enable_dynamic_replan_ = declare_parameter<bool>("enable_dynamic_replan", false);
    replan_service_ = declare_parameter<std::string>("replan_service", "/replan_plan");
    replan_check_rate_ = declare_parameter<double>("replan_check_rate", 1.0);  // 障碍过期检查
    obstacle_cluster_tol_ = declare_parameter<double>("obstacle_cluster_tol", 0.25);
    obstacle_radius_m_ = declare_parameter<double>("obstacle_radius_m", 0.0);
    obstacle_decay_sec_ = declare_parameter<double>("obstacle_decay_sec", 2.0);
    // 入图高程校验 (m)：点 Z 与该层高度图 npy 在此像素的地面高程差超过该值
    // 视为天花板/别的楼层的结构（下楼梯时扫到的楼下天花板差 ~2m），不入图。
    // 仅 z_source=npy 且该层网格有效时生效；设负数关闭。
    obstacle_elev_tol_ = declare_parameter<double>("obstacle_elev_tol", 0.2);
    // safeplanner-2 风格调试窗口（OpenCV）：当前楼层地图 + 全局路径 + 过滤后
    // 障碍点云 + 动态障碍圆盘 + 机器人。随 launch 一起启动。
    replan_debug_window_ = declare_parameter<bool>("replan_debug_window", true);

    // ---- wall (obstacle-proximity) penalty ----
    wall_penalty_gain_ = declare_parameter<double>("wall_penalty_gain", 0.0);
    wall_penalty_clearance_px_ = declare_parameter<double>("wall_penalty_clearance_px", 10.0);

    if (map_resolution_ <= 0.0) {
      RCLCPP_WARN(get_logger(), "map_resolution <= 0, forcing to 1.0");
      map_resolution_ = 1.0;
    }

    // --- publishers ---
    auto latched_qos = rclcpp::QoS(1).transient_local().reliable();
    path_pub_ = create_publisher<nav_msgs::msg::Path>(output_path_topic_, latched_qos);
    path_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(
      output_path_topic_ + "_marker", 10);
    start_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(
      output_path_topic_ + "_start", 10);
    goal_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(
      output_path_topic_ + "_goal", 10);

    // --- subscriptions ---
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10,
      std::bind(&PrmPlannerNode::onOdom, this, std::placeholders::_1));
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      goal_topic_, 10,
      std::bind(&PrmPlannerNode::onGoalPose, this, std::placeholders::_1));
    clicked_point_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      clicked_point_topic_, 10,
      std::bind(&PrmPlannerNode::onClickedPoint, this, std::placeholders::_1));

    // --- dynamic-obstacle replanning (service server) ---
    if (enable_dynamic_replan_) {
      tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
      replan_srv_ = create_service<prm_interfaces::srv::ReplanPlan>(
        replan_service_,
        std::bind(&PrmPlannerNode::onReplanPlan, this,
                  std::placeholders::_1, std::placeholders::_2));
      auto period = std::chrono::milliseconds(
        static_cast<int64_t>(1000.0 / std::max(0.1, replan_check_rate_)));
      replan_timer_ = create_wall_timer(
        period, std::bind(&PrmPlannerNode::onDecayCheck, this));
      RCLCPP_INFO(get_logger(),
        "dynamic replanning service on %s (decay check %.1f Hz, memory %.1f s)",
        replan_service_.c_str(), replan_check_rate_, obstacle_decay_sec_);
    } else {
      RCLCPP_INFO(get_logger(), "dynamic replanning disabled (enable_dynamic_replan=false)");
    }

    // --- build maps + roadmap ---
    map_ready_ = buildMaps();
    if (!map_ready_) {
      RCLCPP_ERROR(get_logger(),
        "Failed to initialise PRM maps/roadmap; node will not plan. "
        "Check floor_maps / connections_json paths.");
      return;
    }
    RCLCPP_INFO(get_logger(), "PRM-3D planner ready (%d floors).", num_floors_);

    // --- safeplanner-2 风格的重规划调试窗口：地图 + 路径 + 障碍点云 ---
    if (replan_debug_window_ && enable_dynamic_replan_ && std::getenv("DISPLAY") != nullptr) {
      debug_running_ = true;
      debug_thread_ = std::thread(&PrmPlannerNode::debugLoop, this);
      debug_goal_timer_ = create_wall_timer(
        std::chrono::milliseconds(50),
        std::bind(&PrmPlannerNode::processPendingDebugGoal, this));
      RCLCPP_INFO(get_logger(), "replan debug window enabled ('PRM Replan Debug')");
    } else if (replan_debug_window_ && enable_dynamic_replan_) {
      RCLCPP_WARN(get_logger(), "replan_debug_window=true but $DISPLAY is unset; window disabled");
    }
  }

  ~PrmPlannerNode() override
  {
    debug_running_ = false;
    if (debug_thread_.joinable()) {
      debug_thread_.join();
    }
  }

private:
  // ---------------------------------------------------------------- init
  bool buildMaps()
  {
    if (floor_maps_.empty()) {
      RCLCPP_ERROR(get_logger(), "floor_maps is empty");
      return false;
    }
    num_floors_ = static_cast<int>(floor_maps_.size());
    active_obs_pts_.assign(num_floors_, {});   // 点级动态障碍：每层一份存活点集

    // Build per-floor BImap (same recipe as sim2d.cpp).
    floor_maps_im_.resize(num_floors_);
    for (int i = 0; i < num_floors_; ++i) {
      cv::Mat img = cv::imread(floor_maps_[i], cv::IMREAD_GRAYSCALE);
      if (img.empty()) {
        RCLCPP_ERROR(get_logger(), "cannot load floor map: %s", floor_maps_[i].c_str());
        return false;
      }
      BImap & m = floor_maps_im_[i];
      m.shapeX = img.cols;
      m.shapeY = img.rows;
      m.mapratio = map_resolution_;
      m.robotsize = 0.9;

      cv::Mat obstacle_mask = (img <= 128);
      int ksz = static_cast<int>(2.0 * expansion_radius_ + 1.0);
      cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksz, ksz));
      cv::Mat dilated;
      cv::dilate(obstacle_mask, dilated, kernel);
      m.BIimap = cv::Mat::zeros(img.rows, img.cols, CV_16U);
      m.BIimap.setTo(cv::Scalar(0xC000), dilated);
      cv::cvtColor(img, m.BIdmap, cv::COLOR_GRAY2BGR);
    }

    map_width_ = static_cast<int>(floor_maps_im_[0].shapeX);
    map_height_ = static_cast<int>(floor_maps_im_[0].shapeY);

    // Normalise per-floor Z ranges into (min, max) pairs.
    z_ranges_.clear();
    z_ranges_.reserve(num_floors_);
    for (int i = 0; i < num_floors_; ++i) {
      double a = (i < static_cast<int>(floor_z_mins_.size())) ? floor_z_mins_[i]
                                                              : static_cast<double>(i);
      double b = (i < static_cast<int>(floor_z_maxs_.size())) ? floor_z_maxs_[i]
                                                              : static_cast<double>(i + 1);
      z_ranges_.push_back({std::min(a, b), std::max(a, b)});
    }

    // Load the global PCD map used for path-Z lookups (optional). Without it
    // every path point falls back to the floor's ground / z_min.
    std::vector<PcdPoint> pcd_points;
    bool have_pcd = false;
    if (!pcd_map_path_.empty()) {
      if (!readPcdXyz(pcd_map_path_, pcd_points)) {
        RCLCPP_WARN(get_logger(),
          "failed to read PCD map %s (path Z will fall back to floor ground/z_min)",
          pcd_map_path_.c_str());
      } else if (!buildPcdZIndex(pcd_points, 0.25, pcd_index_)) {
        RCLCPP_WARN(get_logger(),
          "failed to index PCD map %s (path Z will fall back to floor ground/z_min)",
          pcd_map_path_.c_str());
      } else {
        have_pcd = true;
        RCLCPP_INFO(get_logger(), "indexed %zu PCD points from %s for path-Z lookups",
          pcd_points.size(), pcd_map_path_.c_str());
      }
    } else {
      RCLCPP_WARN(get_logger(), "pcd_map_path is empty; path Z will fall back to floor ground/z_min");
    }

    // Load the per-floor elevation grids (z_source "npy"). A grid whose shape
    // does not match its floor map is rejected: indexing it would silently
    // read wrong cells.
    floor_grids_.assign(num_floors_, ElevationGrid{});
    int grids_loaded = 0;
    for (int i = 0; i < num_floors_; ++i) {
      const std::string & npy = (i < static_cast<int>(floor_elevation_npys_.size()))
        ? floor_elevation_npys_[i]
        : std::string();
      if (npy.empty()) {
        continue;
      }
      if (!readNpyGrid(npy, floor_grids_[i])) {
        RCLCPP_WARN(get_logger(), "floor %d: cannot read elevation grid %s", i, npy.c_str());
        floor_grids_[i] = ElevationGrid{};
        continue;
      }
      if (floor_grids_[i].width != floor_maps_im_[i].shapeX ||
        floor_grids_[i].height != floor_maps_im_[i].shapeY)
      {
        RCLCPP_WARN(get_logger(),
          "floor %d: elevation grid %s is %dx%d but map is %dx%d — grid ignored",
          i, npy.c_str(), floor_grids_[i].width, floor_grids_[i].height,
          floor_maps_im_[i].shapeX, floor_maps_im_[i].shapeY);
        floor_grids_[i] = ElevationGrid{};
        continue;
      }
      ++grids_loaded;
    }

    // Resolve which source answers path Z on each floor: elevation grid when
    // available (z_source "npy"), else the PCD query, else ground/z_min.
    floor_use_npy_.assign(num_floors_, false);
    if (z_source_ == "npy") {
      int npy_floors = 0;
      for (int i = 0; i < num_floors_; ++i) {
        if (floor_grids_[i].valid) {
          floor_use_npy_[i] = true;
          ++npy_floors;
        } else {
          RCLCPP_WARN(get_logger(),
            "floor %d: no elevation grid (z_source=npy); its path Z falls back to %s",
            i, pcd_index_.valid ? "PCD queries" : "ground/z_min");
        }
      }
      if (npy_floors == 0) {
        RCLCPP_WARN(get_logger(),
          "z_source=npy but no elevation grid loaded — generate them with "
          "process_multi_floor_elevations.py; every floor falls back to %s",
          pcd_index_.valid ? "PCD queries" : "ground/z_min");
      } else {
        RCLCPP_INFO(get_logger(), "path Z source: elevation grids (%d/%d floors)",
          npy_floors, num_floors_);
      }
    } else {
      RCLCPP_INFO(get_logger(),
        "path Z source: direct PCD queries (%d elevation grids loaded but unused)",
        grids_loaded);
    }

    // Resolve each floor's walking-surface height: explicit floor_ground_zs
    // entries win; the rest are estimated from the PCD (dominant Z peak inside
    // the floor map's free area).
    resolveFloorGroundZ(have_pcd ? &pcd_points : nullptr);

    // Initialise planner (same sequence as sim2d.cpp).
    planner_.initialize(num_floors_);
    planner_.setPRMParams(prm_k_nodes_, prm_r_nei_);
    planner_.setWallPenalty(wall_penalty_gain_, wall_penalty_clearance_px_);
    // 动态障碍入图膨胀 (px)：入图(markObstacle)即写 本体+该值 的黑色圆盘，
    // 与静态黑区（加载时膨胀 expansion_radius）同待遇，planFused 直接用该半径
    // 切边/软惩罚。dynamic_inflation_px 缺省(-1)跟随静态 expansion_radius。
    dyn_inflate_px_ =
      (dynamic_inflation_px_ >= 0.0 ? dynamic_inflation_px_ : expansion_radius_) +
      std::max(0.0, obstacle_radius_m_) / map_resolution_;
    BIgraph empty_graph;
    for (int i = 0; i < num_floors_; ++i) {
      planner_.setFloorPRMParams(i, prm_k_nodes_, prm_r_nei_);
      planner_.loadFloorGraph(i, empty_graph);
      planner_.setMapReference(i, &floor_maps_im_[i]);
    }
    for (int i = 0; i < num_floors_; ++i) {
      planner_.buildRoadmap(i);
    }
    RCLCPP_INFO(get_logger(), "built roadmaps for %d floors", num_floors_);

    if (connections_json_.empty()) {
      RCLCPP_WARN(get_logger(), "connections_json is empty; no cross-floor edges will exist");
      return true;
    }
    if (!planner_.loadConnectionsJson(connections_json_)) {
      RCLCPP_ERROR(get_logger(), "loadConnectionsJson failed: %s", connections_json_.c_str());
      return false;
    }
    planner_.fuseGraph();
    planner_.setFuseParams(fuse_k_neighbors_, fuse_cross_radius_);
    RCLCPP_INFO(get_logger(), "fused multi-floor graph ready");
    return true;
  }

  // ------------------------------------------------- per-floor ground height
  // Fill floor_ground_z_: explicit floor_ground_zs_ entries first, then the
  // PCD-based estimate for every floor left as NaN.
  void resolveFloorGroundZ(const std::vector<PcdPoint> * points)
  {
    floor_ground_z_.assign(num_floors_, std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < floor_ground_zs_.size() && i < floor_ground_z_.size(); ++i) {
      floor_ground_z_[i] = floor_ground_zs_[i];
    }
    if (points == nullptr) {
      for (int i = 0; i < num_floors_; ++i) {
        if (!std::isnan(floor_ground_z_[i])) {
          RCLCPP_INFO(get_logger(), "floor %d ground Z = %.2f m (from config)",
            i, floor_ground_z_[i]);
        } else {
          RCLCPP_WARN(get_logger(),
            "floor %d has no ground height (no PCD loaded); path Z falls back to its z band", i);
        }
      }
      return;
    }
    for (int i = 0; i < num_floors_; ++i) {
      if (!std::isnan(floor_ground_z_[i])) {
        RCLCPP_INFO(get_logger(), "floor %d ground Z = %.2f m (from config)",
          i, floor_ground_z_[i]);
        continue;
      }
      floor_ground_z_[i] = estimateFloorGroundZ(i, *points);
    }
  }

  // The walking surface of floor `f` is the dominant Z peak among PCD points
  // that lie inside the floor map's free area (not obstacle, not inflated) and
  // inside the floor's [z_min, z_max] band: the floor's own ground covers far
  // more free pixels than furniture, stray points, or (where z bands overlap)
  // other floors' surfaces seen through shared openings. Histogram bins are
  // 0.1 m; the peak is the densest 0.3 m window, reported as a weighted mean.
  double estimateFloorGroundZ(int f, const std::vector<PcdPoint> & points)
  {
    const BImap & m = floor_maps_im_[f];
    const auto & band = z_ranges_[f];
    const double bin_w = 0.1;
    const int nbins = std::max(
      1, static_cast<int>(std::ceil((band.second - band.first) / bin_w)));
    std::vector<size_t> hist(static_cast<size_t>(nbins), 0);
    size_t total = 0;
    for (const PcdPoint & p : points) {
      if (p.z < band.first || p.z > band.second) {
        continue;
      }
      const double px = (p.x - map_origin_x_) / map_resolution_;
      const double py = flip_y_
        ? (map_height_ - 1) - (p.y - map_origin_y_) / map_resolution_
        : (p.y - map_origin_y_) / map_resolution_;
      const int ix = static_cast<int>(std::lround(px));
      const int iy = static_cast<int>(std::lround(py));
      if (ix < 0 || ix >= m.shapeX || iy < 0 || iy >= m.shapeY) {
        continue;
      }
      if ((m.BIimap.at<uint16_t>(iy, ix) & 0xC000) != 0) {
        continue;  // obstacle or inflated
      }
      ++hist[static_cast<size_t>(
        std::min(nbins - 1, static_cast<int>((p.z - band.first) / bin_w)))];
      ++total;
    }
    const int win = std::max(1, static_cast<int>(std::lround(0.3 / bin_w)));
    std::vector<size_t> pre(hist.size() + 1, 0);
    for (size_t b = 0; b < hist.size(); ++b) {
      pre[b + 1] = pre[b] + hist[b];
    }
    size_t best_cnt = 0;
    int best_b = 0;
    for (int b = 0; b + win <= nbins; ++b) {
      const size_t cnt = pre[static_cast<size_t>(b + win)] - pre[static_cast<size_t>(b)];
      if (cnt > best_cnt) {
        best_cnt = cnt;
        best_b = b;
      }
    }
    if (best_cnt < 200 || total == 0) {
      RCLCPP_WARN(get_logger(),
        "floor %d: too few free-area PCD points (%zu) for a ground estimate; "
        "its z band will be used instead", f, total);
      return std::numeric_limits<double>::quiet_NaN();
    }
    double weighted = 0.0;
    for (int b = best_b; b < best_b + win; ++b) {
      weighted += (band.first + (b + 0.5) * bin_w) *
                  static_cast<double>(hist[static_cast<size_t>(b)]);
    }
    const double ground = weighted / static_cast<double>(best_cnt);
    RCLCPP_INFO(get_logger(),
      "floor %d ground Z = %.2f m (estimated: peak %.2f-%.2f holds %zu/%zu free-area pts)",
      f, ground, band.first + best_b * bin_w, band.first + (best_b + win) * bin_w,
      best_cnt, total);
    return ground;
  }

  // ------------------------------------------------------------- helpers
  // Floor for a pose, by z interval: the first floor whose [z_min, z_max]
  // contains z wins (checked in floor order, so earlier floors own shared
  // boundaries); a z in a gap snaps to the nearest band edge.
  int floorFromZ(double z) const
  {
    int best = 0;
    double best_d = std::numeric_limits<double>::max();
    for (int i = 0; i < static_cast<int>(z_ranges_.size()); ++i) {
      const auto & r = z_ranges_[i];
      if (z >= r.first && z <= r.second) {
        return i;
      }
      double d = (z < r.first) ? (r.first - z) : (z - r.second);
      if (d < best_d) {
        best_d = d;
        best = i;
      }
    }
    return best;
  }

  double zMid(int floor) const
  {
    if (z_ranges_.empty()) {
      return 0.0;
    }
    int f = std::clamp(floor, 0, static_cast<int>(z_ranges_.size()) - 1);
    return (z_ranges_[f].first + z_ranges_[f].second) / 2.0;
  }

  // Fallback height for a floor when the PCD has no support near a path point:
  // its ground height when known, else the band's z_min.
  double defaultZ(int floor) const
  {
    if (z_ranges_.empty()) {
      return 0.0;
    }
    int f = std::clamp(floor, 0, static_cast<int>(z_ranges_.size()) - 1);
    if (f < static_cast<int>(floor_ground_z_.size()) &&
      !std::isnan(floor_ground_z_[static_cast<size_t>(f)]))
    {
      return floor_ground_z_[static_cast<size_t>(f)];
    }
    return z_ranges_[f].first;
  }

  BIpoint metricToPixel(double x, double y, double z) const
  {
    return metricToPixelOnFloor(x, y, floorFromZ(z));
  }

  BIpoint metricToPixelOnFloor(double x, double y, int floor) const
  {
    double px = (x - map_origin_x_) / map_resolution_;
    double py = flip_y_
      ? (map_height_ - 1) - (y - map_origin_y_) / map_resolution_
      : (y - map_origin_y_) / map_resolution_;
    const int f = num_floors_ > 0 ? std::clamp(floor, 0, num_floors_ - 1) : floor;
    return BIpoint{px, py, f};
  }

  void pixelToMetric(const BIpoint & p, double & x, double & y, double & z) const
  {
    x = map_origin_x_ + p.x * map_resolution_;
    y = flip_y_
      ? map_origin_y_ + (map_height_ - 1 - p.y) * map_resolution_
      : map_origin_y_ + p.y * map_resolution_;
    z = zMid(p.floor);
  }

  // Assign a ground-height Z to every point of a 2D PRM path.
  //
  // npy mode: each point is looked up (bilinearly) in its floor's elevation
  // grid; the grid shares the map's pixel frame, so the lookup coordinates are
  // the point's own pixel x and the y-unflipped v = H-1-p.y. A NaN there
  // (point off the walkable area) drops through to the PCD query below, so a
  // missing grid cell still gets the best available answer.
  //
  // PCD mode (and fallback): the point's metric (x, y) is queried against the
  // PCD map inside a tight window around its floor's ground height
  // (floor_ground_z_ ± pcd_ground_tol_below_/above_): the window keeps other
  // floors' surfaces — visible wherever the z bands overlap — and ceilings out
  // of the candidate set. Inside the window the value is the densest-Z-cluster
  // + pcd_z_quantile_ quantile of points within pcd_search_radius_ (radius
  // escalates x2/x4). Points off the window (mid-staircase) fall back to one
  // full-band query so stair-surface points are used; points that still find
  // nothing inherit the previous valid Z, then the floor's ground / z_min.
  std::vector<double> assignPathZ(const std::vector<BIpoint> & path) const
  {
    std::vector<double> zs(path.size());
    double last = 0.0;
    bool has_last = false;
    for (size_t i = 0; i < path.size(); ++i) {
      const BIpoint & p = path[i];
      double mx = 0.0;
      double my = 0.0;
      double mz = 0.0;
      pixelToMetric(p, mx, my, mz);

      double z = std::numeric_limits<double>::quiet_NaN();
      if (p.floor >= 0 && p.floor < num_floors_ && floor_use_npy_[p.floor]) {
        // Elevation-grid lookup: u is the map column, v the unflipped row.
        const double v = flip_y_ ? (map_height_ - 1) - p.y : p.y;
        z = lookupElevation(floor_grids_[p.floor], p.x, v);
      }
      if (std::isnan(z) && pcd_index_.valid && p.floor >= 0 && p.floor < static_cast<int>(z_ranges_.size())) {
        const auto & band = z_ranges_[p.floor];
        const double ground =
          (p.floor < static_cast<int>(floor_ground_z_.size()))
          ? floor_ground_z_[static_cast<size_t>(p.floor)]
          : std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(ground)) {
          // No ground anchor for this floor: query the whole z band.
          double r = pcd_search_radius_;
          for (int attempt = 0; attempt < 3 && std::isnan(z); ++attempt, r *= 2.0) {
            z = queryQuantileZ(
              pcd_index_, mx, my, r, band.first, band.second, pcd_z_quantile_,
              pcd_z_cluster_tol_);
          }
        } else {
          // Anchored query: a tight window around this floor's own ground.
          double r = pcd_search_radius_;
          for (int attempt = 0; attempt < 3 && std::isnan(z); ++attempt, r *= 2.0) {
            z = queryQuantileZ(
              pcd_index_, mx, my, r, ground - pcd_ground_tol_below_,
              ground + pcd_ground_tol_above_, pcd_z_quantile_, pcd_z_cluster_tol_);
          }
          if (std::isnan(z)) {
            // Staircases leave the ground window mid-flight; one full-band
            // query lets the stair's own surface points answer.
            z = queryQuantileZ(
              pcd_index_, mx, my, pcd_search_radius_ * 2.0, band.first, band.second,
              pcd_z_quantile_, pcd_z_cluster_tol_);
          }
        }
      }
      if (std::isnan(z)) {
        z = has_last ? last : defaultZ(p.floor);
      }
      last = z;
      has_last = true;
      zs[i] = z;
    }
    return zs;
  }

  // Post-process the per-point Z values so the path never makes sharp vertical
  // jumps. Two steps:
  //   1. sliding-window median — kills isolated outliers from the elevation
  //      grids (stray PCD points / wrong z-band pixels show up as single-point
  //      spikes into the sky or underground);
  //   2. slope clamp — |dz| between consecutive points is limited to
  //      z_max_slope * horizontal distance, so any remaining step (e.g. a
  //      floor-band gap at a stairwell) becomes a ramp instead of a wall.
  void postProcessZ(std::vector<double> & zs, const std::vector<BIpoint> & path) const
  {
    const int n = static_cast<int>(zs.size());
    if (n < 3) {
      return;
    }

    // 1. sliding-window median (edges use a clipped window)
    int window = z_median_filter_window_;
    if (window >= 3) {
      if (window % 2 == 0) {
        --window;
      }
      const int half = window / 2;
      std::vector<double> filtered(n);
      std::vector<double> buf;
      buf.reserve(window);
      for (int i = 0; i < n; ++i) {
        buf.clear();
        const int lo = std::max(0, i - half);
        const int hi = std::min(n - 1, i + half);
        for (int j = lo; j <= hi; ++j) {
          buf.push_back(zs[j]);
        }
        std::sort(buf.begin(), buf.end());
        filtered[i] = buf[buf.size() / 2];
      }
      zs = filtered;
    }

    // 2. slope clamp, alternating forward/backward passes so a jump is spread
    //    symmetrically onto its neighbours instead of only shifted forward
    if (z_max_slope_ > 0.0) {
      for (int round = 0; round < 2; ++round) {
        for (int i = 1; i < n; ++i) {
          const double dx = std::hypot(
            path[i].x - path[i - 1].x, path[i].y - path[i - 1].y) * map_resolution_;
          const double dz = zs[i] - zs[i - 1];
          const double limit = z_max_slope_ * std::max(dx, 1e-6);
          if (std::abs(dz) > limit) {
            zs[i] = zs[i - 1] + (dz > 0.0 ? limit : -limit);
          }
        }
        for (int i = n - 2; i >= 0; --i) {
          const double dx = std::hypot(
            path[i].x - path[i + 1].x, path[i].y - path[i + 1].y) * map_resolution_;
          const double dz = zs[i] - zs[i + 1];
          const double limit = z_max_slope_ * std::max(dx, 1e-6);
          if (std::abs(dz) > limit) {
            zs[i] = zs[i + 1] + (dz > 0.0 ? limit : -limit);
          }
        }
      }
    }
  }

  // ------------------------------------------------------- callbacks
  void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // Just cache the pose: planning is triggered by goals and the low-rate
    // replan timer, not by every odometry message.
    latest_odom_ = msg;
    has_start_ = true;
  }

  void onGoalPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    const auto & pos = msg->pose.position;
    goal_ = metricToPixel(pos.x, pos.y, pos.z);
    has_goal_ = true;
    RCLCPP_INFO(get_logger(), "[发布终点] (%.2f, %.2f, %.2f) -> 楼层 F%d",
      pos.x, pos.y, pos.z, goal_.floor);
    publishPoseMarker(goal_, "goal", makeColor(1.0f, 0.0f, 0.0f, 1.0f), goal_marker_pub_);
    refreshDebugGoalSnap();
    planIfReady();
  }

  void onClickedPoint(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    // Two-click mode: first click sets start, second sets goal.
    BIpoint pt = metricToPixel(msg->point.x, msg->point.y, msg->point.z);
    if (!has_start_ || start_from_odom_) {
      start_ = pt;
      has_start_ = true;
      start_from_odom_ = false;
      publishPoseMarker(start_, "start", makeColor(0.0f, 1.0f, 0.0f, 1.0f), start_marker_pub_);
      RCLCPP_INFO(get_logger(), "clicked start -> floor %d, pixel (%.1f, %.1f)",
        start_.floor, start_.x, start_.y);
    } else {
      goal_ = pt;
      has_goal_ = true;
      publishPoseMarker(goal_, "goal", makeColor(1.0f, 0.0f, 0.0f, 1.0f), goal_marker_pub_);
      refreshDebugGoalSnap();
      RCLCPP_INFO(get_logger(), "clicked goal -> floor %d, pixel (%.1f, %.1f)",
        goal_.floor, goal_.x, goal_.y);
      planIfReady();
    }
  }

  // ---------------------------------------------- dynamic-obstacle replanning
  // 障碍点云唯一入口 = ReplanPlan 服务请求（local_replan 已做 Z 带/感知半径/
  // 栅格降采样，map 帧）——本节点不再自己订阅点云：单一障碍来源，杜绝
  // 「触发器看得见、全局标记不到」的双源不同步。
  //
  // 点级动态障碍（参考 safeplanner-2 updateDynamicObstacles）：不做圆盘拟合
  // ——局部观测只扫到障碍物边角时，拟合理的圆心会把禁行区放错位置。每个点
  // 独立维护：与本层老点距离 < obstacle_cluster_tol → 刷新时间戳（还活着）；
  // 否则作为新点加入。入图时以每个点为圆心 mark 一个 forbid 半径
  //（dyn_inflate_px_）的黑色圆盘——多点圆盘自然并成障碍真实形状，边角不会
  // 错位。记忆按点过期（expire_sec 随请求可覆盖：cmu_replan 传短记忆，
  // local_replan 不传用服务端默认），过期/新增后清动态标记重打全部存活点。
  void ingestObstacles(const sensor_msgs::msg::PointCloud2 & cloud,
                       const geometry_msgs::msg::PoseStamped & start,
                       double decay_sec)
  {
    if (!map_ready_) return;
    const auto now = get_clock()->now();
    // 本批点的记忆时长：请求 decay_sec>0 用请求值，否则用服务端 obstacle_decay_sec_
    const double expire_sec = decay_sec > 0.0 ? decay_sec : obstacle_decay_sec_;

    // 1. 请求点云应在 map 帧；不是则 TF 变换（时间戳查不到时回退最新变换，
    //    与 local_replan 的做法一致——雷达时间戳略有超前时不至于整帧丢弃）
    sensor_msgs::msg::PointCloud2 cloud_map;
    if (!cloud.header.frame_id.empty() && cloud.header.frame_id != frame_id_) {
      geometry_msgs::msg::TransformStamped tf;
      try {
        tf = tf_buffer_->lookupTransform(
          frame_id_, cloud.header.frame_id, cloud.header.stamp,
          std::chrono::milliseconds(50));
      } catch (const tf2::TransformException & ex) {
        try {  // 回退：取最新的可用变换
          tf = tf_buffer_->lookupTransform(
            frame_id_, cloud.header.frame_id, tf2::TimePointZero);
        } catch (const tf2::TransformException & ex2) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
            "replan request cloud frame '%s' -> '%s' TF unavailable; obstacles ignored",
            cloud.header.frame_id.c_str(), frame_id_.c_str());
          return;
        }
      }
      tf2::doTransform(cloud, cloud_map, tf);
    } else {
      cloud_map = cloud;
    }

    // 2. 机器人位姿 / 楼层（像素）——观测落在哪层（起点 Z 带判定）就写进哪层
    const auto & pos = start.pose.position;
    const int floor = floorFromZ(pos.z);
    const BIpoint robot_px = metricToPixel(pos.x, pos.y, pos.z);
    const auto & quat = start.pose.orientation;
    const double robot_yaw = std::atan2(
      2.0 * (quat.w * quat.z + quat.x * quat.y),
      1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z));

    // 3. 逐点归属（local_replan 已过滤/降采样，服务端三步定每个点进哪层图）：
    //    ① Z 带定候选层：点 Z 落在哪些层的 [z_min, z_max] 内（楼梯带重叠 → 可能多层）；
    //    ② npy 高程定归属：候选层中该像素地面高程与点 Z 差最小、且 ≤
    //       obstacle_elev_tol 者胜出 —— 机器人还在楼上时，楼下地面上的障碍物
    //       就能写进楼下层（不用等机器人自己过界）；全部不匹配（天花板/异层
    //       结构，Δ~2m）或无候选（层间缝隙）→ 剔除；
    //    ③ 静态空白校验：归属层该像素在原地图中是空白（可通行）区域才入图，
    //       墙/栏杆回波不算动态障碍。
    //    obstacle_elev_tol < 0 关闭逐点归属，整批按机器人层（旧行为）；楼层全
    //    无 npy（pcd 模式）时同样退回旧行为。
    bool npy_any = false;
    for (bool b : floor_use_npy_) npy_any = npy_any || b;
    const bool per_point = obstacle_elev_tol_ >= 0.0 && npy_any;
    struct KeptPt { float px; float py; int fl; };
    std::vector<KeptPt> kept;
    std::vector<cv::Point2f> pts_px;      // 机器人楼层的本批点（调试窗口显示）
    int rejected = 0;
    for (sensor_msgs::PointCloud2ConstIterator<float> it(cloud_map, "x"); it != it.end(); ++it) {
      const float pxf = it[0], pyf = it[1], pzf = it[2];
      if (!std::isfinite(pxf) || !std::isfinite(pyf) || !std::isfinite(pzf)) continue;
      const BIpoint pp = metricToPixel(static_cast<double>(pxf), static_cast<double>(pyf),
                                       static_cast<double>(pzf));
      int fl = floor;
      if (per_point) {
        fl = -1;
        double best_d = std::numeric_limits<double>::max();
        for (int f = 0; f < num_floors_; ++f) {
          // ① Z 带候选（重叠带内多层都进候选，交给 ② 分辨）
          const auto & band = z_ranges_[f];
          if (pzf < band.first || pzf > band.second) continue;
          if (!floor_use_npy_[f]) continue;
          const double v = flip_y_ ? (map_height_ - 1) - pp.y : pp.y;  // npy 行不翻转
          const double zsurf = lookupElevation(floor_grids_[f], pp.x, v);
          if (!std::isfinite(zsurf)) continue;   // 该像素不在 f 层可通行区
          // ② 高程差最小者胜
          const double d = std::fabs(pzf - zsurf);
          if (d < best_d) {
            best_d = d;
            fl = f;
          }
        }
        if (fl < 0 || best_d > obstacle_elev_tol_) {  // 无归属或天花板/异层
          ++rejected;
          continue;
        }
      }
      // ③ 归属层静态空白校验
      if (!planner_.isPointTraversableStatic(fl, pp.x, pp.y)) {
        ++rejected;
        continue;
      }
      kept.push_back({static_cast<float>(pp.x), static_cast<float>(pp.y), fl});
      if (fl == floor) pts_px.emplace_back(static_cast<float>(pp.x), static_cast<float>(pp.y));
    }

    // 4. 过期 + 逐点折叠进归属层的记忆（同层内距离 < match_px → 刷新；否则
    //    新增），dirty 时清动态标记逐点膨胀重打
    bool dirty = expireObstacles(now);
    const double match_px = obstacle_cluster_tol_ / map_resolution_;
    int added = 0;
    for (const auto & p : kept) {
      auto & fl_pts = active_obs_pts_[p.fl];
      bool matched = false;
      for (auto & q : fl_pts) {
        if (std::hypot(q.px - p.px, q.py - p.py) < match_px) {
          q.stamp = now;  // 仍在视野内，刷新记忆
          q.expire_sec = expire_sec;
          matched = true;
          break;
        }
      }
      if (!matched) {
        fl_pts.push_back({p.px, p.py, now, expire_sec});
        ++added;
        dirty = true;
      }
    }
    if (added > 0 || rejected > 0) {
      std::ostringstream alive;
      for (size_t f = 0; f < active_obs_pts_.size(); ++f)
        if (!active_obs_pts_[f].empty()) alive << " F" << f << "=" << active_obs_pts_[f].size();
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 10000,
        "[障碍入图] 本批 +%d, 剔除 %d (异层/天花板/墙), 存活%s (膨胀 r=%.2f m)",
        added, rejected, alive.str().c_str(), dyn_inflate_px_ * map_resolution_);
    }
    remarkIfDirty(dirty);

    // ---- 调试窗口快照（唯一写入点；窗口线程只读）----
    updateDebugSnap(floor, robot_px, robot_yaw, pts_px);
  }

  // 记忆过期：所有楼层里超过各自 expire_sec（入图时随请求定）未再观测到
  // 的点抹除。返回是否有删减。
  bool expireObstacles(const rclcpp::Time & now)
  {
    bool dirty = false;
    for (auto & fl : active_obs_pts_) {
      for (auto it = fl.begin(); it != fl.end();) {
        if ((now - it->stamp).seconds() > it->expire_sec) {
          it = fl.erase(it);
          dirty = true;
        } else {
          ++it;
        }
      }
    }
    return dirty;
  }

  // dirty 时清动态标记，重打所有楼层的存活点（逐点膨胀入图）
  void remarkIfDirty(bool dirty)
  {
    if (!dirty) return;
    planner_.clearDynamicObstacles();
    for (size_t f = 0; f < active_obs_pts_.size(); ++f) {
      for (const auto & q : active_obs_pts_[f]) {
        planner_.markObstacle(static_cast<int>(f), q.px, q.py, dyn_inflate_px_);
      }
    }
  }

  // 调试窗口快照（唯一写入点，debug_mutex_ 保护；窗口线程只读）。
  // 携带**所有楼层**的记忆障碍点（窗口可切层查看，机器人不在该层也能显示）；
  // cloud_px 为空表示保留上次的点云显示（onDecayCheck 只刷机器人位姿时用）。
  void updateDebugSnap(int floor, const BIpoint & robot_px, double robot_yaw,
                       const std::vector<cv::Point2f> & cloud_px)
  {
    std::lock_guard<std::mutex> lk(debug_mutex_);
    debug_snap_.floor = floor;
    debug_snap_.robot = cv::Point2f(robot_px.x, robot_px.y);
    debug_snap_.yaw = static_cast<float>(robot_yaw);
    if (!cloud_px.empty()) {
      debug_snap_.cloud = cloud_px;
      debug_snap_.cloud_floor = floor;
    }
    debug_snap_.obs_by_floor.assign(active_obs_pts_.size(), {});
    for (size_t f = 0; f < active_obs_pts_.size(); ++f) {
      auto & dst = debug_snap_.obs_by_floor[f];
      dst.reserve(active_obs_pts_[f].size());
      for (const auto & q : active_obs_pts_[f]) dst.emplace_back(q.px, q.py);
    }
    debug_snap_.path = last_path_;
    debug_snap_.has_goal = has_goal_;
    if (has_goal_) {
      debug_snap_.goal_floor = goal_.floor;
      debug_snap_.goal = cv::Point2f(goal_.x, goal_.y);
    } else {
      debug_snap_.goal_floor = -1;
    }
  }

  // 周期任务：只做过期（机器人走后障碍记忆到龄抹除，重规划不再绕已消失的
  // 障碍）+ 用 odom 刷新调试窗口的机器人位姿（cloud 保留最近一次请求的）。
  void onDecayCheck()
  {
    if (!map_ready_) return;
    remarkIfDirty(expireObstacles(get_clock()->now()));
    if (!latest_odom_) return;
    const auto & pos = latest_odom_->pose.pose.position;
    const int fl = floorFromZ(pos.z);
    const auto & quat = latest_odom_->pose.pose.orientation;
    const double yaw = std::atan2(
      2.0 * (quat.w * quat.z + quat.x * quat.y),
      1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z));
    const BIpoint rp = metricToPixel(pos.x, pos.y, pos.z);
    updateDebugSnap(fl, rp, yaw, {});   // 空点云 = 保留上次的点云显示
  }

  // ---------------------------------------------------------- debug window
  static void debugMouseThunk(int event, int x, int y, int flags, void * userdata)
  {
    (void)flags;
    if (event != cv::EVENT_LBUTTONDOWN || userdata == nullptr) {
      return;
    }
    static_cast<PrmPlannerNode *>(userdata)->onDebugMouseClick(x, y);
  }

  void onDebugMouseClick(int screen_x, int screen_y)
  {
    DebugViewState state;
    {
      std::lock_guard<std::mutex> lk(debug_view_mutex_);
      state = debug_view_state_;
    }
    if (!state.valid || state.floor < 0 || state.floor >= num_floors_ || state.scale <= 0.0) {
      return;
    }
    if (screen_x < state.ox || screen_y < state.oy ||
        screen_x >= state.ox + state.draw_cols ||
        screen_y >= state.oy + state.draw_rows)
    {
      return;
    }

    const double px = (static_cast<double>(screen_x - state.ox)) / state.scale;
    const double py = (static_cast<double>(screen_y - state.oy)) / state.scale;
    if (px < 0.0 || py < 0.0 || px >= state.map_cols || py >= state.map_rows) {
      return;
    }

    BIpoint clicked_px{px, py, state.floor};
    double mx = 0.0;
    double my = 0.0;
    double mz_unused = 0.0;
    pixelToMetric(clicked_px, mx, my, mz_unused);
    BIpoint roundtrip_px = metricToPixelOnFloor(mx, my, state.floor);

    {
      std::lock_guard<std::mutex> lk(pending_debug_goal_mutex_);
      pending_debug_goal_px_ = roundtrip_px;
      pending_debug_goal_metric_x_ = mx;
      pending_debug_goal_metric_y_ = my;
      pending_debug_goal_ = true;
    }

    RCLCPP_INFO(get_logger(),
      "[debug点击终点] screen=(%d,%d) -> F%d pixel=(%.1f, %.1f) -> map_xy=(%.3f, %.3f)",
      screen_x, screen_y, roundtrip_px.floor, roundtrip_px.x, roundtrip_px.y, mx, my);
  }

  void processPendingDebugGoal()
  {
    BIpoint goal_px;
    double goal_x = 0.0;
    double goal_y = 0.0;
    {
      std::lock_guard<std::mutex> lk(pending_debug_goal_mutex_);
      if (!pending_debug_goal_) {
        return;
      }
      goal_px = pending_debug_goal_px_;
      goal_x = pending_debug_goal_metric_x_;
      goal_y = pending_debug_goal_metric_y_;
      pending_debug_goal_ = false;
    }

    if (!map_ready_ || goal_px.floor < 0 || goal_px.floor >= num_floors_) {
      return;
    }
    goal_ = goal_px;
    has_goal_ = true;
    publishPoseMarker(goal_, "goal", makeColor(1.0f, 0.0f, 0.0f, 1.0f), goal_marker_pub_);
    if (!planner_.pointTraversable(goal_.floor, goal_.x, goal_.y)) {
      RCLCPP_WARN(get_logger(),
        "[debug点击终点] F%d pixel=(%.1f, %.1f), map_xy=(%.3f, %.3f) 当前不可通行，仍尝试规划",
        goal_.floor, goal_.x, goal_.y, goal_x, goal_y);
    }
    planIfReady();
    refreshDebugGoalSnap();
  }

  void refreshDebugGoalSnap()
  {
    std::lock_guard<std::mutex> lk(debug_mutex_);
    debug_snap_.has_goal = has_goal_;
    if (has_goal_) {
      debug_snap_.goal_floor = goal_.floor;
      debug_snap_.goal = cv::Point2f(goal_.x, goal_.y);
    } else {
      debug_snap_.goal_floor = -1;
    }
    debug_snap_.path = last_path_;
  }

  // safeplanner-2 waypoint_go_to_node::debugLoop 的多楼层版。**显示楼层可切换**：
  // 默认自动跟随机器人所在楼层（状态条 view=F* 带 * 号）；窗口下沿 "floor" 滑条
  // 或数字键 0-9 手动锁定任意楼层（机器人不在该层也能看它的入图/路径），按 'a'
  // 恢复跟随。鼠标左键点击当前地图区域会把该点作为“当前显示楼层”的终点，
  // 走 pixel -> map xy -> pixel 的指定楼层转换后触发规划；不需要输入 z，也不
  // 用 z 反推楼层。滑条画在图像下方的窗口边框区，不遮挡地图。
  // 背景 = 选中楼层静态底图（BIdmap + 静态障碍多边形涂黑），叠加：动态障碍
  // 膨胀区（红实心圆盘 r=dyn_inflate_px_，＝ markObstacle 写进可通行缓存的
  // 禁行区）、障碍本体（黑：该层记忆点+本帧点云，静态已在底图涂黑）、全局
  // 路径（蓝折线，只画选中楼层的段）、机器人（绿点+朝向箭头，仅在它所在
  // 楼层显示）、顶部状态条（楼层/障碍点数/点数/路径点数 + 黑/红图例块）。
  void debugLoop()
  {
    const std::string win = "PRM Replan Debug";
    // 固定窗口尺寸：大图缩下来贴进窗口，小图（≤2x）适当放大，比例不符黑边留空
    const cv::Size win_sz(1280, 800);
    cv::namedWindow(win, cv::WINDOW_NORMAL);
    cv::resizeWindow(win, win_sz.width, win_sz.height);
    cv::setMouseCallback(win, &PrmPlannerNode::debugMouseThunk, this);
    int view_floor = -1;        // -1 = 自动跟随机器人楼层；>=0 = 手动锁定显示该层
    int tb_val = 0;             // 滑条关联值（debugLoop 生命周期内有效）
    bool tb_created = false;
    int tb_last = -1;           // 上次我们设置/读到的滑条位置（识别用户拖动）
    while (debug_running_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      DebugSnap snap;
      {
        std::lock_guard<std::mutex> lk(debug_mutex_);
        snap = debug_snap_;
      }
      if (snap.floor < 0 || snap.floor >= num_floors_) continue;

      // 楼层选择滑条（首帧有效数据后创建；图像下方边框区，不遮地图）
      if (!tb_created && num_floors_ > 1) {
        cv::createTrackbar("floor", win, &tb_val, num_floors_ - 1);
        tb_created = true;
        tb_last = tb_val;
      }
      if (tb_created) {
        const int tb = cv::getTrackbarPos("floor", win);
        if (tb >= 0 && tb != tb_last) {   // 位置变了且不是我们设的 → 用户拖动
          view_floor = tb;
          tb_last = tb;
        } else if (view_floor < 0) {      // 跟随模式：滑条同步机器人楼层
          cv::setTrackbarPos("floor", win, snap.floor);
          tb_last = snap.floor;
        }
      }
      const int view = (view_floor >= 0 && view_floor < num_floors_) ? view_floor : snap.floor;

      const cv::Mat & base = floor_maps_im_[view].BIdmap;
      const cv::Mat & iim = floor_maps_im_[view].BIimap;
      if (base.empty() || iim.empty()) continue;

      // 静态底图（每层一次性，仅窗口线程读写）：BIdmap 上把静态障碍多边形
      // （BIimap 高 2 位非 0，同 clearDynamicObstacles 判据）涂成纯黑 ——
      // 黑色统一表示障碍区域（静态）。
      if (static_cast<int>(debug_static_base_.size()) != num_floors_)
        debug_static_base_.assign(num_floors_, cv::Mat());
      cv::Mat & static_base = debug_static_base_[view];
      if (static_base.empty()) {
        static_base = base.clone();
        for (int y = 0; y < static_base.rows; ++y) {
          auto * drow = static_base.ptr<cv::Vec3b>(y);
          const auto * irow = iim.ptr<uint16_t>(y);
          for (int x = 0; x < static_base.cols; ++x)
            if (((irow[x] >> 14) & 0x3) != 0) drow[x] = cv::Vec3b(0, 0, 0);
        }
      }

      // 地图等比缩放进固定窗口（放大封顶 2x 防小图糊），居中贴到黑底画布上，
      // imshow 尺寸恒定 → 窗口大小不再随地图分辨率变化
      const double scale = std::min(
        {2.0, static_cast<double>(win_sz.width) / base.cols,
         static_cast<double>(win_sz.height) / base.rows});
      cv::Mat resized;
      cv::resize(static_base, resized, cv::Size(), scale, scale, cv::INTER_NEAREST);
      cv::Mat img(win_sz.height, win_sz.width, CV_8UC3, cv::Scalar(0, 0, 0));
      const int ox = (win_sz.width - resized.cols) / 2;
      const int oy = (win_sz.height - resized.rows) / 2;
      resized.copyTo(img(cv::Rect(ox, oy, resized.cols, resized.rows)));
      {
        std::lock_guard<std::mutex> lk(debug_view_mutex_);
        debug_view_state_.valid = true;
        debug_view_state_.floor = view;
        debug_view_state_.scale = scale;
        debug_view_state_.ox = ox;
        debug_view_state_.oy = oy;
        debug_view_state_.draw_cols = resized.cols;
        debug_view_state_.draw_rows = resized.rows;
        debug_view_state_.map_cols = base.cols;
        debug_view_state_.map_rows = base.rows;
      }
      auto to_px = [scale, ox, oy](const cv::Point2f & p) {
        return cv::Point(cvRound(p.x * scale) + ox, cvRound(p.y * scale) + oy);
      };

      // 选中楼层的记忆障碍点（快照携带所有楼层，看哪层画哪层）
      const std::vector<cv::Point2f> no_obs;
      const std::vector<cv::Point2f> & obs_pts =
        (view < static_cast<int>(snap.obs_by_floor.size())) ? snap.obs_by_floor[view] : no_obs;
      // 动态障碍膨胀区（红实心圆盘）＝ markObstacle 写入可通行缓存的禁行圆盘
      // （每点 r=dyn_inflate_px_，与规划看到的禁行区一致）
      const int inflate_r = std::max(2, cvRound(dyn_inflate_px_ * scale));
      for (const auto & p : obs_pts) {
        cv::circle(img, to_px(p), inflate_r, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
      }
      // 障碍本体（黑）：记忆中的动态障碍点（大点）+ 本帧过滤后的点云（小点，
      // 仅当点云属于选中楼层）；静态障碍已在底图涂黑
      for (const auto & p : obs_pts) {
        cv::circle(img, to_px(p), 3, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
      }
      if (snap.cloud_floor == view) {
        for (const auto & p : snap.cloud) {
          cv::circle(img, to_px(p), 2, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
        }
      }
      // 全局路径（蓝，画最上层：压过红盘能看清路径是否穿进禁行区）
      for (size_t i = 0; i + 1 < snap.path.size(); ++i) {
        if (snap.path[i].floor != view || snap.path[i + 1].floor != view) continue;
        cv::line(img,
                 to_px(cv::Point2f(snap.path[i].x, snap.path[i].y)),
                 to_px(cv::Point2f(snap.path[i + 1].x, snap.path[i + 1].y)),
                 cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
      }
      // 机器人（绿 + 朝向箭头；flip_y → 像素 y 取负）—— 只在它所在楼层显示
      if (snap.floor == view) {
        const cv::Point rp = to_px(snap.robot);
        cv::circle(img, rp, 5, cv::Scalar(0, 200, 0), -1, cv::LINE_AA);
        cv::arrowedLine(img, rp,
                        cv::Point(rp.x + cvRound(25.0 * std::cos(snap.yaw) * scale),
                                  rp.y - cvRound(25.0 * std::sin(snap.yaw) * scale)),
                        cv::Scalar(0, 200, 0), 2, cv::LINE_AA, 0, 0.3);
      }
      if (snap.has_goal && snap.goal_floor == view) {
        const cv::Point gp = to_px(snap.goal);
        cv::circle(img, gp, 7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::line(img, cv::Point(gp.x - 9, gp.y), cv::Point(gp.x + 9, gp.y),
                 cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::line(img, cv::Point(gp.x, gp.y - 9), cv::Point(gp.x, gp.y + 9),
                 cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
      }
      // 状态条
      cv::rectangle(img, cv::Point(0, 0), cv::Point(img.cols, 34), cv::Scalar(255, 255, 255), -1);
      std::ostringstream ss;
      ss << "view=F" << view << "/" << num_floors_ << (view_floor < 0 ? "*" : " ")
         << " robot=F" << snap.floor
         << "  obsPts=" << obs_pts.size()
         << "  cloud=" << (snap.cloud_floor == view ? snap.cloud.size() : 0)
         << "  path=" << snap.path.size() << "pts"
         << "  inflateR=" << dyn_inflate_px_ << "px  left-click=goal  [a]=follow";
      cv::putText(img, ss.str(), cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                  cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
      // 状态条右端图例：黑块=障碍本体，红块=膨胀禁行区
      cv::rectangle(img, cv::Point(img.cols - 96, 10), cv::Point(img.cols - 84, 22),
                    cv::Scalar(0, 0, 0), -1);
      cv::rectangle(img, cv::Point(img.cols - 76, 10), cv::Point(img.cols - 64, 22),
                    cv::Scalar(0, 0, 255), -1);

      cv::imshow(win, img);
      // 楼层切换快捷键：0-9 锁定对应楼层，'a' 恢复自动跟随
      const int key = cv::waitKey(1);
      if (key >= '0' && key <= '9') {
        const int f = key - '0';
        if (f < num_floors_) {
          view_floor = f;
          if (tb_created) {
            cv::setTrackbarPos("floor", win, f);
            tb_last = f;
          }
        }
      } else if (key == 'a' || key == 'A') {
        view_floor = -1;
      }
    }
    {
      std::lock_guard<std::mutex> lk(debug_view_mutex_);
      debug_view_state_.valid = false;
    }
    cv::destroyWindow(win);
  }

  // -------------------------------------------------------- replan service
  // local_replan 的重规划请求（safeplanner-2 架构）：起点（当前位姿）+ 原终点
  // + 过滤/降采样后的障碍点云。处理 = updateDynamicObstacles 等价物（障碍点
  // 入图膨胀 + 记忆匹配/刷新）→ planFused（受影响边重算/切边 + 间距降级阶梯）
  // → 成功发布 /global_path 并经响应返回新路径；失败 success=false（局部端
  // 停车保持，障碍过期/挪走后其持续重试会自动恢复）。
  void onReplanPlan(const std::shared_ptr<prm_interfaces::srv::ReplanPlan::Request> req,
                    const std::shared_ptr<prm_interfaces::srv::ReplanPlan::Response> resp)
  {
    if (!enable_dynamic_replan_ || !map_ready_) {
      resp->success = false;
      return;
    }

    // cmu_replan 的持续入图请求（ingest_only）：障碍写入记忆/地图但**不规划
    // 不发布**——先累积，等局部端卡死触发的完整重规划来消费这份记忆。
    // 入图频率 Hz 量级，日志必须节流。
    if (req->ingest_only) {
      ingestObstacles(req->obstacles, req->start, req->decay_sec);
      resp->success = true;             // path 留空：本轮没有新路径
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 10000,
        "[障碍入图] 持续累积中 (不触发重规划, 等待局部端触发)");
      return;
    }

    // 触发事件由局部端（local_replan/cmu_replan）的 [触发重规划] 行播报，
    // 服务端只留 DEBUG 级回执，避免同一事件双份刷屏。
    RCLCPP_DEBUG(get_logger(),
      "replan service called: start F%d (%.2f, %.2f, %.2f), %ux%u obstacle pts",
      floorFromZ(req->start.pose.position.z),
      req->start.pose.position.x, req->start.pose.position.y, req->start.pose.position.z,
      req->obstacles.width, req->obstacles.height);

    // 1. 障碍入图（记忆 + 膨胀标记）
    ingestObstacles(req->obstacles, req->start, req->decay_sec);

    // 2. 起点 = 请求里的当前位姿，终点 = 原任务目标（非内部缓存）
    const auto & sp = req->start.pose.position;
    const auto & gp = req->goal.pose.position;
    start_ = metricToPixel(sp.x, sp.y, sp.z);
    has_start_ = true;
    goal_ = metricToPixel(gp.x, gp.y, gp.z);
    has_goal_ = true;
    publishPoseMarker(start_, "start", makeColor(0.0f, 1.0f, 0.0f, 1.0f), start_marker_pub_);
    publishPoseMarker(goal_, "goal", makeColor(1.0f, 0.0f, 0.0f, 1.0f), goal_marker_pub_);

    // 3. 规划（含间距降级阶梯）并回包
    const nav_msgs::msg::Path path = planIfReady();
    resp->path = path;
    resp->success = !path.poses.empty();
    resp->cost = last_cost_;
  }

  // ----------------------------------------------------------- planning
  // 成功：发布 /global_path 并返回该消息；失败：不发（RViz 保留旧路径）、
  // 返回空消息，由调用方（重规划服务）以 success=false 告知局部端停车保持。
  nav_msgs::msg::Path planIfReady()
  {
    const auto plan_t0 = std::chrono::steady_clock::now();
    if (!map_ready_) {
      return nav_msgs::msg::Path();
    }
    if (start_from_odom_) {
      if (!latest_odom_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "waiting for odometry on %s", odom_topic_.c_str());
        return nav_msgs::msg::Path();
      }
      const auto & pos = latest_odom_->pose.pose.position;
      start_ = metricToPixel(pos.x, pos.y, pos.z);
      has_start_ = true;
      publishPoseMarker(start_, "start", makeColor(0.0f, 1.0f, 0.0f, 1.0f), start_marker_pub_);
    }
    if (!has_start_ || !has_goal_) {
      return nav_msgs::msg::Path();
    }

    MultiFloorTask task;
    task.start = start_;
    task.goal = goal_;

    // Dynamic obstacle points per floor (safeplanner-2 style: point set, no
    // disk fitting): planFused blocks/penalizes same-floor edges by the
    // edge-to-nearest-point distance. forbid = 硬切边距离，safety = 软惩罚间距。
    std::vector<DynamicObstacleCloud> clouds;
    for (size_t f = 0; f < active_obs_pts_.size(); ++f) {
      if (active_obs_pts_[f].empty()) continue;
      DynamicObstacleCloud c;
      c.floor = static_cast<int>(f);
      c.points.reserve(active_obs_pts_[f].size());
      for (const auto & q : active_obs_pts_[f]) {
        c.points.push_back(BIpoint{q.px, q.py, c.floor});
      }
      clouds.push_back(std::move(c));
    }

    // 间距降级阶梯：全间距无路（障碍封死走廊）时逐级放宽重试——
    //   1) 全间距：forbid=入图膨胀, safety=贴墙 clearance（舒适绕行）
    //   2) 半间距：forbid/2, safety/2（窄通道挤过）
    //   3) 物理最小：forbid=max(2px, 机器人物理半径), safety=0（贴障碍刮过，
    //      仍不压障碍点本体）。最后一级仍无路才算真失败。
    // 避免一次切边过狠 -> 空路径 -> 下游 ignore -> 机器人沿旧路径撞障碍。
    const double min_forbid_px = std::max(2.0, obstacle_radius_m_ / map_resolution_);
    const struct { double forbid, safety; } ladder[3] = {
      {dyn_inflate_px_, wall_penalty_clearance_px_},
      {dyn_inflate_px_ * 0.5, wall_penalty_clearance_px_ * 0.5},
      {std::min(dyn_inflate_px_, std::max(min_forbid_px, dyn_inflate_px_ * 0.25)), 0.0},
    };
    double cost = -1.0;
    for (int li = 0; li < 3 && cost <= 0.0; ++li) {
      for (auto & c : clouds) {
        c.forbidPx = ladder[li].forbid;
        c.safetyPx = ladder[li].safety;
      }
      if (li > 0) {
        RCLCPP_WARN(get_logger(),
          "[重规划情况] 第 %d 档间距无路 -> 降级重试 (forbid=%.1fpx, safety=%.1fpx)",
          li, ladder[li].forbid, ladder[li].safety);
      }
      cost = planner_.planFused(task, clouds);
    }

    std::vector<BIpoint> path;
    if (cost > 0.0 && !task.path.empty()) {
      path.assign(task.path.begin(), task.path.end());
    }
    last_cost_ = cost;

    if (path.empty()) {
      // 全部间距均无路（真死路）：不发 /global_path，失败经服务响应
      // success=false 告知局部端停车保持；障碍过期(obstacle_decay_sec)或
      // 挪走后，local_replan 持续的重规划请求会重新拿到路径并自动恢复。
      size_t total_pts = 0;
      for (const auto & c : clouds) total_pts += c.points.size();
      RCLCPP_ERROR(get_logger(),
        "[重规划情况] 失败: 各级间距均无路 (%zu 障碍点/%zu 楼层, 规划耗时 %.1f ms) "
        "-> 已回包 success=false, 局部端停车保持",
        total_pts, clouds.size(),
        std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - plan_t0).count());
      return nav_msgs::msg::Path();
    }
    const double plan_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - plan_t0).count();
    RCLCPP_INFO(get_logger(),
      "[规划情况] 成功: %zu 点, cost=%.1f, 规划耗时 %.1f ms (起点 F%d -> 终点 F%d)",
      path.size(), cost, plan_ms, start_.floor, goal_.floor);
    return publishPath(path);
  }

  // ---------------------------------------------------------- publishing
  nav_msgs::msg::Path publishPath(const std::vector<BIpoint> & path)
  {
    std::vector<double> zs = assignPathZ(path);
    postProcessZ(zs, path);

    // Cache the path (pixel + metric) for the dynamic-obstacle replan check.
    last_path_ = path;
    last_path_metric_.clear();
    last_path_metric_.reserve(path.size());
    for (const auto & p : path) {
      double x, y, z_unused;
      pixelToMetric(p, x, y, z_unused);
      last_path_metric_.push_back({x, y});
    }

    nav_msgs::msg::Path msg;
    msg.header.frame_id = frame_id_;
    msg.header.stamp = now();
    msg.poses.reserve(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
      double x, y, z_unused;
      pixelToMetric(path[i], x, y, z_unused);
      geometry_msgs::msg::PoseStamped pose;
      pose.header = msg.header;
      pose.pose.position = makePoint(x, y, zs[i] + path_height_offset_);
      pose.pose.orientation.w = 1.0;
      msg.poses.push_back(pose);
    }
    path_pub_->publish(msg);
    refreshDebugGoalSnap();

    // Marker
    visualization_msgs::msg::Marker marker;
    marker.header = msg.header;
    marker.ns = "prm_path";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.05;
    marker.color = makeColor(0.0f, 0.5f, 1.0f, 1.0f);
    marker.pose.orientation.w = 1.0;
    for (size_t i = 0; i < path.size(); ++i) {
      double x, y, z_unused;
      pixelToMetric(path[i], x, y, z_unused);
      marker.points.push_back(makePoint(x, y, zs[i] + path_height_offset_));
    }
    path_marker_pub_->publish(marker);
    return msg;
  }

  void publishPoseMarker(
    const BIpoint & p, const std::string & ns,
    const std_msgs::msg::ColorRGBA & color,
    const rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr & pub)
  {
    double x, y, z;
    pixelToMetric(p, x, y, z);

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id_;
    marker.header.stamp = now();
    marker.ns = ns;
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position = makePoint(x, y, z + path_height_offset_);
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.3;
    marker.scale.y = 0.3;
    marker.scale.z = 0.3;
    marker.color = color;
    pub->publish(marker);
  }

  // ------------------------------------------------------------- state
  std::string frame_id_;
  std::string output_path_topic_;
  std::string odom_topic_;
  std::string goal_topic_;
  std::string clicked_point_topic_;
  bool start_from_odom_ = true;
  double path_height_offset_ = 0.0;

  int num_floors_ = 0;
  std::vector<std::string> floor_maps_;
  std::vector<double> floor_z_mins_;
  std::vector<double> floor_z_maxs_;
  std::vector<double> floor_ground_zs_;   // raw config (per-floor overrides)
  std::vector<double> floor_ground_z_;    // resolved: override or estimate
  std::vector<std::string> floor_elevation_npys_;
  std::string connections_json_;

  double map_resolution_ = 0.05;
  double map_origin_x_ = 0.0;
  double map_origin_y_ = 0.0;
  bool flip_y_ = true;

  double expansion_radius_ = 4.0;
  int prm_k_nodes_ = 500;
  double prm_r_nei_ = 25.0;
  int fuse_k_neighbors_ = 10;
  double fuse_cross_radius_ = 8.0;

  int z_median_filter_window_ = 7;
  double z_max_slope_ = 1.5;

  int map_width_ = 0;
  int map_height_ = 0;
  bool map_ready_ = false;

  std::vector<std::pair<double, double>> z_ranges_;
  std::vector<BImap> floor_maps_im_;

  // path Z sources: per-floor elevation grids (npy) vs direct PCD queries
  std::string z_source_;
  std::vector<ElevationGrid> floor_grids_;
  std::vector<bool> floor_use_npy_;

  PcdZIndex pcd_index_;
  std::string pcd_map_path_;
  double pcd_z_quantile_ = 0.25;
  double pcd_z_cluster_tol_ = 0.15;
  double pcd_ground_tol_below_ = 0.6;   // anchor window: ground - tol
  double pcd_ground_tol_above_ = 0.9;   // anchor window: ground + tol
  double pcd_search_radius_ = 0.2;
  PRMMultiFloor planner_;

  bool has_start_ = false;
  bool has_goal_ = false;
  BIpoint start_;
  BIpoint goal_;
  nav_msgs::msg::Odometry::SharedPtr latest_odom_;

  // ---- dynamic-obstacle replanning state ----
  // 点级障碍（safeplanner-2 风格）：不做圆盘拟合，每个过滤后的点独立维护
  // （匹配刷新时间戳 / 过期），入图与 planFused 都以「点 + forbid 半径」处理。
  struct ActiveObsPoint
  {
    double px = 0.0, py = 0.0;   // 像素坐标
    rclcpp::Time stamp;
    double expire_sec = 0.0;     // 记忆时长（入图请求可覆盖：cmu 短记忆/local 默认）
  };
  bool enable_dynamic_replan_ = false;
  std::string replan_service_;           // 重规划服务名（local_replan 调用）
  double replan_check_rate_ = 1.0;       // 障碍过期检查频率 (Hz)
  double obstacle_cluster_tol_ = 0.25;   // 点匹配距离：新点与老点近于该值 → 刷新记忆
  double obstacle_radius_m_ = 0.30;
  double obstacle_decay_sec_ = 2.0;
  double obstacle_elev_tol_ = 0.2;       // 入图高程校验容差 (m)；<0 关闭（见参数注释）
  double dynamic_inflation_px_ = -1.0;   // <0 = 跟随 expansion_radius
  bool replan_debug_window_ = true;
  double last_cost_ = -1.0;              // 最近一次 planIfReady 的路径代价
  double wall_penalty_gain_ = 0.0;
  double wall_penalty_clearance_px_ = 10.0;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Service<prm_interfaces::srv::ReplanPlan>::SharedPtr replan_srv_;
  rclcpp::TimerBase::SharedPtr replan_timer_;   // 障碍过期 + 调试快照刷新
  std::vector<std::vector<ActiveObsPoint>> active_obs_pts_;   // [floor] 存活中的动态障碍点
  std::vector<BIpoint> last_path_;
  std::vector<std::array<double, 2>> last_path_metric_;

  // ---- replan debug window state（safeplanner-2 风格，随 launch 启动）----
  double dyn_inflate_px_ = 0.0;          // 动态障碍点膨胀半径 (px)，buildMaps 里解析
  std::thread debug_thread_;
  std::atomic<bool> debug_running_{false};
  std::mutex debug_mutex_;               // 只保护 debug_snap_（写入=updateObstacles，读取=窗口线程）
  std::vector<cv::Mat> debug_static_base_;  // 每层静态底图（BIdmap + 静态障碍涂黑），debugLoop 首帧生成，仅窗口线程访问
  struct DebugViewState
  {
    bool valid = false;
    int floor = -1;
    double scale = 1.0;
    int ox = 0;
    int oy = 0;
    int draw_cols = 0;
    int draw_rows = 0;
    int map_cols = 0;
    int map_rows = 0;
  };
  std::mutex debug_view_mutex_;
  DebugViewState debug_view_state_;
  std::mutex pending_debug_goal_mutex_;
  bool pending_debug_goal_ = false;
  BIpoint pending_debug_goal_px_;
  double pending_debug_goal_metric_x_ = 0.0;
  double pending_debug_goal_metric_y_ = 0.0;
  struct DebugSnap
  {
    int floor = -1;       // 机器人当前楼层（自动跟随模式显示它）
    cv::Point2f robot{0.0f, 0.0f};
    float yaw = 0.0f;
    int cloud_floor = -1;               // 最近一批点云所属楼层（查看别的层时不画）
    std::vector<cv::Point2f> cloud;     // 最近一批过滤后的障碍点云（像素，画小黑点）
    std::vector<std::vector<cv::Point2f>> obs_by_floor;  // 每层记忆障碍点（画大黑点+红膨胀盘）
    std::vector<BIpoint> path;          // 最近一次全局路径（像素，.floor 有效）
    bool has_goal = false;
    int goal_floor = -1;
    cv::Point2f goal{0.0f, 0.0f};       // debug 窗口点击/goal_pose 的终点（像素）
  } debug_snap_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr start_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_marker_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr clicked_point_sub_;
  rclcpp::TimerBase::SharedPtr debug_goal_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PrmPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
