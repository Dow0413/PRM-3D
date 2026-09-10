#include "PRMmultifloor.h"
#include <limits>
#include <queue>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <fstream>
#include <cmath>

// Forward declarations
static inline int32_t getPolygonIndexFromMap(BImap* map, BIpoint pt, int32_t floorIndex);

PRMMultiFloor::PRMMultiFloor() : _numFloors(0), _prmKNodes(500), _prmRNei(50.0)
{
}

PRMMultiFloor::~PRMMultiFloor()
{
}

void PRMMultiFloor::initialize(int numFloors)
{
    _numFloors = numFloors;
    _multiFloorGraph.floorGraphs.resize(numFloors);
    _mapReferences.resize(numFloors, nullptr);
    _prmGraphs.resize(numFloors);
    _prmKNodesOverride.resize(numFloors, -1);
    _prmRNeiOverride.resize(numFloors, -1.0);
    _traversableCache.resize(numFloors);
    _cacheWidth.resize(numFloors, 0);
    _cacheHeight.resize(numFloors, 0);
    _roadmapBuilt.resize(numFloors, false);
    _roadmapGrid.resize(numFloors);
    _roadmapGridW.resize(numFloors, 0);
    _roadmapGridH.resize(numFloors, 0);
}

void PRMMultiFloor::loadFloorGraph(int32_t floorIndex, const BIgraph& graph)
{
    if (floorIndex < 0 || floorIndex >= _numFloors)
    {
        printf("error: invalid floor index!!\r\n");
        return;
    }

    // 设置多边形的 floor 字段
    BIgraph graphCopy = graph;
    for (auto& poly : graphCopy.freepolygonList)
    {
        poly.floor = floorIndex;
    }

    _multiFloorGraph.floorGraphs[floorIndex] = graphCopy;
}

void PRMMultiFloor::setMapReference(int32_t floorIndex, BImap* map)
{
    if (floorIndex < 0 || floorIndex >= _numFloors)
    {
        printf("error: invalid floor index!!\r\n");
        return;
    }
    _mapReferences[floorIndex] = map;
}

void PRMMultiFloor::addFloorConnection(BIpoint pointFrom, BIpoint pointTo,
                                       double cost,
                                       FloorConnectionType type)
{
    int32_t floorFrom = pointFrom.floor;
    int32_t floorTo = pointTo.floor;

    if (floorFrom < 0 || floorFrom >= _numFloors ||
        floorTo < 0 || floorTo >= _numFloors)
    {
        printf("error: invalid floor in connection points!!\r\n");
        return;
    }

    // 检查连接点是否在地图范围内
    if (_mapReferences[floorFrom] != nullptr)
    {
        BImap* m = _mapReferences[floorFrom];
        if (!m->BIamap.empty() && (pointFrom.x < 0 || pointFrom.x >= m->shapeX || pointFrom.y < 0 || pointFrom.y >= m->shapeY))
        {
            printf("error: pointFrom(%.1f,%.1f) out of floor %d map bounds!!\r\n",
                   pointFrom.x, pointFrom.y, floorFrom);
            return;
        }
    }
    if (_mapReferences[floorTo] != nullptr)
    {
        BImap* m = _mapReferences[floorTo];
        if (!m->BIamap.empty() && (pointTo.x < 0 || pointTo.x >= m->shapeX || pointTo.y < 0 || pointTo.y >= m->shapeY))
        {
            printf("error: pointTo(%.1f,%.1f) out of floor %d map bounds!!\r\n",
                   pointTo.x, pointTo.y, floorTo);
            return;
        }
    }

    FloorConnection conn;
    conn.id = _multiFloorGraph.connections.size();
    conn.type = type;
    conn.pointFrom = pointFrom;
    conn.pointTo = pointTo;

    // 从 BIimap 读取真实多边形索引（findPolygonIndex 可能不准）
    int32_t fromPoly = -1, toPoly = -1;
    if (_mapReferences[floorFrom] != nullptr)
        fromPoly = getPolygonIndexFromMap(_mapReferences[floorFrom], pointFrom, floorFrom);
    if (fromPoly < 0) fromPoly = findPolygonIndex(floorFrom, pointFrom);
    conn.polyIndexFrom = fromPoly;

    if (_mapReferences[floorTo] != nullptr)
        toPoly = getPolygonIndexFromMap(_mapReferences[floorTo], pointTo, floorTo);
    if (toPoly < 0) toPoly = findPolygonIndex(floorTo, pointTo);
    conn.polyIndexTo = toPoly;
    conn.cost = cost;

    printf("addFloorConnection: F%d(poly%d) -> F%d(poly%d), cost=%.2f\r\n",
           floorFrom, conn.polyIndexFrom, floorTo, conn.polyIndexTo, conn.cost);

    _multiFloorGraph.connections.push_back(conn);
}

void PRMMultiFloor::printGraphInfo() const
{
    printf("=== Multi-Floor Graph Info ===\r\n");
    printf("Number of floors: %d\r\n", _numFloors);
    printf("Number of connections: %zu\r\n", _multiFloorGraph.connections.size());

    for (int32_t i = 0; i < _numFloors; i++)
    {
        const BIgraph& graph = _multiFloorGraph.floorGraphs[i];
        printf("Floor %d: cutlines=%zu, freepolygons=%zu\r\n",
               i, graph.cutlineList.size(), graph.freepolygonList.size());
    }

    printf("\nConnections:\r\n");
    for (const auto& conn : _multiFloorGraph.connections)
    {
        printf("  ID=%d: Floor%d(Poly%d) <-> Floor%d(Poly%d), cost=%.2f\r\n",
               conn.id, conn.pointFrom.floor, conn.polyIndexFrom,
               conn.pointTo.floor, conn.polyIndexTo, conn.cost);
    }
}

int32_t PRMMultiFloor::findPolygonIndex(int32_t floorIndex, BIpoint point)
{
    if (floorIndex < 0 || floorIndex >= _numFloors)
        return -1;

    BIgraph& graph = _multiFloorGraph.floorGraphs[floorIndex];

    double minDist = doubleMax;
    int32_t bestPoly = -1;

    for (size_t i = 0; i < graph.freepolygonList.size(); i++)
    {
        double dist = point % graph.freepolygonList[i].core;
        if (dist < minDist)
        {
            minDist = dist;
            bestPoly = (int32_t)i;
        }
    }

    return bestPoly;
}

// GetLeastHomotopyPath standalone implementation
static void ComputeLeastHomotopyPath(const BIgraph& graph,
                                     const std::vector<BIline>& f_path,
                                     std::list<BIpoint>& Path,
                                     double& PathMinCost)
{
    int32_t fsize = f_path.size();
    std::vector<Node> nodeList;
    nodeList.reserve(fsize * 2);
    nodeList.emplace_back(Node{0, f_path[0].S, -1, 0, 0});
    nodeList.emplace_back(Node{0, f_path[0].E, -1, 0, 0});

    for (int32_t line_i = 1; line_i < fsize; line_i++)
    {
        int32_t now_0 = nodeList.size();
        nodeList.emplace_back(Node{doubleMax, f_path[line_i].S, -1, line_i, now_0});
        nodeList.emplace_back(Node{doubleMax, f_path[line_i].E, -1, line_i, now_0 + 1});

        for (int32_t nowBias = 0; nowBias < 2; nowBias++)
        {
            Node& x_now = nodeList[now_0 + nowBias];
            for (int32_t p0 = now_0 - 2; p0 < now_0; p0++)
            {
                Node* x_p = &nodeList[p0];
                bool should_break = false;

                while (x_p->index != 0)
                {
                    int32_t p_temp = x_p->par;
                    int32_t line_j = nodeList[p_temp].bridge + 1;
                    BIline l_ptemp2now = {nodeList[p_temp].point, x_now.point};

                    for (int32_t line_k = line_j; line_k < line_i; line_k++)
                    {
                        if (!doIntersect_rigorous(l_ptemp2now, f_path[line_k]))
                        {
                            should_break = true;
                            break;
                        }
                    }
                    if (should_break)
                        break;
                    x_p = &nodeList[p_temp];
                }

                double cost_now = x_p->cost + (x_p->point % x_now.point);
                if (x_now.cost > cost_now)
                {
                    x_now.par = x_p->index;
                    x_now.cost = cost_now;
                }
            }
        }
    }

    Path.clear();
    int32_t x = nodeList.back().index;
    while (x != -1)
    {
        Path.push_front(nodeList[x].point);
        x = nodeList[x].par;
    }
    PathMinCost = nodeList.back().cost;
}

static inline int32_t getPolygonIndexFromMap(BImap* map, BIpoint pt, int32_t floorIndex)
{
    if (map == nullptr || map->BIimap.empty()) return -1;
    if (pt.x < 0 || pt.x >= map->shapeX || pt.y < 0 || pt.y >= map->shapeY) return -1;
    uint16_t val = map->BIimap.at<uint16_t>((int)pt.y, (int)pt.x);
    if ((val & 0xC000) != 0) return -1;  // 障碍物
    return val & 0x3FFF;
}

// ========== KDTree2D Implementation ==========

void KDTree2D::build(const std::vector<BIpoint>& points, const std::vector<int>& indices)
{
    _nodes.clear();
    if (points.empty()) return;

    std::vector<std::pair<BIpoint, int>> pts;
    pts.reserve(points.size());
    for (size_t i = 0; i < points.size(); i++)
        pts.emplace_back(points[i], indices[i]);

    buildRecursive(pts, 0, (int)pts.size(), 0);
}

int KDTree2D::buildRecursive(std::vector<std::pair<BIpoint, int>>& pts, int begin, int end, int depth)
{
    if (begin >= end) return -1;
    if (begin + 1 == end)
    {
        KDNode node;
        node.pt = pts[begin].first;
        node.prmNodeIdx = pts[begin].second;
        node.splitDim = depth % 2;
        node.left = -1;
        node.right = -1;
        int idx = (int)_nodes.size();
        _nodes.push_back(node);
        return idx;
    }

    int dim = depth % 2;
    auto cmp = [dim](const std::pair<BIpoint, int>& a, const std::pair<BIpoint, int>& b) {
        return (dim == 0) ? (a.first.x < b.first.x) : (a.first.y < b.first.y);
    };
    std::sort(pts.begin() + begin, pts.begin() + end, cmp);

    int mid = (begin + end) / 2;

    KDNode node;
    node.pt = pts[mid].first;
    node.prmNodeIdx = pts[mid].second;
    node.splitDim = dim;

    int idx = (int)_nodes.size();
    _nodes.push_back(node);

    _nodes[idx].left  = buildRecursive(pts, begin, mid, depth + 1);
    _nodes[idx].right = buildRecursive(pts, mid + 1, end, depth + 1);

    return idx;
}

void KDTree2D::radiusSearch(const BIpoint& center, double radius, std::vector<int>& result) const
{
    result.clear();
    if (_nodes.empty()) return;
    radiusSearchRecursive(0, center, radius, result);
}

void KDTree2D::radiusSearchRecursive(int nodeIdx, const BIpoint& center, double radius, std::vector<int>& result) const
{
    if (nodeIdx < 0 || nodeIdx >= (int)_nodes.size()) return;

    const KDNode& node = _nodes[nodeIdx];

    // Check if this node is within radius
    double dist = node.pt % center;
    if (dist <= radius)
        result.push_back(node.prmNodeIdx);

    // Determine child visit order based on split axis
    int dim = node.splitDim;
    double centerVal = (dim == 0) ? center.x : center.y;
    double nodeVal   = (dim == 0) ? node.pt.x : node.pt.y;
    double diff = centerVal - nodeVal;

    if (diff <= 0)
    {
        // Center is on the left/lower side; search that subtree first
        radiusSearchRecursive(node.left, center, radius, result);
        // Only check far subtree if splitting plane is within radius
        if (fabs(diff) <= radius)
            radiusSearchRecursive(node.right, center, radius, result);
    }
    else
    {
        radiusSearchRecursive(node.right, center, radius, result);
        if (fabs(diff) <= radius)
            radiusSearchRecursive(node.left, center, radius, result);
    }
}

void KDTree2D::collectInRadius(int nodeIdx, const BIpoint& center, double radius, std::vector<int>& nodeIdxs) const
{
    if (nodeIdx < 0 || nodeIdx >= (int)_nodes.size()) return;
    const KDNode& node = _nodes[nodeIdx];

    if ((node.pt % center) <= radius)
        nodeIdxs.push_back(nodeIdx);

    int dim = node.splitDim;
    double centerVal = (dim == 0) ? center.x : center.y;
    double nodeVal   = (dim == 0) ? node.pt.x : node.pt.y;
    double diff = centerVal - nodeVal;

    if (diff <= 0)
    {
        collectInRadius(node.left, center, radius, nodeIdxs);
        if (fabs(diff) <= radius)
            collectInRadius(node.right, center, radius, nodeIdxs);
    }
    else
    {
        collectInRadius(node.right, center, radius, nodeIdxs);
        if (fabs(diff) <= radius)
            collectInRadius(node.left, center, radius, nodeIdxs);
    }
}

void KDTree2D::kNearest(const BIpoint& center, int k, std::vector<int>& result) const
{
    result.clear();
    if (_nodes.empty() || k <= 0) return;

    // 半径指数增长，直到收集到 >= k 个候选（保证取到的确实是全局最近 k 个）
    std::vector<int> nodeIdxs;
    double r = 8.0;
    for (int it = 0; it < 32 && (int)nodeIdxs.size() < k; ++it)
    {
        collectInRadius(0, center, r, nodeIdxs);
        if ((int)nodeIdxs.size() >= k) break;
        r *= 2.0;
    }

    // 按距离升序排序，取前 k 个 prmNodeIdx（对融合图而言 prmNodeIdx = 全局节点 id）
    std::vector<std::pair<double, int>> dists;
    dists.reserve(nodeIdxs.size());
    for (int ni : nodeIdxs)
        dists.emplace_back(_nodes[ni].pt % center, _nodes[ni].prmNodeIdx);
    std::sort(dists.begin(), dists.end());
    int n = std::min(k, (int)dists.size());
    result.reserve((size_t)n);
    for (int i = 0; i < n; ++i)
        result.push_back(dists[i].second);
}

// ========== PRM Collision Checking Helpers ==========

bool PRMMultiFloor::isPointTraversable(int floorIdx, double x, double y) const
{
    if (floorIdx < 0 || floorIdx >= _numFloors) return false;
    int w = _cacheWidth[floorIdx], h = _cacheHeight[floorIdx];
    if (w == 0) return true;  // no cache built yet, assume traversable
    int ix = (int)x, iy = (int)y;
    if (ix < 0 || ix >= w || iy < 0 || iy >= h) return false;
    return _traversableCache[floorIdx][iy * w + ix] != 0;
}

bool PRMMultiFloor::isPointTraversableStatic(int floorIdx, double x, double y) const
{
    if (floorIdx < 0 || floorIdx >= _numFloors) return false;
    BImap* map = _mapReferences[floorIdx];
    if (map == nullptr || map->BIimap.empty()) return true;  // no map: assume free
    int w = (int)map->shapeX, h = (int)map->shapeY;
    int ix = (int)x, iy = (int)y;
    if (ix < 0 || ix >= w || iy < 0 || iy >= h) return false;
    const uint16_t* row = map->BIimap.ptr<uint16_t>(iy);
    return ((row[ix] >> 14) & 0x3) == 0;   // same rule as clearDynamicObstacles
}

bool PRMMultiFloor::isSegmentFree(int floorIdx, const BIpoint& a, const BIpoint& b) const
{
    if (floorIdx < 0 || floorIdx >= _numFloors) return false;
    int w = _cacheWidth[floorIdx], h = _cacheHeight[floorIdx];
    if (w == 0) return true;

    double dist = a % b;
    if (dist < 1e-6) return isPointTraversable(floorIdx, a.x, a.y);

    double dx = b.x - a.x, dy = b.y - a.y;
    int numSamples = std::max(2, (int)(dist / 5.0) + 1);  // 5-pixel step
    double invN = 1.0 / (double)numSamples;
    const uint8_t* cachePtr = _traversableCache[floorIdx].data();
    for (int i = 0; i <= numSamples; i++)
    {
        double t = (double)i * invN;
        int ix = (int)(a.x + t * dx);
        int iy = (int)(a.y + t * dy);
        if (ix < 0 || ix >= w || iy < 0 || iy >= h) return false;
        if (cachePtr[iy * w + ix] == 0) return false;
    }
    return true;
}

// ========== PRM Parameter Configuration ==========

void PRMMultiFloor::setPRMParams(int k_nodes, double r_nei)
{
    _prmKNodes = k_nodes;
    _prmRNei = r_nei;
}

void PRMMultiFloor::setFloorPRMParams(int floorIdx, int k_nodes, double r_nei)
{
    if (floorIdx < 0 || floorIdx >= _numFloors)
    {
        printf("setFloorPRMParams: invalid floor index %d\r\n", floorIdx);
        return;
    }
    _prmKNodesOverride[floorIdx] = k_nodes;
    _prmRNeiOverride[floorIdx] = r_nei;
}

// ========== PRM Building ==========

void PRMMultiFloor::buildPRMForFloor(int floorIdx, BIpoint start, BIpoint goal)
{
    if (floorIdx < 0 || floorIdx >= _numFloors)
    {
        printf("buildPRMForFloor: invalid floor index %d\r\n", floorIdx);
        return;
    }

    BImap* map = _mapReferences[floorIdx];
    if (map == nullptr || map->BIimap.empty())
    {
        printf("buildPRMForFloor: no map reference for floor %d\r\n", floorIdx);
        return;
    }

    int64_t t_start = utime_ns(), t0, t1;

    // 获取该楼层的有效参数：优先用覆盖值，否则用全局默认值
    int effKNodes  = (_prmKNodesOverride[floorIdx] >= 0)  ? _prmKNodesOverride[floorIdx]  : _prmKNodes;
    double effRNei = (_prmRNeiOverride[floorIdx]  >= 0.0) ? _prmRNeiOverride[floorIdx] : _prmRNei;

    PRMGraph& prm = _prmGraphs[floorIdx];
    prm.nodes.clear();
    prm.connIdToNodeIdx.clear();
    prm.k_nodes = effKNodes;
    prm.r_nei = effRNei;

    printf("Floor %d: k_nodes=%d, r_nei=%.1f%s\r\n",
           floorIdx, effKNodes, effRNei,
           (_prmKNodesOverride[floorIdx] >= 0) ? " (per-floor override)" : " (global default)");

    // Step 0: 构建可通行性缓存 (uint8_t 扁平数组, raw pointer 访问)
    int w = (int)map->shapeX;
    int h = (int)map->shapeY;
    std::vector<uint8_t>& cache = _traversableCache[floorIdx];
    cache.resize(w * h);
    const uint8_t* cachePtr = cache.data();  // raw pointer for fast access

    for (int y = 0; y < h; y++)
    {
        const uint16_t* row = map->BIimap.ptr<uint16_t>(y);
        uint8_t* dst = cache.data() + y * w;
        for (int x = 0; x < w; x++)
            dst[x] = ((row[x] >> 14) & 0x3) == 0 ? 1 : 0;
    }
    _cacheWidth[floorIdx] = w;
    _cacheHeight[floorIdx] = h;

    // 贴墙惩罚：每个自由像素到最近障碍的欧氏距离(px)。自由=255 障碍=0，
    // distanceTransform 输出 CV_32F，自由像素得到到最近 0 的距离。
    if ((int)_clearanceMaps.size() != _numFloors) _clearanceMaps.resize(_numFloors);
    {
        cv::Mat mask(h, w, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < h; ++y)
        {
            uint8_t* mr = mask.ptr<uint8_t>(y);
            const uint8_t* cr = cache.data() + y * w;
            for (int x = 0; x < w; ++x) mr[x] = cr[x] ? 255 : 0;
        }
        cv::Mat dist;
        cv::distanceTransform(mask, dist, cv::DIST_L2, 3);
        std::vector<float>& cm = _clearanceMaps[floorIdx];
        cm.resize((size_t)w * h);
        for (int y = 0; y < h; ++y)
        {
            const float* dr = dist.ptr<float>(y);
            std::copy(dr, dr + w, cm.begin() + y * w);
        }
    }

    // Inline point-check lambda (raw pointer, no bounds-check overhead)
    auto pointFree = [&](double x, double y) -> bool {
        int ix = (int)x, iy = (int)y;
        if (ix < 0 || ix >= w || iy < 0 || iy >= h) return false;
        return cachePtr[iy * w + ix] != 0;
    };

    // Inline segment-check lambda (raw pointer, samples at step=max(1, r_nei/5))
    int sampleStep = std::max(1, (int)(effRNei / 5.0));
    auto segmentFree = [&](const BIpoint& a, const BIpoint& b) -> bool {
        double dist = a % b;
        if (dist < 1e-6) return pointFree(a.x, a.y);
        int numSamples = std::max(2, (int)(dist / sampleStep) + 1);
        double dx = b.x - a.x, dy = b.y - a.y;
        double invN = 1.0 / (double)numSamples;
        for (int i = 0; i <= numSamples; i++)
        {
            double t = (double)i * invN;
            if (!pointFree(a.x + t * dx, a.y + t * dy))
                return false;
        }
        return true;
    };

    // Step 1: 先收集所有自由像素，再从中均匀采样（无放回）。
    // 自由空间很稀疏时，对全图 rejection sampling 几乎所有尝试都落空，这里直接采样自由像素。
    t0 = utime_ns();
    std::vector<std::pair<int, int>> freePx;
    freePx.reserve(1 << 16);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (cachePtr[y * w + x]) freePx.emplace_back(x, y);

    if (freePx.empty())
    {
        printf("buildPRMForFloor: no traversable points found on floor %d!\r\n", floorIdx);
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(freePx.begin(), freePx.end(), gen);
    int target = std::min((int)freePx.size(), effKNodes);
    prm.nodes.reserve(target);
    for (int i = 0; i < target; ++i)
    {
        PRMNode node;
        node.point = {(double)freePx[i].first, (double)freePx[i].second, floorIdx};
        prm.nodes.push_back(node);
    }
    t1 = utime_ns();
    printf("Floor %d: sampling %zu nodes in %.1f ms (free px=%zu)\r\n",
           floorIdx, prm.nodes.size(), (t1 - t0) / 1e6, freePx.size());

    // Step 2: Add start, goal, and connection points
    auto addPoint = [&](BIpoint pt, int connId) -> int {
        if (!pointFree(pt.x, pt.y))
        {
            printf("Warning: point (%.1f,%.1f) on floor %d is on obstacle, skipping\r\n",
                   pt.x, pt.y, floorIdx);
            return -1;
        }
        int idx = (int)prm.nodes.size();
        PRMNode node;
        node.point = pt;
        node.point.floor = floorIdx;
        prm.nodes.push_back(node);
        return idx;
    };

    int startIdx = -1, goalIdx = -1;
    if (start.x >= 0 && start.floor == floorIdx)
    { startIdx = addPoint(start, -1); printf("Floor %d: start node = %d\r\n", floorIdx, startIdx); }
    if (goal.x >= 0 && goal.floor == floorIdx)
    { goalIdx = addPoint(goal, -1); printf("Floor %d: goal node = %d\r\n", floorIdx, goalIdx); }

    for (const auto& conn : _multiFloorGraph.connections)
    {
        if (conn.pointFrom.floor == floorIdx)
        {
            int idx = addPoint(conn.pointFrom, conn.id);
            if (idx >= 0) { prm.connIdToNodeIdx[conn.id] = idx;
                printf("Floor %d: conn %d (from) -> node %d\r\n", floorIdx, conn.id, idx); }
        }
        if (conn.pointTo.floor == floorIdx)
        {
            int idx = addPoint(conn.pointTo, conn.id);
            if (idx >= 0) { prm.connIdToNodeIdx[conn.id] = idx;
                printf("Floor %d: conn %d (to) -> node %d\r\n", floorIdx, conn.id, idx); }
        }
    }

    // Step 3: 均匀网格空间索引（替代 KD-tree，O(1) 查询每个节点）
    t0 = utime_ns();
    int gridW = std::max(1, (int)(w / effRNei));
    int gridH = std::max(1, (int)(h / effRNei));
    std::vector<std::vector<int>> grid(gridW * gridH);
    for (size_t i = 0; i < prm.nodes.size(); i++)
    {
        int gx = std::min(gridW - 1, std::max(0, (int)(prm.nodes[i].point.x / effRNei)));
        int gy = std::min(gridH - 1, std::max(0, (int)(prm.nodes[i].point.y / effRNei)));
        grid[gy * gridW + gx].push_back((int)i);
    }
    t1 = utime_ns();
    printf("Floor %d: grid [%dx%d] built in %.1f ms\r\n", floorIdx, gridW, gridH, (t1-t0)/1e6);

    // Step 4: 半径建边（3×3 网格邻域 + r_nei 距离 + segmentFree）。
    // galileo 自由区是「面积紧凑但高度非凸（迷宫式）」——短直线(r_nei 内)能穿过通道把图连起来，
    // 而 k-最近邻会跨过内部墙、被 segmentFree 拒掉导致碎片化，故这里用半径建边。
    // 密度由采样节点数 k 控制（见 Step 1，自由像素较多时只取子集），避免边爆炸。
    t0 = utime_ns();
    int totalEdges = 0;
    for (size_t i = 0; i < prm.nodes.size(); i++)
    {
        int gx = (int)(prm.nodes[i].point.x / effRNei);
        int gy = (int)(prm.nodes[i].point.y / effRNei);
        for (int dy = -1; dy <= 1; dy++)
        {
            int ny = gy + dy;
            if (ny < 0 || ny >= gridH) continue;
            for (int dx = -1; dx <= 1; dx++)
            {
                int nx = gx + dx;
                if (nx < 0 || nx >= gridW) continue;
                for (int j : grid[(size_t)ny * gridW + nx])
                {
                    if (j <= (int)i) continue;
                    double dx_ij = prm.nodes[i].point.x - prm.nodes[j].point.x;
                    double dy_ij = prm.nodes[i].point.y - prm.nodes[j].point.y;
                    if (dx_ij * dx_ij + dy_ij * dy_ij > effRNei * effRNei) continue;
                    if (segmentFree(prm.nodes[i].point, prm.nodes[j].point))
                    {
                        prm.nodes[i].neighbors.push_back(j);
                        prm.nodes[j].neighbors.push_back(i);
                        totalEdges++;
                    }
                }
            }
        }
    }
    t1 = utime_ns();
    printf("Floor %d: %d edges built in %.1f ms\r\n", floorIdx, totalEdges, (t1 - t0) / 1e6);

    printf("Floor %d: total build = %.1f ms\r\n", floorIdx, (t1 - t_start) / 1e6);
}

// ========== 一次建图 + 多次重规划（动态避障仿真用） ==========

// 基于已构建的 _traversableCache 做线段碰撞检测（buildRoadmap / replan 共用同一判定）
bool PRMMultiFloor::segmentFreeCached(int floorIdx, const BIpoint& a, const BIpoint& b) const
{
    if (floorIdx < 0 || floorIdx >= _numFloors) return false;
    int w = _cacheWidth[floorIdx], h = _cacheHeight[floorIdx];
    if (w == 0) return true;  // 尚未建缓存，按可通行处理

    double dist = a % b;
    if (dist < 1e-6)
    {
        int ix = (int)a.x, iy = (int)a.y;
        if (ix < 0 || ix >= w || iy < 0 || iy >= h) return false;
        return _traversableCache[floorIdx][iy * w + ix] != 0;
    }

    int sampleStep = std::max(1, (int)(_prmGraphs[floorIdx].r_nei / 5.0));
    int numSamples = std::max(2, (int)(dist / sampleStep) + 1);
    double dx = b.x - a.x, dy = b.y - a.y;
    double invN = 1.0 / (double)numSamples;
    const uint8_t* cachePtr = _traversableCache[floorIdx].data();
    for (int i = 0; i <= numSamples; i++)
    {
        double t = (double)i * invN;
        int ix = (int)(a.x + t * dx);
        int iy = (int)(a.y + t * dy);
        if (ix < 0 || ix >= w || iy < 0 || iy >= h) return false;
        if (cachePtr[iy * w + ix] == 0) return false;
    }
    return true;
}

// ========== 动态障碍：写入/清除可通行栅格（在线重规划用） ==========

bool PRMMultiFloor::markObstacle(int floorIdx, double px, double py, double radiusPx)
{
    if (floorIdx < 0 || floorIdx >= _numFloors)
    {
        printf("markObstacle: 非法楼层 %d\r\n", floorIdx);
        return false;
    }
    int w = _cacheWidth[floorIdx], h = _cacheHeight[floorIdx];
    if (w <= 0 || h <= 0)
    {
        printf("markObstacle: floor %d 尚未建缓存\r\n", floorIdx);
        return false;
    }
    int cx = (int)px, cy = (int)py, r = std::max(1, (int)std::ceil(radiusPx));
    int x0 = std::max(0, cx - r), x1 = std::min(w - 1, cx + r);
    int y0 = std::max(0, cy - r), y1 = std::min(h - 1, cy + r);
    if (x0 > x1 || y0 > y1) return false;
    uint8_t* cache = _traversableCache[floorIdx].data();
    double r2 = radiusPx * radiusPx;
    int stamped = 0;
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x)
        {
            double dx = x - px, dy = y - py;
            if (dx * dx + dy * dy <= r2 && cache[y * w + x])
            {
                cache[y * w + x] = 0;
                ++stamped;
            }
        }
    printf("markObstacle: F%d 圆盘(%.1f,%.1f r=%.1fpx) 清除 %d 个自由像素\r\n",
           floorIdx, px, py, radiusPx, stamped);
    return stamped > 0;
}

bool PRMMultiFloor::clearDynamicObstacles()
{
    for (int f = 0; f < _numFloors; ++f)
    {
        BImap* map = _mapReferences[f];
        if (map == nullptr || map->BIimap.empty()) continue;
        int w = (int)map->shapeX, h = (int)map->shapeY;
        if ((int)_traversableCache[f].size() != w * h) continue;
        for (int y = 0; y < h; ++y)
        {
            const uint16_t* row = map->BIimap.ptr<uint16_t>(y);
            uint8_t* dst = _traversableCache[f].data() + y * w;
            for (int x = 0; x < w; ++x)
                dst[x] = ((row[x] >> 14) & 0x3) == 0 ? 1 : 0;
        }
    }
    printf("clearDynamicObstacles: 已按静态地图重建全部可通行缓存\r\n");
    return true;
}

// ========== 贴墙惩罚 ==========

void PRMMultiFloor::setWallPenalty(double gain, double clearancePx)
{
    _wallPenaltyGain = std::max(0.0, gain);
    _wallPenaltyClearancePx = std::max(1.0, clearancePx);
    if (_wallPenaltyGain > 0.0)
        printf("setWallPenalty: gain=%.2f, clearancePx=%.1f\r\n",
               _wallPenaltyGain, _wallPenaltyClearancePx);
}

double PRMMultiFloor::clearanceAt(int floorIdx, double px, double py) const
{
    if (floorIdx < 0 || floorIdx >= (int)_clearanceMaps.size() ||
        _clearanceMaps[floorIdx].empty())
        return _wallPenaltyClearancePx + 1.0;
    int w = _cacheWidth[floorIdx], h = _cacheHeight[floorIdx];
    int ix = (int)px, iy = (int)py;
    if (ix < 0 || ix >= w || iy < 0 || iy >= h) return 0.0;
    return (double)_clearanceMaps[floorIdx][(size_t)iy * w + ix];
}

double PRMMultiFloor::minClearanceAlongEdge(int floorIdx, const BIpoint& a, const BIpoint& b) const
{
    if (floorIdx < 0 || floorIdx >= (int)_clearanceMaps.size() ||
        _clearanceMaps[floorIdx].empty())
        return _wallPenaltyClearancePx + 1.0;
    double dist = a % b;
    int n = std::max(2, (int)(dist / 5.0) + 1);  // 每 ~5px 采样一次
    double c = 1e9;
    for (int i = 0; i <= n; ++i)
    {
        double t = (double)i / (double)n;
        c = std::min(c, clearanceAt(floorIdx, a.x + t * (b.x - a.x),
                                    a.y + t * (b.y - a.y)));
    }
    return c;
}

bool PRMMultiFloor::buildRoadmap(int floorIdx)
{
    if (floorIdx < 0 || floorIdx >= _numFloors)
    {
        printf("buildRoadmap: invalid floor index %d\r\n", floorIdx);
        return false;
    }

    // 复用现有建图逻辑：传入无效 start/goal，buildPRMForFloor 只构建采样节点 + 边 + 可通行缓存
    // （其 Step2 因 start.x<0 且无 connection 而不插入任何特殊节点）
    BIpoint dummyStart{-1, -1, -1}, dummyGoal{-1, -1, -1};
    buildPRMForFloor(floorIdx, dummyStart, dummyGoal);

    const PRMGraph& prm = _prmGraphs[floorIdx];
    if (prm.nodes.empty())
    {
        printf("buildRoadmap: no traversable nodes on floor %d!\r\n", floorIdx);
        _roadmapBuilt[floorIdx] = false;
        return false;
    }

    // 重建并缓存空间网格（buildPRMForFloor 内的 grid 是局部变量），供 replan 给临时 start/goal 找邻居
    int w = _cacheWidth[floorIdx], h = _cacheHeight[floorIdx];
    double r = prm.r_nei > 0 ? prm.r_nei : _prmRNei;
    int gridW = std::max(1, (int)(w / r));
    int gridH = std::max(1, (int)(h / r));
    auto& grid = _roadmapGrid[floorIdx];
    grid.assign((size_t)gridW * gridH, {});
    for (size_t i = 0; i < prm.nodes.size(); i++)
    {
        int gx = std::min(gridW - 1, std::max(0, (int)(prm.nodes[i].point.x / r)));
        int gy = std::min(gridH - 1, std::max(0, (int)(prm.nodes[i].point.y / r)));
        grid[(size_t)gy * gridW + gx].push_back((int)i);
    }
    _roadmapGridW[floorIdx] = gridW;
    _roadmapGridH[floorIdx] = gridH;
    _roadmapBuilt[floorIdx] = true;

    printf("buildRoadmap: floor %d cached %zu nodes, grid [%dx%d]\r\n",
           floorIdx, prm.nodes.size(), gridW, gridH);
    return true;
}

double PRMMultiFloor::replan(int floorIdx, BIpoint start, BIpoint goal,
                             const std::vector<DynamicObstacle>& obstacles,
                             std::list<BIpoint>& path)
{
    path.clear();
    if (floorIdx < 0 || floorIdx >= _numFloors) return -1.0;
    if (!_roadmapBuilt[floorIdx])
    {
        printf("replan: roadmap not built for floor %d\r\n", floorIdx);
        return -1.0;
    }
    if (!isPointTraversable(floorIdx, start.x, start.y))
    {
        printf("replan: start F%d (%.1f,%.1f) on obstacle\r\n", floorIdx, start.x, start.y);
        return -1.0;
    }
    if (!isPointTraversable(floorIdx, goal.x, goal.y))
    {
        printf("replan: goal F%d (%.1f,%.1f) on obstacle\r\n", floorIdx, goal.x, goal.y);
        return -1.0;
    }

    int64_t t0 = utime_ns();
    start.floor = floorIdx;
    goal.floor = floorIdx;

    const PRMGraph& cached = _prmGraphs[floorIdx];
    const int N = (int)cached.nodes.size();
    const int startId = N;
    const int goalId = N + 1;

    // 拷贝静态节点为工作副本（不修改缓存路网），追加临时 start/goal 节点
    std::vector<PRMNode> work = cached.nodes;  // 深拷贝（含各自的 neighbors）
    work.push_back({start, {}});
    work.push_back({goal, {}});

    // 给临时 start/goal 找邻居：3×3 网格 + r_nei 预检 + segmentFreeCached，双向加边
    double r = cached.r_nei > 0 ? cached.r_nei : _prmRNei;
    int gridW = _roadmapGridW[floorIdx], gridH = _roadmapGridH[floorIdx];
    const auto& grid = _roadmapGrid[floorIdx];
    auto connectTemp = [&](int id)
    {
        int gx = (int)(work[id].point.x / r);
        int gy = (int)(work[id].point.y / r);
        for (int dy = -1; dy <= 1; dy++)
        {
            int ny = gy + dy;
            if (ny < 0 || ny >= gridH) continue;
            for (int dx = -1; dx <= 1; dx++)
            {
                int nx = gx + dx;
                if (nx < 0 || nx >= gridW) continue;
                for (int j : grid[(size_t)ny * gridW + nx])
                {
                    if (j == id) continue;
                    double ddx = work[id].point.x - work[j].point.x;
                    double ddy = work[id].point.y - work[j].point.y;
                    if (ddx * ddx + ddy * ddy > r * r) continue;
                    if (segmentFreeCached(floorIdx, work[id].point, work[j].point))
                    {
                        work[id].neighbors.push_back(j);
                        work[j].neighbors.push_back(id);
                    }
                }
            }
        }
    };
    connectTemp(startId);
    connectTemp(goalId);

    // 障碍物边阻挡判定：任一障碍圆心到该边的最近距离 <= radius 即视为被挡
    auto edgeBlocked = [&](int u, int v) -> bool
    {
        const BIpoint& a = work[u].point;
        const BIpoint& b = work[v].point;
        for (const auto& obs : obstacles)
            if (Point2LineDistance(obs.center, a, b) <= obs.radius) return true;
        return false;
    };

    // Dijkstra（单层，状态 = nodeIdx）
    std::vector<double> gScore(work.size(), std::numeric_limits<double>::max());
    std::vector<int> parent(work.size(), -1);
    std::vector<char> visited(work.size(), 0);
    struct PQE
    {
        double g;
        int node;
        bool operator>(const PQE& o) const { return g > o.g; }
    };
    std::priority_queue<PQE, std::vector<PQE>, std::greater<PQE>> open;

    gScore[startId] = 0.0;
    open.push({0.0, startId});

    bool found = false;
    while (!open.empty())
    {
        PQE cur = open.top();
        open.pop();
        if (visited[cur.node]) continue;
        visited[cur.node] = 1;
        if (cur.node == goalId) { found = true; break; }

        for (int nb : work[cur.node].neighbors)
        {
            if (visited[nb]) continue;
            if (edgeBlocked(cur.node, nb)) continue;
            double ec = work[cur.node].point % work[nb].point;
            double ng = cur.g + ec;
            if (ng < gScore[nb])
            {
                gScore[nb] = ng;
                parent[nb] = cur.node;
                open.push({ng, nb});
            }
        }
    }

    if (!found)
    {
        printf("replan: no path to goal (obstacles=%zu)\r\n", obstacles.size());
        return -1.0;
    }

    // 回溯重建路径
    std::list<BIpoint> result;
    for (int c = goalId; c >= 0; c = parent[c])
    {
        BIpoint pt = work[c].point;
        pt.floor = floorIdx;
        result.push_front(pt);
    }
    path = result;

    int64_t t1 = utime_ns();
    printf("replan: %zu pts, cost=%.2f, %.2f ms (obstacles=%zu)\r\n",
           path.size(), gScore[goalId], (t1 - t0) / 1e6, obstacles.size());
    return gScore[goalId];
}

// 多楼层重规划：复用各层缓存路网（含连接点节点），虚拟插入跨楼层 start/goal，
// 多楼层 Dijkstra（同楼层边按动态障碍阻挡 + 跨楼层连接边）。不重建路网、不修改缓存。
double PRMMultiFloor::replanMultiFloor(MultiFloorTask& task,
                                       const std::vector<DynamicObstacle>& obstacles)
{
    task.path.clear();
    task.totalCost = 0;

    int startFloor = task.start.floor;
    int goalFloor = task.goal.floor;
    if (startFloor < 0 || startFloor >= _numFloors ||
        goalFloor < 0 || goalFloor >= _numFloors) return -1.0;
    if (!_roadmapBuilt[startFloor] || !_roadmapBuilt[goalFloor])
    {
        printf("replanMultiFloor: roadmap not built (F%d/F%d)\r\n", startFloor, goalFloor);
        return -1.0;
    }
    if (!isPointTraversable(startFloor, task.start.x, task.start.y))
    {
        printf("replanMultiFloor: start F%d (%.1f,%.1f) on obstacle\r\n",
               startFloor, task.start.x, task.start.y);
        return -1.0;
    }
    if (!isPointTraversable(goalFloor, task.goal.x, task.goal.y))
    {
        printf("replanMultiFloor: goal F%d (%.1f,%.1f) on obstacle\r\n",
               goalFloor, task.goal.x, task.goal.y);
        return -1.0;
    }

    int64_t t0 = utime_ns();
    BIpoint S = task.start, G = task.goal;
    S.floor = startFloor; G.floor = goalFloor;

    // 拷贝各层缓存节点为工作副本（追加临时 start/goal 不污染缓存）
    std::vector<std::vector<PRMNode>> work(_numFloors);
    for (int f = 0; f < _numFloors; ++f) work[f] = _prmGraphs[f].nodes;

    int startId = (int)work[startFloor].size();
    work[startFloor].push_back({S, {}});
    int goalId = (int)work[goalFloor].size();
    work[goalFloor].push_back({G, {}});

    // 给临时 start/goal 连邻居（用缓存空间网格 + segmentFreeCached，双向加边）
    auto connectTemp = [&](int fl, int id)
    {
        double r = _prmGraphs[fl].r_nei > 0 ? _prmGraphs[fl].r_nei : _prmRNei;
        int gw = _roadmapGridW[fl], gh = _roadmapGridH[fl];
        const auto& grid = _roadmapGrid[fl];
        int gx = (int)(work[fl][id].point.x / r);
        int gy = (int)(work[fl][id].point.y / r);
        for (int dy = -1; dy <= 1; ++dy)
        {
            int ny = gy + dy;
            if (ny < 0 || ny >= gh) continue;
            for (int dx = -1; dx <= 1; ++dx)
            {
                int nx = gx + dx;
                if (nx < 0 || nx >= gw) continue;
                for (int j : grid[(size_t)ny * gw + nx])
                {
                    if (j == id) continue;
                    double ddx = work[fl][id].point.x - work[fl][j].point.x;
                    double ddy = work[fl][id].point.y - work[fl][j].point.y;
                    if (ddx * ddx + ddy * ddy > r * r) continue;
                    if (segmentFreeCached(fl, work[fl][id].point, work[fl][j].point))
                    {
                        work[fl][id].neighbors.push_back(j);
                        work[fl][j].neighbors.push_back(id);
                    }
                }
            }
        }
    };
    connectTemp(startFloor, startId);
    connectTemp(goalFloor, goalId);

    // 跨楼层连接边（用缓存 connIdToNodeIdx；连接点节点在 work 中索引不变）
    struct CrossEdge { int tgtFloor; int tgtNode; double cost; };
    std::map<std::pair<int, int>, std::vector<CrossEdge>> crossEdges;
    for (const auto& conn : _multiFloorGraph.connections)
    {
        int f1 = conn.pointFrom.floor, f2 = conn.pointTo.floor;
        if (f1 < 0 || f1 >= _numFloors || f2 < 0 || f2 >= _numFloors) continue;
        auto it1 = _prmGraphs[f1].connIdToNodeIdx.find(conn.id);
        auto it2 = _prmGraphs[f2].connIdToNodeIdx.find(conn.id);
        if (it1 == _prmGraphs[f1].connIdToNodeIdx.end() ||
            it2 == _prmGraphs[f2].connIdToNodeIdx.end()) continue;
        crossEdges[{f1, it1->second}].push_back({f2, it2->second, conn.cost});
        crossEdges[{f2, it2->second}].push_back({f1, it1->second, conn.cost});
    }

    // 同楼层边是否被该层动态障碍挡住
    auto edgeBlocked = [&](int fl, int u, int v) -> bool
    {
        const BIpoint& a = work[fl][u].point;
        const BIpoint& b = work[fl][v].point;
        for (const auto& obs : obstacles)
            if (obs.floor == fl && Point2LineDistance(obs.center, a, b) <= obs.radius) return true;
        return false;
    };

    // 多楼层 Dijkstra（状态 = (floor, nodeIdx)）
    struct PQE
    {
        double g;
        int floor;
        int node;
        int pFloor;
        int pNode;
        bool operator>(const PQE& o) const { return g > o.g; }
    };
    struct PairHash
    {
        size_t operator()(const std::pair<int, int>& p) const
        {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };
    struct ClosedInfo { double g; int pFloor; int pNode; };
    std::unordered_map<std::pair<int, int>, ClosedInfo, PairHash> closed;
    std::priority_queue<PQE, std::vector<PQE>, std::greater<PQE>> open;

    open.push({0.0, startFloor, startId, -1, -1});
    bool found = false;
    PQE goalEntry;
    while (!open.empty())
    {
        PQE cur = open.top();
        open.pop();
        auto key = std::make_pair(cur.floor, cur.node);
        if (closed.count(key)) continue;
        closed[key] = {cur.g, cur.pFloor, cur.pNode};
        if (cur.floor == goalFloor && cur.node == goalId) { found = true; goalEntry = cur; break; }

        // 同楼层邻居（跳过被障碍挡住的边）
        for (int nb : work[cur.floor][cur.node].neighbors)
        {
            auto nk = std::make_pair(cur.floor, nb);
            if (closed.count(nk)) continue;
            if (edgeBlocked(cur.floor, cur.node, nb)) continue;
            double ec = work[cur.floor][cur.node].point % work[cur.floor][nb].point;
            open.push({cur.g + ec, cur.floor, nb, cur.floor, cur.node});
        }
        // 跨楼层连接边
        auto cf = crossEdges.find(key);
        if (cf != crossEdges.end())
            for (const auto& ce : cf->second)
            {
                auto nk = std::make_pair(ce.tgtFloor, ce.tgtNode);
                if (closed.count(nk)) continue;
                open.push({cur.g + ce.cost, ce.tgtFloor, ce.tgtNode, cur.floor, cur.node});
            }
    }

    if (!found) { printf("replanMultiFloor: no path\r\n"); return -1.0; }

    // 回溯重建路径
    std::list<BIpoint> path;
    int cf2 = goalEntry.floor, cn = goalEntry.node;
    while (cn >= 0)
    {
        BIpoint pt = work[cf2][cn].point;
        pt.floor = cf2;
        path.push_front(pt);
        auto it = closed.find(std::make_pair(cf2, cn));
        if (it == closed.end()) break;
        cf2 = it->second.pFloor;
        cn = it->second.pNode;
    }
    task.path = path;
    task.totalCost = goalEntry.g;

    int64_t t1 = utime_ns();
    printf("replanMultiFloor: %zu pts, cost=%.2f, %.2f ms (obstacles=%zu)\r\n",
           path.size(), goalEntry.g, (t1 - t0) / 1e6, obstacles.size());
    return goalEntry.g;
}

const PRMGraph& PRMMultiFloor::getRoadmap(int floorIdx) const
{
    static const PRMGraph kEmpty = PRMGraph{};
    if (floorIdx < 0 || floorIdx >= _numFloors) return kEmpty;
    return _prmGraphs[floorIdx];
}

bool PRMMultiFloor::roadmapBuilt(int floorIdx) const
{
    return (floorIdx >= 0 && floorIdx < _numFloors) ? _roadmapBuilt[floorIdx] : false;
}

bool PRMMultiFloor::pointTraversable(int floorIdx, double x, double y) const
{
    return isPointTraversable(floorIdx, x, y);
}

double PRMMultiFloor::planInSingleFloor(int32_t floorIndex,
                                        BIpoint start, BIpoint goal,
                                        int32_t startPoly, int32_t goalPoly,
                                        std::list<BIpoint>& path)
{
    if (floorIndex < 0 || floorIndex >= _numFloors)
        return -1;

    if (startPoly < 0 || goalPoly < 0)
    {
        printf("error: start or goal on obstacle!!\r\n");
        return -1;
    }

    if (startPoly == goalPoly)
    {
        path.clear();
        path.push_back(start);
        path.push_back(goal);
        for (auto& pt : path) pt.floor = floorIndex;
        return start % goal;
    }

    BIgraph& graph = _multiFloorGraph.floorGraphs[floorIndex];

    // Dijkstra：按 ComputeLeastHomotopyPath 计算的几何成本排序
    struct DijkstraNode
    {
        double cost;
        int32_t polyIndex;
        std::vector<int32_t> path;
        bool isgoal = false;

        bool operator>(const DijkstraNode& other) const
        {
            return cost > other.cost;
        }
    };

    std::priority_queue<DijkstraNode, std::vector<DijkstraNode>, std::greater<DijkstraNode>> pq;
    std::vector<bool> visited(graph.freepolygonList.size(), false);

    std::vector<int32_t> initialPath = {startPoly};
    pq.push({0, startPoly, initialPath, false});
    visited[startPoly] = true;

    std::vector<int32_t> polyPath;

    while (!pq.empty())
    {
        DijkstraNode current = pq.top();
        pq.pop();

        if (current.isgoal)
        {
            polyPath = current.path;
            break;
        }

        if (current.polyIndex == goalPoly)
        {
            // 到达goalPoly不等于找到最优路径，需要计算沿着当前path的真实成本
            std::vector<BIline> goalCpath;
            goalCpath.push_back({start, start});
            for (size_t i = 0; i < current.path.size() - 1; i++)
            {
                int32_t ci = graph.invnode2cutlineMap[{current.path[i], current.path[i + 1]}];
                goalCpath.push_back(graph.cutlineList[ci].line);
            }
            goalCpath.push_back({goal, goal});

            std::list<BIpoint> goalPath;
            double goalCost;
            ComputeLeastHomotopyPath(graph, goalCpath, goalPath, goalCost);

            std::vector<int32_t> goalPolyPath = current.path;
            pq.push({goalCost, goalPoly, goalPolyPath, true});
            continue;
        }

        BIfreepolygon& poly = graph.freepolygonList[current.polyIndex];
        for (int32_t neighborPoly : poly.polygonlink)
        {
            if (neighborPoly < 0 || neighborPoly >= (int32_t)graph.freepolygonList.size()) continue;
            if (neighborPoly != goalPoly && visited[neighborPoly]) continue;

            visited[neighborPoly] = true;

            // 构建 cpath：起点 → 中间分割线 → 最后一个分割线中点
            std::vector<BIline> tmpCpath;
            tmpCpath.push_back({start, start});

            // 只添加中间分割线（不包括最后一条，最后一条的中点是终点）
            for (size_t i = 0; i < current.path.size() - 1; i++)
            {
                int32_t ci = graph.invnode2cutlineMap[{current.path[i], current.path[i + 1]}];
                tmpCpath.push_back(graph.cutlineList[ci].line);
            }

            // 最后一条分割线的中点是终点
            int32_t lastCutlineIdx = graph.invnode2cutlineMap[{current.polyIndex, neighborPoly}];
            BIpoint midPoint = {
                (graph.cutlineList[lastCutlineIdx].line.S.x + graph.cutlineList[lastCutlineIdx].line.E.x) / 2.0,
                (graph.cutlineList[lastCutlineIdx].line.S.y + graph.cutlineList[lastCutlineIdx].line.E.y) / 2.0
            };
            tmpCpath.push_back({midPoint, midPoint});

            std::list<BIpoint> tmpPath;
            double pathCost;
            ComputeLeastHomotopyPath(graph, tmpCpath, tmpPath, pathCost);

            if (tmpPath.empty()) continue;

            std::vector<int32_t> newPath = current.path;
            newPath.push_back(neighborPoly);
            pq.push({pathCost, neighborPoly, newPath, false});
        }
    }

    if (polyPath.empty())
    {
        printf("error: no path in floor %d!!\r\n", floorIndex);
        return -1;
    }

    std::vector<BIline> cpath;
    cpath.emplace_back(start, start);
    for (size_t i = 0; i < polyPath.size() - 1; i++)
    {
        int32_t cutlineIndex = graph.invnode2cutlineMap[{polyPath[i], polyPath[i + 1]}];
        cpath.push_back(graph.cutlineList[cutlineIndex].line);
    }
    cpath.emplace_back(goal, goal);

    double pathCost;
    ComputeLeastHomotopyPath(graph, cpath, path, pathCost);

    for (auto& pt : path) pt.floor = floorIndex;
    return pathCost;
}

double PRMMultiFloor::plan(MultiFloorTask& task)
{
    task.path.clear();
    task.totalCost = 0;

    int startFloor = task.start.floor;
    int goalFloor = task.goal.floor;

    // Validate start/goal traversability
    if (!isPointTraversable(startFloor, task.start.x, task.start.y))
    {
        printf("error: start on obstacle!!\r\n");
        return -1;
    }
    if (!isPointTraversable(goalFloor, task.goal.x, task.goal.y))
    {
        printf("error: goal on obstacle!!\r\n");
        return -1;
    }

    printf("PRM planning: F%d -> F%d, k_nodes=%d, r_nei=%.1f\r\n",
           startFloor, goalFloor, _prmKNodes, _prmRNei);

    // Step 1: 确定需要构建 PRM 的楼层，只构建必要楼层
    {
        // BFS 从起点楼层经连接边找到目标楼层
        std::set<int> neededFloors;
        std::map<int, std::vector<int>> floorAdj;  // floor → neighbor floors via connections
        for (const auto& conn : _multiFloorGraph.connections)
        {
            floorAdj[conn.pointFrom.floor].push_back(conn.pointTo.floor);
            floorAdj[conn.pointTo.floor].push_back(conn.pointFrom.floor);
        }

        std::queue<int> q;
        std::set<int> visited;
        q.push(startFloor);
        visited.insert(startFloor);
        while (!q.empty())
        {
            int f = q.front(); q.pop();
            neededFloors.insert(f);
            if (f == goalFloor) continue;
            for (int nf : floorAdj[f])
                if (!visited.count(nf)) { visited.insert(nf); q.push(nf); }
        }
        // 如果没有路径到目标楼层（同楼层或未连接），至少要构建起终点楼层
        neededFloors.insert(startFloor);
        neededFloors.insert(goalFloor);

        printf("Building PRM for %zu/%d floors\r\n", neededFloors.size(), _numFloors);
        for (int i = 0; i < _numFloors; i++)
        {
            if (!neededFloors.count(i)) continue;
            BIpoint st = (i == startFloor) ? task.start : BIpoint{-1, -1, -1};
            BIpoint gl = (i == goalFloor)  ? task.goal  : BIpoint{-1, -1, -1};
            buildPRMForFloor(i, st, gl);
        }
    }

    // Step 2: Build cross-floor edge map
    // Key: (floor, nodeIdx) -> list of (targetFloor, targetNodeIdx, cost)
    struct CrossEdge
    {
        int tgtFloor;
        int tgtNode;
        double cost;
    };
    std::map<std::pair<int, int>, std::vector<CrossEdge>> crossEdges;

    for (const auto& conn : _multiFloorGraph.connections)
    {
        int f1 = conn.pointFrom.floor;
        int f2 = conn.pointTo.floor;
        if (f1 < 0 || f1 >= _numFloors || f2 < 0 || f2 >= _numFloors) continue;

        auto it1 = _prmGraphs[f1].connIdToNodeIdx.find(conn.id);
        auto it2 = _prmGraphs[f2].connIdToNodeIdx.find(conn.id);
        if (it1 == _prmGraphs[f1].connIdToNodeIdx.end() ||
            it2 == _prmGraphs[f2].connIdToNodeIdx.end())
        {
            printf("Warning: conn %d missing node on floor %d or %d\r\n", conn.id, f1, f2);
            continue;
        }

        int n1 = it1->second;
        int n2 = it2->second;
        crossEdges[{f1, n1}].push_back({f2, n2, conn.cost});
        crossEdges[{f2, n2}].push_back({f1, n1, conn.cost});
        printf("Cross edge: F%d[%d] <-> F%d[%d], cost=%.2f\r\n", f1, n1, f2, n2, conn.cost);
    }

    // Step 3: Find start and goal node indices in their PRM graphs
    // They are the last added nodes on their respective floors
    // We need to find them by checking the nodes we added
    int startNodeIdx = -1, goalNodeIdx = -1;

    // Start is always added as the last "special" point before connections
    // Better: search for the node with matching position
    PRMGraph& startPRM = _prmGraphs[startFloor];
    for (int i = (int)startPRM.nodes.size() - 1; i >= 0; i--)
    {
        double d = startPRM.nodes[i].point % task.start;
        if (d < 1e-6)
        {
            startNodeIdx = i;
            break;
        }
    }

    PRMGraph& goalPRM = _prmGraphs[goalFloor];
    for (int i = (int)goalPRM.nodes.size() - 1; i >= 0; i--)
    {
        double d = goalPRM.nodes[i].point % task.goal;
        if (d < 1e-6)
        {
            goalNodeIdx = i;
            break;
        }
    }

    if (startNodeIdx < 0 || goalNodeIdx < 0)
    {
        printf("error: start or goal node not found in PRM!\r\n");
        return -1;
    }
    printf("Start: F%d[%d], Goal: F%d[%d]\r\n", startFloor, startNodeIdx, goalFloor, goalNodeIdx);

    // Step 4: A* search (Dijkstra: f = g only)
    int64_t t_search_start = utime_ns();
    struct PQEntry
    {
        double g;
        int floor;
        int nodeIdx;
        int parentFloor;
        int parentNodeIdx;

        bool operator>(const PQEntry& other) const { return g > other.g; }
    };

    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> open;

    // Closed set & parent tracking
    struct ClosedInfo
    {
        double g;
        int parentFloor;
        int parentNodeIdx;
    };
    // Hash for pair<int,int>
    struct PairHash
    {
        size_t operator()(const std::pair<int, int>& p) const
        {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<int, int>, ClosedInfo, PairHash> closed;

    open.push({0.0, startFloor, startNodeIdx, -1, -1});

    bool found = false;
    PQEntry goalEntry;

    while (!open.empty())
    {
        PQEntry cur = open.top();
        open.pop();

        auto key = std::make_pair(cur.floor, cur.nodeIdx);
        if (closed.count(key)) continue;
        closed[key] = {cur.g, cur.parentFloor, cur.parentNodeIdx};

        // Check goal
        if (cur.floor == goalFloor && cur.nodeIdx == goalNodeIdx)
        {
            found = true;
            goalEntry = cur;
            break;
        }

        PRMGraph& prm = _prmGraphs[cur.floor];

        // Expand same-floor neighbors
        const PRMNode& curnode = prm.nodes[cur.nodeIdx];
        for (int nb : curnode.neighbors)
        {
            auto nk = std::make_pair(cur.floor, nb);
            if (closed.count(nk)) continue;
            double edgeCost = curnode.point % prm.nodes[nb].point;
            open.push({cur.g + edgeCost, cur.floor, nb, cur.floor, cur.nodeIdx});
        }

        // Expand cross-floor edges
        auto cfIt = crossEdges.find(key);
        if (cfIt != crossEdges.end())
        {
            for (const auto& ce : cfIt->second)
            {
                auto nk = std::make_pair(ce.tgtFloor, ce.tgtNode);
                if (closed.count(nk)) continue;
                open.push({cur.g + ce.cost, ce.tgtFloor, ce.tgtNode, cur.floor, cur.nodeIdx});
            }
        }
    }

    int64_t t_search_end = utime_ns();
    printf("A* search + path reconstruction: %.1f ms\r\n", (t_search_end - t_search_start) / 1e6);

    if (!found)
    {
        printf("error: no path to goal!!\r\n");
        return -1;
    }

    // Step 5: Reconstruct path
    std::list<BIpoint> path;
    int curFloor = goalEntry.floor;
    int curNode = goalEntry.nodeIdx;

    while (curNode >= 0)
    {
        BIpoint pt = _prmGraphs[curFloor].nodes[curNode].point;
        pt.floor = curFloor;
        path.push_front(pt);

        auto key = std::make_pair(curFloor, curNode);
        auto it = closed.find(key);
        if (it == closed.end()) break;

        int pFloor = it->second.parentFloor;
        int pNode  = it->second.parentNodeIdx;
        curFloor = pFloor;
        curNode  = pNode;
    }

    task.path = path;
    task.totalCost = goalEntry.g;

    printf("PRM path found: %zu points, total cost: %.2f\r\n", path.size(), goalEntry.g);
    return task.totalCost;
}

// ========== 全局融合图：离线图融合 + 在线起点插入 ==========

const FusedGraph& PRMMultiFloor::getFusedGraph() const { return _fused; }
const std::vector<TopoPolygon>& PRMMultiFloor::getTopoPolygons() const { return _topoPolygons; }
bool PRMMultiFloor::fusedBuilt() const { return _fusedBuilt; }

void PRMMultiFloor::setFuseParams(int kNeighbors, double crossRadius)
{
    _fuseKNodes = std::max(1, kNeighbors);
    if (crossRadius > 0) _fuseCrossRadius = crossRadius;
}

bool PRMMultiFloor::loadConnectionsJson(const std::string& path)
{
    std::ifstream fin(path);
    if (!fin.is_open())
    {
        printf("loadConnectionsJson: 无法打开 %s\r\n", path.c_str());
        return false;
    }
    json j;
    fin >> j;

    _topoPolygons.clear();
    // 顶层 expandRadius 兼作多边形内跨层连边的距离阈值（可再用 setFuseParams 覆盖）
    if (j.contains("expandRadius") && j["expandRadius"].is_number())
        _fuseCrossRadius = j["expandRadius"].get<double>();

    if (j.contains("connections") && j["connections"].is_array())
    {
        for (const auto& c : j["connections"])
        {
            TopoPolygon tp;
            tp.fromFloor = c.value("fromFloor", -1);
            tp.toFloor   = c.value("toFloor", -1);
            tp.cost      = c.value("cost", 15.0);
            if (c.contains("polygon") && c["polygon"].is_array())
            {
                for (const auto& v : c["polygon"])
                {
                    if (v.is_array() && v.size() >= 2)
                        tp.vertices.push_back(BIpoint{v.at(0).get<double>(),
                                                      v.at(1).get<double>(), 0});
                }
            }
            // 有效条件：楼层合法 + 至少 3 个顶点（成面）
            if (tp.fromFloor >= 0 && tp.toFloor >= 0 && tp.vertices.size() >= 3)
                _topoPolygons.push_back(std::move(tp));
        }
    }
    printf("loadConnectionsJson: 从 %s 解析到 %zu 个并集多边形, crossRadius=%.1f\r\n",
           path.c_str(), _topoPolygons.size(), _fuseCrossRadius);
    return !_topoPolygons.empty();
}

bool PRMMultiFloor::fuseGraph()
{
    if (_numFloors <= 0)
    {
        printf("fuseGraph: 未初始化楼层数\r\n");
        return false;
    }

    // 1. 把所有楼层已缓存的路网节点缝合成全局扁平节点表，赋全局 id
    _fused = FusedGraph{};
    _fused.floorNodeStart.assign((size_t)_numFloors + 1, 0);
    for (int f = 0; f < _numFloors; ++f)
    {
        _fused.floorNodeStart[f] = (int)_fused.nodes.size();
        if (!_roadmapBuilt[f])
            printf("fuseGraph: floor %d 尚未 buildRoadmap，节点可能为空\r\n", f);
        const PRMGraph& prm = _prmGraphs[f];
        for (size_t i = 0; i < prm.nodes.size(); ++i)
        {
            MapNode n;
            n.id = (int)_fused.nodes.size();
            n.x = prm.nodes[i].point.x;
            n.y = prm.nodes[i].point.y;
            n.floor_id = f;
            _fused.nodes.push_back(n);
        }
    }
    _fused.floorNodeStart[_numFloors] = (int)_fused.nodes.size();
    _fused.adj.assign(_fused.nodes.size(), {});
    _fused.edges.clear();

    // 无向加边：一条边存一份，两端邻接表共享同一下标
    auto addEdge = [&](int u, int v, double w, bool cross)
    {
        int e = (int)_fused.edges.size();
        _fused.edges.push_back(MapEdge{u, v, w, cross});
        _fused.adj[u].push_back(e);
        _fused.adj[v].push_back(e);
    };

    // 2. 同层边：从各层 PRM 的 neighbors 拷贝，权重 = 欧氏距离（可加贴墙惩罚）
    for (int f = 0; f < _numFloors; ++f)
    {
        const PRMGraph& prm = _prmGraphs[f];
        int base = _fused.floorNodeStart[f];
        for (size_t i = 0; i < prm.nodes.size(); ++i)
        {
            int u = base + (int)i;
            for (int nb : prm.nodes[i].neighbors)
            {
                if (nb <= (int)i) continue;  // 每条同层边只加一次
                int v = base + nb;
                double w = prm.nodes[i].point % prm.nodes[nb].point;
                // 贴墙惩罚：边越贴近障碍(clearance 越小)权重越大，引导路径走安全通道
                if (_wallPenaltyGain > 0.0)
                {
                    double c = minClearanceAlongEdge(f, prm.nodes[i].point, prm.nodes[nb].point);
                    if (c < _wallPenaltyClearancePx)
                        w *= 1.0 + _wallPenaltyGain * (1.0 - c / _wallPenaltyClearancePx);
                }
                addEdge(u, v, w, false);
            }
        }
    }

    // 3. 跨层边：对每个拓扑并集多边形，取两楼层落在多边形内的节点，
    //    欧氏距离 < _fuseCrossRadius 则建跨层边（权重 = 楼梯代价 cost）
    for (const TopoPolygon& tp : _topoPolygons)
    {
        int f1 = tp.fromFloor, f2 = tp.toFloor;
        if (f1 < 0 || f1 >= _numFloors || f2 < 0 || f2 >= _numFloors) continue;

        std::vector<int> in1, in2;
        for (int id = _fused.floorNodeStart[f1]; id < _fused.floorNodeStart[f1 + 1]; ++id)
            if (tp.contains(_fused.nodes[id].x, _fused.nodes[id].y)) in1.push_back(id);
        for (int id = _fused.floorNodeStart[f2]; id < _fused.floorNodeStart[f2 + 1]; ++id)
            if (tp.contains(_fused.nodes[id].x, _fused.nodes[id].y)) in2.push_back(id);

        int cnt = 0;
        for (int a : in1)
        {
            for (int b : in2)
            {
                double dx = _fused.nodes[a].x - _fused.nodes[b].x;
                double dy = _fused.nodes[a].y - _fused.nodes[b].y;
                if (std::sqrt(dx * dx + dy * dy) < _fuseCrossRadius)
                {
                    addEdge(a, b, tp.cost, true);
                    ++cnt;
                }
            }
        }
        printf("fuseGraph: F%d<->F%d 多边形内节点 %zu/%zu，跨层边 %d\r\n",
               f1, f2, in1.size(), in2.size(), cnt);
    }

    // 4. 每层建 KD-Tree（存全局节点 id），供在线插入 KNN
    _fusedKD.assign((size_t)_numFloors, KDTree2D{});
    for (int f = 0; f < _numFloors; ++f)
    {
        int cnt = _fused.floorNodeStart[f + 1] - _fused.floorNodeStart[f];
        std::vector<BIpoint> pts;
        std::vector<int> ids;
        pts.reserve((size_t)cnt);
        ids.reserve((size_t)cnt);
        for (int id = _fused.floorNodeStart[f]; id < _fused.floorNodeStart[f + 1]; ++id)
        {
            pts.push_back(BIpoint{_fused.nodes[id].x, _fused.nodes[id].y, f});
            ids.push_back(id);
        }
        if (!pts.empty()) _fusedKD[f].build(pts, ids);
    }

    _fusedBuilt = true;
    printf("fuseGraph: 融合完成，总节点 %zu，总边 %zu\r\n", _fused.nodes.size(), _fused.edges.size());
    return true;
}

double PRMMultiFloor::planFused(MultiFloorTask& task, const std::vector<DynamicObstacleCloud>& dynClouds)
{
    task.path.clear();
    task.totalCost = 0.0;

    if (!_fusedBuilt)
    {
        printf("planFused: 请先调用 fuseGraph()\r\n");
        return -1.0;
    }
    int startFloor = task.start.floor;
    int goalFloor = task.goal.floor;
    if (startFloor < 0 || startFloor >= _numFloors ||
        goalFloor < 0 || goalFloor >= _numFloors)
    {
        printf("planFused: 起终点楼层非法 (%d -> %d)\r\n", startFloor, goalFloor);
        return -1.0;
    }
    if (!isPointTraversableStatic(startFloor, task.start.x, task.start.y))
    {
        printf("planFused: 起点 F%d (%.1f,%.1f) 在静态障碍上\r\n", startFloor, task.start.x, task.start.y);
        return -1.0;
    }
    if (!isPointTraversableStatic(goalFloor, task.goal.x, task.goal.y))
    {
        printf("planFused: 终点 F%d (%.1f,%.1f) 在静态障碍上\r\n", goalFloor, task.goal.x, task.goal.y);
        return -1.0;
    }

    // 工作副本：不修改融合图本身，sim2d 多次重规划也不累积临时节点
    std::vector<MapNode> nodes = _fused.nodes;
    std::vector<MapEdge> edges = _fused.edges;
    std::vector<std::vector<int>> adj = _fused.adj;

    // 在线插入 start/goal：多边形感知 KNN
    auto insertNode = [&](const BIpoint& pt) -> int
    {
        int id = (int)nodes.size();
        nodes.push_back(MapNode{id, pt.x, pt.y, pt.floor});
        adj.push_back({});

        // 情况1：pt 落在「本楼层作为端点」的并集多边形内 → 同时查该过渡区两端楼层的 KD-Tree
        //   （"无视 floor_id" 的落点：不限定只连 floor_id 那一层，两端都连，兼容过渡期楼层号未刷新；
        //     但只认「机器人当前楼层参与的」过渡区，避免两个楼梯井重叠在相同 (x,y) 却服务
        //     不同楼层时误连到不相邻楼层——即避免"跳楼"）
        // 情况2：普通平地区域 → 只查本层 KD-Tree
        std::set<int> floors;
        for (const TopoPolygon& tp : _topoPolygons)
        {
            if (pt.floor != tp.fromFloor && pt.floor != tp.toFloor) continue;
            if (tp.contains(pt.x, pt.y))
            {
                floors.insert(tp.fromFloor);
                floors.insert(tp.toFloor);
            }
        }
        if (floors.empty())
            floors.insert(pt.floor);

        for (int f : floors)
        {
            if (f < 0 || f >= _numFloors) continue;
            std::vector<int> kn;
            _fusedKD[f].kNearest(BIpoint{pt.x, pt.y, f}, _fuseKNodes, kn);
            for (int nb : kn)
            {
                double dx = pt.x - nodes[nb].x;
                double dy = pt.y - nodes[nb].y;
                double w = std::sqrt(dx * dx + dy * dy);
                int e = (int)edges.size();
                edges.push_back(MapEdge{id, nb, w, false});
                adj[id].push_back(e);
                adj[nb].push_back(e);
            }
        }
        return id;
    };

    int startId = insertNode(task.start);
    int goalId  = insertNode(task.goal);

    // ---- 动态障碍点云：按楼层索引 + 受影响边过滤（safeplanner-2 思路）----
    // 先找障碍点影响半径内的路网节点，只对这些节点上的同层边做精确的
    // 边-点距离检查；其余边远离障碍，直接用原权重（KDTree 的替代：节点数
    // 有限，暴力半径判定即可）。在线插入的 start/goal 边长无上界，全部精确
    // 检查，避免长边"两端都在影响半径外、中间穿障碍"被漏掉。
    std::vector<const DynamicObstacleCloud*> cloudOf(_numFloors, nullptr);
    bool anyCloud = false;
    for (const auto & c : dynClouds) {
        if (c.floor >= 0 && c.floor < _numFloors && !c.points.empty()) {
            cloudOf[c.floor] = &c;
            anyCloud = true;
        }
    }
    std::vector<char> edgeAffected(edges.size(), 0);
    if (anyCloud) {
        for (int f = 0; f < _numFloors; ++f) {
            const DynamicObstacleCloud * c = cloudOf[f];
            if (!c) continue;
            const double affect = 1.5 * (c->forbidPx + c->safetyPx);  // safeplanner-2 同款
            const double affect2 = affect * affect;
            for (int u = _fused.floorNodeStart[f]; u < _fused.floorNodeStart[f + 1]; ++u) {
                const double ux = nodes[u].x, uy = nodes[u].y;
                bool nearObs = false;
                for (const auto & p : c->points) {
                    double dx = p.x - ux, dy = p.y - uy;
                    if (dx * dx + dy * dy < affect2) { nearObs = true; break; }
                }
                if (!nearObs) continue;
                for (int e : adj[u]) {
                    if (!edges[e].is_cross_map) edgeAffected[e] = 1;
                }
            }
        }
        // 在线插入节点（原始路网之外）的边不受影响半径保护，全部精确检查
        for (int u = _fused.floorNodeStart[_numFloors]; u < (int)nodes.size(); ++u) {
            for (int e : adj[u]) {
                if (!edges[e].is_cross_map) edgeAffected[e] = 1;
            }
        }
    }

    // Dijkstra（全局节点 id 上的单源最短路）
    const double DINF = std::numeric_limits<double>::infinity();
    int N = (int)nodes.size();
    std::vector<double> g(N, DINF);
    std::vector<int> parent(N, -1);

    using PQ = std::pair<double, int>;  // (cost, node_id)
    std::priority_queue<PQ, std::vector<PQ>, std::greater<PQ>> open;
    g[startId] = 0.0;
    open.push({0.0, startId});

    bool found = false;
    while (!open.empty())
    {
        auto [gc, u] = open.top();
        open.pop();
        if (gc > g[u]) continue;
        if (u == goalId) { found = true; break; }

        for (int e : adj[u])
        {
            const MapEdge& ed = edges[e];
            int v = (ed.from_id == u) ? ed.to_id : ed.from_id;

            // 动态障碍点级判定（safeplanner-2 computeDynamicEdgeCost 的连续版）：
            // 只查受影响的同层边——边到本层最近障碍点的距离 dmin（用点到线段
            // 距离代替逐段采样），dmin < forbidPx → 压到障碍点的膨胀区，切边；
            // 间隙 < safetyPx → 复用贴墙惩罚公式加权（越近代价越高，空间允许
            // 时规划自然绕远，窄通道只是变贵仍可走）。
            double w = ed.weight;
            if (anyCloud && !ed.is_cross_map && edgeAffected[e])
            {
                const DynamicObstacleCloud * c = cloudOf[nodes[u].floor_id];
                if (c)
                {
                    double dmin = std::numeric_limits<double>::infinity();
                    const BIpoint a{nodes[u].x, nodes[u].y, 0};
                    const BIpoint b{nodes[v].x, nodes[v].y, 0};
                    for (const auto & p : c->points)
                    {
                        double d = Point2LineDistance(p, a, b);
                        if (d < dmin) dmin = d;
                    }
                    if (dmin < c->forbidPx) continue;  // 切边
                    if (_wallPenaltyGain > 0.0)
                    {
                        double gap = dmin - c->forbidPx;
                        if (gap < c->safetyPx)
                            w *= 1.0 + _wallPenaltyGain * (1.0 - gap / c->safetyPx);
                    }
                }
            }

            double ng = gc + w;
            if (ng < g[v])
            {
                g[v] = ng;
                parent[v] = u;
                open.push({ng, v});
            }
        }
    }

    if (!found || g[goalId] >= DINF)
    {
        printf("planFused: 未找到路径 (F%d -> F%d)\r\n", startFloor, goalFloor);
        return -1.0;
    }

    // 回溯路径（每个点带 floor）
    std::list<BIpoint> path;
    for (int c = goalId; c != -1; c = parent[c])
        path.push_front(BIpoint{nodes[c].x, nodes[c].y, nodes[c].floor_id});

    task.path = path;
    task.totalCost = g[goalId];
    printf("planFused: 找到路径 %zu 点，总代价 %.2f\r\n", path.size(), task.totalCost);
    return task.totalCost;
}
