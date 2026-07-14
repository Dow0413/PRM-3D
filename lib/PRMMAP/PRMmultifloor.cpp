#include "PRMmultifloor.h"
#include <limits>
#include <queue>
#include <algorithm>
#include <random>
#include <unordered_map>

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

    // Step 1: Random sampling
    t0 = utime_ns();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distX(0.0, (double)(w - 1));
    std::uniform_real_distribution<double> distY(0.0, (double)(h - 1));

    int maxAttempts = effKNodes * 10;
    int attempts = 0;
    while ((int)prm.nodes.size() < effKNodes && attempts < maxAttempts)
    {
        attempts++;
        double x = distX(gen), y = distY(gen);
        if (pointFree(x, y))
        {
            PRMNode node;
            node.point = {x, y, floorIdx};
            prm.nodes.push_back(node);
        }
    }
    t1 = utime_ns();
    printf("Floor %d: sampling %zu nodes in %.1f ms (attempts=%d)\r\n",
           floorIdx, prm.nodes.size(), (t1-t0)/1e6, attempts);

    if (prm.nodes.empty())
    {
        printf("buildPRMForFloor: no traversable points found on floor %d!\r\n", floorIdx);
        return;
    }

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

    // Step 4: Build neighbor edges (grid candidate lookup, raw pointer collision check)
    t0 = utime_ns();
    int totalEdges = 0;
    // 预计算每个格子在当前节点坐标的 gx,gy（在循环里算也行，不预先算更省内存）
    for (size_t i = 0; i < prm.nodes.size(); i++)
    {
        int gx = (int)(prm.nodes[i].point.x / effRNei);
        int gy = (int)(prm.nodes[i].point.y / effRNei);

        // 检查 3×3 邻域格子
        for (int dy = -1; dy <= 1; dy++)
        {
            int ny = gy + dy;
            if (ny < 0 || ny >= gridH) continue;
            for (int dx = -1; dx <= 1; dx++)
            {
                int nx = gx + dx;
                if (nx < 0 || nx >= gridW) continue;

                for (int j : grid[ny * gridW + nx])
                {
                    if (j <= (int)i) continue;
                    double dx_ij = prm.nodes[i].point.x - prm.nodes[j].point.x;
                    double dy_ij = prm.nodes[i].point.y - prm.nodes[j].point.y;
                    // 快速距离预检（避免 sqrt），半径平方
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
    printf("Floor %d: %d edges built in %.1f ms\r\n", floorIdx, totalEdges, (t1-t0)/1e6);

    printf("Floor %d: total build = %.1f ms\r\n", floorIdx, (t1-t_start)/1e6);
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
