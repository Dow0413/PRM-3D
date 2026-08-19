#ifndef __PRM_MULTIFLOOR_H
#define __PRM_MULTIFLOOR_H

#include "PRMmap.h"

// 楼层连接点类型
enum class FloorConnectionType
{
    STAIRS,    // 楼梯
    ELEVATOR,  // 电梯
    RAMP       // 坡道
};

// 拓扑连接边：由两个 BIpoint 构成（含 floor 属性）
struct FloorConnection
{
    int32_t id;
    FloorConnectionType type;
    BIpoint pointFrom;          // 起点楼层连接点（floor 指定楼层）
    BIpoint pointTo;            // 终点楼层连接点（floor 指定楼层）
    int32_t polyIndexFrom;      // 起点楼层多边形索引（自动计算）
    int32_t polyIndexTo;        // 终点楼层多边形索引（自动计算）
    double cost;

    // A* 搜索用：反向连接（用于从 to 多边形扩展到 from 多边形）
    int32_t polyIndexTo_rev;    // 从 pointTo 多边形看的"入口"多边形索引（同 polyIndexFrom）
    int32_t polyIndexFrom_rev;  // 从 pointFrom 多边形看的"入口"多边形索引（同 polyIndexTo）
};

// 路径规划任务
struct MultiFloorTask
{
    BIpoint start;              // 起点（floor 指定楼层）
    BIpoint goal;               // 终点（floor 指定楼层）
    int32_t startPoly;          // 起点所在多边形
    int32_t goalPoly;           // 终点所在多边形
    std::list<BIpoint> path;    // 规划出的路径
    double totalCost;           // 总成本
};

// 跨楼层 PRM 图结构
struct MultiFloorGraph
{
    std::vector<BIgraph> floorGraphs;       // 每层的 PRM 图
    std::vector<FloorConnection> connections; // 拓扑连接边列表
};

// ========== PRM (Probabilistic Roadmap) Data Structures ==========

// 2D KD-Tree for efficient radius search
class KDTree2D
{
public:
    KDTree2D() {}
    void build(const std::vector<BIpoint>& points, const std::vector<int>& indices);
    void radiusSearch(const BIpoint& center, double radius, std::vector<int>& result) const;
    // 查询距 center 最近的 k 个点的 prmNodeIdx（按距离升序），供在线插入起点/终点用
    void kNearest(const BIpoint& center, int k, std::vector<int>& result) const;
    bool empty() const { return _nodes.empty(); }

private:
    struct KDNode
    {
        BIpoint pt;
        int prmNodeIdx;   // index into PRMGraph::nodes
        int splitDim;     // 0=x, 1=y
        int left;         // child index, -1 = none
        int right;        // child index, -1 = none
    };
    std::vector<KDNode> _nodes;

    int buildRecursive(std::vector<std::pair<BIpoint, int>>& pts, int begin, int end, int depth);
    void radiusSearchRecursive(int nodeIdx, const BIpoint& center, double radius, std::vector<int>& result) const;
    // 收集半径内节点的 _nodes 下标（不返回 prmNodeIdx，供 kNearest 排序距离用）
    void collectInRadius(int nodeIdx, const BIpoint& center, double radius, std::vector<int>& nodeIdxs) const;
};

// PRM node on a single floor
struct PRMNode
{
    BIpoint point;
    std::vector<int> neighbors;  // indices into PRMGraph::nodes (same floor)
};

// PRM graph for one floor
struct PRMGraph
{
    std::vector<PRMNode> nodes;
    int k_nodes;
    double r_nei;
    KDTree2D kdtree;
    std::unordered_map<int, int> connIdToNodeIdx;  // FloorConnection.id -> node index in this graph
};

// ========== End PRM Structures ==========

// 动态障碍物（圆形）。radius 为膨胀半径（像素），通常 = 机器狗半径 forbid_distance。
// replan()/replanMultiFloor() 会跳过任意一点到线段距离 <= radius 的 PRM 边。
// floor 指定障碍所在楼层（仅影响该楼层的同楼层边）。
struct DynamicObstacle
{
    BIpoint center;
    double radius;
    int floor = 0;
};

// ========== 全局融合图（离线融合 + 在线插入） ==========

// 全局融合图节点：跨楼层的扁平节点表
struct MapNode
{
    int id;        // 全局唯一 id（= 在 FusedGraph::nodes 中的下标）
    double x, y;
    int floor_id;
};

// 全局融合图边：无向边，只存一份，两端邻接表共享同一下标
struct MapEdge
{
    int from_id, to_id;
    double weight;      // 同层边 = 欧氏距离；跨层边 = 楼梯代价 cost
    bool is_cross_map;  // true = 跨层边（楼梯连接）
};

// 拓扑并集空间多边形：提供射线法 Point-in-Polygon
struct TopoPolygon
{
    int fromFloor;
    int toFloor;
    double cost;                    // 跨层代价（走一层楼梯）
    std::vector<BIpoint> vertices;  // 闭合多边形顶点（像素坐标，首尾相同）

    // 射线法：从 (x,y) 向右发水平射线，与边界交点数奇偶判断内外
    bool contains(double x, double y) const
    {
        int n = (int)vertices.size();
        if (n < 3) return false;
        bool inside = false;
        for (int i = 0, j = n - 1; i < n; j = i++)
        {
            double xi = vertices[i].x, yi = vertices[i].y;
            double xj = vertices[j].x, yj = vertices[j].y;
            if (((yi > y) != (yj > y)) &&
                (x < (xj - xi) * (y - yi) / (yj - yi) + xi))
                inside = !inside;
        }
        return inside;
    }
};

// 全局融合大图：所有楼层 PRM 节点 + 同层边 + 跨层边
struct FusedGraph
{
    std::vector<MapNode> nodes;          // floor f 的节点区间 = [floorNodeStart[f], floorNodeStart[f+1])
    std::vector<int> floorNodeStart;     // 长度 = 楼层数 + 1
    std::vector<MapEdge> edges;          // 所有无向边（每条一份）
    std::vector<std::vector<int>> adj;   // adj[node_id] = 关联边在 edges 中的下标
};

class PRMMultiFloor
{
public:
    PRMMultiFloor();
    ~PRMMultiFloor();

    void initialize(int numFloors);

    // 添加拓扑连接边（两个 BIpoint 为一组，floor 指定各自楼层）
    void addFloorConnection(BIpoint pointFrom, BIpoint pointTo,
                           double cost,
                           FloorConnectionType type = FloorConnectionType::STAIRS);

    void loadFloorGraph(int32_t floorIndex, const BIgraph& graph);
    void setMapReference(int32_t floorIndex, BImap* map);

    // PRM 参数设置（所有楼层统一默认值）
    void setPRMParams(int k_nodes, double r_nei);

    // 按楼层单独设置 PRM 参数，覆盖全局默认值
    void setFloorPRMParams(int floorIdx, int k_nodes, double r_nei);

    // PRM-based 路径规划
    double plan(MultiFloorTask& task);

    // ====== 一次建图 + 多次重规划（用于动态避障仿真） ======
    // 一次性构建并缓存静态 PRM 路网（采样节点 + 边 + 空间网格），不含 start/goal。
    // 成功后后续 replan() 直接复用该图，不再重采样。返回是否成功。
    bool buildRoadmap(int floorIdx);

    // 在已缓存的路网上规划/重规划：虚拟插入 start/goal 临时节点，Dijkstra 跳过被
    // 动态障碍物阻挡的边。返回路径成本（<=0 失败），路径写入 path。不修改缓存路网。
    double replan(int floorIdx, BIpoint start, BIpoint goal,
                  const std::vector<DynamicObstacle>& obstacles,
                  std::list<BIpoint>& path);

    // 多楼层版重规划：复用各层缓存路网（含连接点节点），虚拟插入跨楼层 start/goal，
    // 多楼层 Dijkstra（同楼层边按障碍阻挡 + 跨楼层连接边）。task.start.floor/goal.floor
    // 决定起终点楼层，task.path 输出（点带 floor）。返回成本（<=0 失败）。
    double replanMultiFloor(MultiFloorTask& task,
                            const std::vector<DynamicObstacle>& obstacles);

    // 供仿真层绘制路网 / 做前瞻碰撞检测
    const PRMGraph& getRoadmap(int floorIdx) const;
    bool roadmapBuilt(int floorIdx) const;
    bool pointTraversable(int floorIdx, double x, double y) const;

    void printGraphInfo() const;

    // ====== 离线图融合 + 在线起点插入（多边形跨层，供 sim2d 用） ======
    // 解析 connections.json 得到拓扑并集空间多边形（fromFloor/toFloor/cost/polygon）
    bool loadConnectionsJson(const std::string& path);
    // 离线融合：把各层 PRM 图缝合成全局大图（同层边 + 多边形内跨层边 + 每层 KD-Tree）
    bool fuseGraph();
    // 在融合大图上规划：在线插入 start/goal（多边形感知 KNN），Dijkstra 搜索
    double planFused(MultiFloorTask& task, const std::vector<DynamicObstacle>& obstacles);
    // 融合参数：K = 在线插入的最近邻数；crossRadius = 多边形内跨层连边的距离阈值
    void setFuseParams(int kNeighbors, double crossRadius);
    const FusedGraph& getFusedGraph() const;
    const std::vector<TopoPolygon>& getTopoPolygons() const;
    bool fusedBuilt() const;

private:
    MultiFloorGraph _multiFloorGraph;
    int32_t _numFloors;
    std::vector<BImap*> _mapReferences;

    // PRM data
    std::vector<PRMGraph> _prmGraphs;
    int _prmKNodes;
    double _prmRNei;
    std::vector<int> _prmKNodesOverride;    // 按楼层的 k_nodes 覆盖值，-1 表示不覆盖
    std::vector<double> _prmRNeiOverride;   // 按楼层的 r_nei 覆盖值，<0 表示不覆盖

    // 预计算的可通行性缓存：uint8_t 扁平数组，1=可通行 0=障碍
    // 用于替代 cv::Mat::at<>() 避免 OpenCV 每次边界检查开销
    std::vector<std::vector<uint8_t>> _traversableCache;
    std::vector<int> _cacheWidth;
    std::vector<int> _cacheHeight;

    // ====== 缓存路网（buildRoadmap 一次构建，replan 复用） ======
    std::vector<bool> _roadmapBuilt;                          // 每层是否已建静态路网
    std::vector<std::vector<std::vector<int>>> _roadmapGrid;  // [floor] -> 空间网格 cells
    std::vector<int> _roadmapGridW;                           // [floor] 网格列数
    std::vector<int> _roadmapGridH;                           // [floor] 网格行数

    // ====== 融合图状态 ======
    FusedGraph _fused;                       // 全局融合大图（fuseGraph 构建）
    std::vector<TopoPolygon> _topoPolygons;  // 拓扑并集空间多边形（loadConnectionsJson 解析）
    std::vector<KDTree2D> _fusedKD;          // 每层 KD-Tree（供在线插入 KNN）
    int _fuseKNodes = 10;                    // 在线插入的最近邻数 K
    double _fuseCrossRadius = 8.0;           // 多边形内跨层连边距离阈值
    bool _fusedBuilt = false;

    // PRM methods
    void buildPRMForFloor(int floorIdx, BIpoint start, BIpoint goal);
    bool isPointTraversable(int floorIdx, double x, double y) const;
    bool isSegmentFree(int floorIdx, const BIpoint& a, const BIpoint& b) const;
    // 基于缓存的可通行性数组做线段碰撞检测（buildRoadmap / replan 共用）
    bool segmentFreeCached(int floorIdx, const BIpoint& a, const BIpoint& b) const;

    int32_t findPolygonIndex(int32_t floorIndex, BIpoint point);

    double planInSingleFloor(int32_t floorIndex,
                            BIpoint start, BIpoint goal,
                            int32_t startPoly, int32_t goalPoly,
                            std::list<BIpoint>& path);
};

#endif // __PRM_MULTIFLOOR_H
