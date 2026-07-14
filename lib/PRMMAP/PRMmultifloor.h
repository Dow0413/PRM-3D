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

    void printGraphInfo() const;

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

    // PRM methods
    void buildPRMForFloor(int floorIdx, BIpoint start, BIpoint goal);
    bool isPointTraversable(int floorIdx, double x, double y) const;
    bool isSegmentFree(int floorIdx, const BIpoint& a, const BIpoint& b) const;

    int32_t findPolygonIndex(int32_t floorIndex, BIpoint point);

    double planInSingleFloor(int32_t floorIndex,
                            BIpoint start, BIpoint goal,
                            int32_t startPoly, int32_t goalPoly,
                            std::list<BIpoint>& path);
};

#endif // __PRM_MULTIFLOOR_H
