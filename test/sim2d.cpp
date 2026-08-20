// sim2d: PRM-3D 跨楼层（多楼层）交互仿真 —— 全局规划 + 动态避障 + 反应式重规划
//
// 规划用 PRMMultiFloor 的「离线图融合 + 在线插入」流程：先 buildRoadmap 构建各层路网，
// 再 loadConnectionsJson + fuseGraph 把各层缝合成全局大图（并集多边形内连跨层边 + 每层
// KD-Tree），之后 planFused() 在线插入起终点（多边形感知 KNN）+ Dijkstra。
// 本文件只负责窗口显示 / 交互逻辑 / 按键处理。
//
// 显示：一个大的窗口横向平铺 5 张楼层地图。机器狗按当前所在楼层渲染在对应 tile 里，
// 跨楼梯时会从一张地图「跳」到另一张地图。鼠标点哪张地图，起终点/障碍就属于那层。
//
// 操作：
//   s + 左键 : 设起点（点在哪层=起点在那层）
//   g + 左键 : 设终点（点在哪层=终点在那层）   设好起终点后机器狗自动沿规划路径动画前进、跨楼梯
//   o + 左键 : 在该层放动态障碍物（红点 + 橙色 forbid 警戒环）
//              当障碍进入机器狗前方前瞻距离并挡路时，重新规划绕开（galileo 走廊细，未必有绕行替代路径）
//   空格     : 暂停 / 继续
//   c        : 清空所有动态障碍物
//   r        : 重置
//   q / ESC  : 退出

#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <cmath>
#include <climits>

#include "PRMmap.h"
#include "PRMmultifloor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// ==================== 配置（照搬 testMultiFloor） ====================
static const int    NUM_FLOORS           = 6;
static const char*  FLOOR_PATHS[] = {
    "/home/dow/DOW/PRM-3D/expMap/35_all/356_1.png",
    "/home/dow/DOW/PRM-3D/expMap/35_all/356_2.png",
    "/home/dow/DOW/PRM-3D/expMap/35_all/356_3.png",
    "/home/dow/DOW/PRM-3D/expMap/35_all/356_4.png",
    "/home/dow/DOW/PRM-3D/expMap/35_all/356_5.png",
    "/home/dow/DOW/PRM-3D/expMap/35_all/356_6.png"
};
static const double EXPANSION_RADIUS     = 4.0;    // 静态障碍膨胀
static const double FORBID_RADIUS        = 4.0;    // 动态障碍膨胀半径 = 机器狗 keepout
static const double LOOKAHEAD_PX         = 120.0;  // 反应式重规划前瞻距离
static const double SPEED_PX_PER_FRAME   = 2.0;    // 动画速度（像素/帧）
static const int    FRAME_DELAY_MS       = 20;

// 平铺布局
static const int    TILE_W               = 440;    // 每层 tile 显示宽（像素）
static const int    GAP                  = 12;     // tile 间距
static const int    MARGIN               = 12;     // 画布边距
static const int    LABEL_H              = 24;     // 顶部楼层标签条高度
static const int    HUD_H                = 64;     // 底部 HUD 条高度

struct FloorPRM { int k; double r; };
// 节点数 k 控制路网密度：galileo 自由区面积小且非凸，采样全部自由像素会让边爆炸、规划变慢；
// 取 k≈500（自由像素的子集）+ r=25 半径建边（短直线穿过迷宫通道），既连通又快（跨楼层规划~75ms）。
static const FloorPRM FLOOR_PRM[NUM_FLOORS] = {
    {500, 25.0}, {500, 25.0}, {500, 25.0}, {500, 25.0}, {500, 25.0}, {500, 25.0}
};

// 拓扑并集空间（多边形）由 mark_connections.py 生成，规划器从 JSON 解析后做离线融合。
// 不再手工维护连接点：相邻楼层两图都可行的重叠区域 = 跨层过渡区。
static const char* CONNECTIONS_JSON = "/home/dow/DOW/PRM-3D/connections35_all.json";
// ====================================================================

struct Obstacle { BIpoint center; double radius; int floor; };

struct Sim
{
    PRMMultiFloor planner;
    std::vector<BImap> floorMaps;
    std::vector<cv::Mat> cleanBIimap;                 // 每层静态障碍 BIimap 副本（烘焙前复位用）
    std::vector<cv::Mat> baseImg;                     // 每层 BIdmap（渲染底图）
    std::vector<std::vector<std::vector<cv::Point>>> floorPolys;  // 每层并集空间多边形轮廓（画黄线）
    int imgW = 0, imgH = 0;                           // 原图尺寸（galileo 各层相同）
    int tileH = 0;                                    // 单层 tile 显示高
    double disp2img = 1.0;                            // 显示 tile 像素 -> 原图像素

    std::string mode = "s";
    bool hasStart = false, hasGoal = false;
    BIpoint start{-1, -1, -1}, goal{-1, -1, -1};
    std::vector<Obstacle> obstacles;

    std::vector<BIpoint> pathV;
    size_t segIdx = 0;
    BIpoint robotPos{-1, -1, -1};
    bool running = false, paused = false, arrived = false, blocked = false;
    int replanCount = 0;
    std::vector<BIpoint> trail;
};

// ---- 工具 ----
static std::vector<BIpoint> listToVec(const std::list<BIpoint>& l) { return {l.begin(), l.end()}; }

// 把 (x,y) 在该层 cleanBIimap 上 snap 到最近的自由像素（galileo 细走廊可用性）
static void snapToFree(Sim& sim, int floor, double& x, double& y)
{
    if (floor < 0 || floor >= NUM_FLOORS) return;
    const cv::Mat& bi = sim.cleanBIimap[floor];
    int W = bi.cols, H = bi.rows;
    int ix = (int)x, iy = (int)y;
    auto isFree = [&](int xx, int yy) -> bool {
        if (xx < 0 || xx >= W || yy < 0 || yy >= H) return false;
        return (bi.at<uint16_t>(yy, xx) & 0xC000) == 0;
    };
    if (isFree(ix, iy)) return;
    for (int r = 1; r <= 50; ++r)
    {
        int bestD = INT_MAX, bx = ix, by = iy; bool found = false;
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx)
            {
                if (std::max(std::abs(dx), std::abs(dy)) != r) continue;  // 只查半径 r 的外环
                if (isFree(ix + dx, iy + dy))
                {
                    int d = dx * dx + dy * dy;
                    if (d < bestD) { bestD = d; bx = ix + dx; by = iy + dy; found = true; }
                }
            }
        if (found) { x = bx; y = by; return; }
    }
}

// 把本仿真层的障碍转成库的 DynamicObstacle（带 floor），
// 供 planFused 在 Dijkstra 里按「点到边距离」判阻挡（不再烘焙 BIimap）
static std::vector<DynamicObstacle> buildDynObstacles(const Sim& sim)
{
    std::vector<DynamicObstacle> dyn;
    dyn.reserve(sim.obstacles.size());
    for (const auto& obs : sim.obstacles)
        dyn.push_back({obs.center, obs.radius, obs.floor});
    return dyn;
}

// 跨楼层前瞻：从 robotPos 沿 pathV 向前看 lookaheadPx，任一同楼层段被该层障碍挡住返回 true
static bool pathBlockedAhead(const Sim& sim, double lookahead)
{
    const auto& pv = sim.pathV;
    if (pv.size() < 2 || sim.segIdx + 1 >= pv.size()) return false;
    BIpoint a = sim.robotPos;
    double acc = 0.0;
    for (size_t i = sim.segIdx; i + 1 < pv.size() && acc < lookahead; ++i)
    {
        BIpoint b = pv[i + 1];
        if (a.floor == b.floor)
        {
            for (const auto& obs : sim.obstacles)
                if (obs.floor == a.floor && Point2LineDistance(obs.center, a, b) <= obs.radius)
                    return true;
            acc += a % b;
        }
        a = b;
    }
    return false;
}

static void advanceRobot(Sim& sim, double step)
{
    auto& pv = sim.pathV;
    double rem = step;
    while (rem > 0 && sim.segIdx + 1 < pv.size())
    {
        BIpoint b = pv[sim.segIdx + 1];
        if (sim.robotPos.floor != b.floor) { sim.robotPos = b; sim.segIdx++; continue; }  // 楼梯跳跃
        double d = sim.robotPos % b;
        if (d <= rem) { sim.robotPos = b; sim.segIdx++; rem -= d; }
        else if (d > 1e-9)
        {
            int fl = sim.robotPos.floor;  // 注意：BIpoint 的 +-*/ 不保留 floor 字段，需手动存回
            sim.robotPos = sim.robotPos + (b - sim.robotPos) * (rem / d);
            sim.robotPos.floor = fl;
            rem = 0;
        }
        else { sim.segIdx++; }
    }
    if (sim.segIdx + 1 >= pv.size()) { sim.arrived = true; sim.running = false; }
}

// 鼠标：点哪张地图(tile) = 哪层；坐标换算回该层原图像素
static void onMouse(int event, int x, int y, int /*flags*/, void* userdata)
{
    if (event != cv::EVENT_LBUTTONDOWN) return;
    Sim* sim = static_cast<Sim*>(userdata);
    int tx = x - MARGIN;
    if (tx < 0) return;
    int f = tx / (TILE_W + GAP);
    int withinX = tx - f * (TILE_W + GAP);
    if (f < 0 || f >= NUM_FLOORS || withinX > TILE_W) return;   // 落在 gap 或越界
    int ty = y - (MARGIN + LABEL_H);
    if (ty < 0 || ty > sim->tileH) return;

    double ix = withinX * sim->disp2img;
    double iy = ty * sim->disp2img;
    snapToFree(*sim, f, ix, iy);

    if (sim->mode == "s")
    {
        sim->start = {ix, iy, f}; sim->hasStart = true;
        sim->pathV.clear(); sim->running = sim->arrived = sim->blocked = false;
        printf("[start] F%d (%.0f,%.0f)\n", f, ix, iy);
    }
    else if (sim->mode == "g")
    {
        sim->goal = {ix, iy, f}; sim->hasGoal = true;
        sim->pathV.clear(); sim->running = sim->arrived = sim->blocked = false;
        printf("[goal]  F%d (%.0f,%.0f)\n", f, ix, iy);
    }
    else if (sim->mode == "o")
    {
        sim->obstacles.push_back({{ix, iy, f}, FORBID_RADIUS, f});
        printf("[obstacle] #%zu F%d (%.0f,%.0f) forbid=%.0fpx\n",
               sim->obstacles.size(), f, ix, iy, FORBID_RADIUS);
    }
}

// 画某一层的 scene（原图分辨率，不含标签/HUD —— 这些在画布上统一画）
static void drawFloor(const Sim& sim, int f, cv::Mat& scene)
{
    scene = sim.baseImg[f].clone();
    auto Ptx = [](const BIpoint& p) { return cv::Point((int)p.x, (int)p.y); };
    const auto& pv = sim.pathV;

    // 路径：该层同楼层段；已走过=暗色细线，剩余=亮青粗线
    for (size_t i = 0; i + 1 < pv.size(); ++i)
    {
        if (pv[i].floor != f || pv[i + 1].floor != f) continue;
        bool traveled = (i + 1) <= sim.segIdx;
        cv::line(scene, Ptx(pv[i]), Ptx(pv[i + 1]),
                 traveled ? cv::Scalar(90, 60, 0) : cv::Scalar(255, 255, 0),
                 traveled ? 1 : 2);
    }
    if ((sim.running || sim.arrived) && sim.robotPos.floor == f &&
        sim.segIdx + 1 < pv.size() && pv[sim.segIdx + 1].floor == f)
        cv::line(scene, Ptx(sim.robotPos), Ptx(pv[sim.segIdx + 1]), cv::Scalar(255, 255, 0), 2);

    // 拓扑并集空间多边形轮廓（黄，跨层过渡区）
    for (const auto& poly : sim.floorPolys[f])
        if (poly.size() >= 2)
            cv::polylines(scene, poly, true, cv::Scalar(0, 255, 255), 1);

    // 障碍物（红点 + forbid 环）
    for (const auto& obs : sim.obstacles)
        if (obs.floor == f)
        {
            cv::circle(scene, Ptx(obs.center), 4, cv::Scalar(0, 0, 255), -1);
            cv::circle(scene, Ptx(obs.center), (int)std::max(4.0, obs.radius), cv::Scalar(0, 128, 255), 1);
        }

    // 尾迹（本层）
    for (size_t i = 0; i < sim.trail.size(); ++i)
        if (sim.trail[i].floor == f)
        {
            int a = 80 + (int)(160.0 * i / std::max((size_t)1, sim.trail.size()));
            cv::circle(scene, Ptx(sim.trail[i]), 1, cv::Scalar(a, 0, 0), -1);
        }

    // 起终点
    if (sim.hasStart && sim.start.floor == f)
    {
        cv::circle(scene, Ptx(sim.start), 5, cv::Scalar(0, 255, 0), -1);
        cv::putText(scene, "S", cv::Point(Ptx(sim.start).x + 8, Ptx(sim.start).y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
    if (sim.hasGoal && sim.goal.floor == f)
    {
        cv::circle(scene, Ptx(sim.goal), 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(scene, "G", cv::Point(Ptx(sim.goal).x + 8, Ptx(sim.goal).y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }

    // 机器狗（仅当当前在本层）
    if ((sim.running || sim.arrived) && sim.robotPos.floor == f)
    {
        cv::circle(scene, Ptx(sim.robotPos), (int)std::max(5.0, FORBID_RADIUS), cv::Scalar(255, 128, 0), 1);
        cv::circle(scene, Ptx(sim.robotPos), 5, cv::Scalar(255, 0, 0), -1);
    }
}

int main()
{
    printf("=== PRM-3D 多楼层仿真：规划 + 避障 + 重规划 (galileo x%d) ===\n", NUM_FLOORS);
    printf("forbid=%.0fpx  lookahead=%.0fpx  speed=%.1fpx/frame\n\n", FORBID_RADIUS, LOOKAHEAD_PX, SPEED_PX_PER_FRAME);

    Sim sim;
    sim.floorMaps.resize(NUM_FLOORS);
    sim.cleanBIimap.resize(NUM_FLOORS);
    sim.baseImg.resize(NUM_FLOORS);
    sim.floorPolys.resize(NUM_FLOORS);

    // ---- 1. 逐层建图（testMultiFloor 配方）----
    for (int i = 0; i < NUM_FLOORS; ++i)
    {
        cv::Mat img = cv::imread(FLOOR_PATHS[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) { printf("无法加载 %s\n", FLOOR_PATHS[i]); return -1; }

        BImap& m = sim.floorMaps[i];
        m.shapeX = img.cols; m.shapeY = img.rows;
        m.mapratio = 0.1; m.robotsize = 0.9;

        cv::Mat obstacleMask = (img <= 128);
        int ksz = (int)(2.0 * EXPANSION_RADIUS + 1.0);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksz, ksz));
        cv::Mat dilated;
        cv::dilate(obstacleMask, dilated, kernel);
        m.BIimap = cv::Mat::zeros(img.rows, img.cols, CV_16U);
        m.BIimap.setTo(cv::Scalar(0xC000), dilated);

        cv::cvtColor(img, m.BIdmap, cv::COLOR_GRAY2BGR);
        sim.cleanBIimap[i] = m.BIimap.clone();
        sim.baseImg[i] = m.BIdmap.clone();
    }
    sim.imgW = sim.baseImg[0].cols;
    sim.imgH = sim.baseImg[0].rows;
    sim.tileH = (int)(TILE_W * (double)sim.imgH / sim.imgW);
    sim.disp2img = (double)sim.imgW / TILE_W;

    // ---- 2. 规划器配置 ----
    sim.planner.initialize(NUM_FLOORS);
    sim.planner.setPRMParams(5000, 50.0);
    for (int i = 0; i < NUM_FLOORS; ++i)
        sim.planner.setFloorPRMParams(i, FLOOR_PRM[i].k, FLOOR_PRM[i].r);

    BIgraph emptyGraph;
    for (int i = 0; i < NUM_FLOORS; ++i)
    {
        sim.planner.loadFloorGraph(i, emptyGraph);
        sim.planner.setMapReference(i, &sim.floorMaps[i]);
    }

    // 一次性构建并缓存各层 PRM 路网（采样节点 + 同层边）。
    printf("Building roadmaps (one-time)...\n");
    for (int i = 0; i < NUM_FLOORS; ++i)
        sim.planner.buildRoadmap(i);
    printf("Roadmaps ready.\n");

    // 离线图融合：解析 connections.json 的并集多边形，把各层路网缝合成全局大图
    // （同层边 + 多边形内跨层边 + 每层 KD-Tree）。之后所有规划/重规划走 planFused。
    if (!sim.planner.loadConnectionsJson(CONNECTIONS_JSON))
    {
        printf("加载连接配置失败: %s\n", CONNECTIONS_JSON);
        return -1;
    }
    sim.planner.fuseGraph();

    // 把每层的并集多边形轮廓取出来供渲染（画黄线）
    for (const TopoPolygon& tp : sim.planner.getTopoPolygons())
    {
        std::vector<cv::Point> outline;
        outline.reserve(tp.vertices.size());
        for (const BIpoint& v : tp.vertices)
            outline.push_back(cv::Point((int)v.x, (int)v.y));
        if (tp.fromFloor >= 0 && tp.fromFloor < NUM_FLOORS)
            sim.floorPolys[tp.fromFloor].push_back(outline);
        if (tp.toFloor >= 0 && tp.toFloor < NUM_FLOORS)
            sim.floorPolys[tp.toFloor].push_back(outline);
    }
    printf("Fusion ready.\n\n");

    // ---- 3. 单一大窗口（横向平铺 5 层）----
    const char* WIN = "PRM-3D sim2d (multi-floor)";
    int canvasW = MARGIN + NUM_FLOORS * TILE_W + (NUM_FLOORS - 1) * GAP + MARGIN;
    int canvasH = MARGIN + LABEL_H + sim.tileH + HUD_H + MARGIN;
    std::vector<cv::Point> tileOrigin(NUM_FLOORS);
    for (int f = 0; f < NUM_FLOORS; ++f)
        tileOrigin[f] = cv::Point(MARGIN + f * (TILE_W + GAP), MARGIN + LABEL_H);

    cv::namedWindow(WIN, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(WIN, onMouse, &sim);

    printf("操作：s/g/o + 左键点击（点哪张地图=哪层）；空格暂停；c 清障碍；r 重置；q 退出\n\n");

    // ---- 4. 主循环 ----
    while (true)
    {
        // (a) 触发初始规划（在缓存路网上重搜，毫秒级）
        if (sim.hasStart && sim.hasGoal && sim.pathV.empty())
        {
            MultiFloorTask task;
            task.start = sim.start; task.goal = sim.goal;
            double cost = sim.planner.planFused(task, buildDynObstacles(sim));
            if (cost > 0)
            {
                sim.pathV = listToVec(task.path);
                sim.segIdx = 0;
                sim.robotPos = sim.pathV.front();
                sim.running = true; sim.arrived = false; sim.blocked = false;
                sim.trail.clear(); sim.trail.push_back(sim.robotPos);
            }
            else printf("初始规划失败，请检查起终点/连接\n");
        }

        // (b) 反应式重规划 + 动画推进
        if (sim.running && !sim.paused && !sim.arrived)
        {
            if (pathBlockedAhead(sim, LOOKAHEAD_PX))
            {
                MultiFloorTask task;
                task.start = sim.robotPos; task.goal = sim.goal;
                double cost = sim.planner.planFused(task, buildDynObstacles(sim));
                if (cost > 0)
                {
                    sim.pathV = listToVec(task.path);
                    sim.segIdx = 0;
                    sim.robotPos = sim.pathV.front();
                    sim.replanCount++; sim.blocked = false;
                    printf(">> 触发重规划 #%d\n", sim.replanCount);
                }
                else sim.blocked = true;
            }
            if (!sim.blocked)
            {
                advanceRobot(sim, SPEED_PX_PER_FRAME);
                sim.trail.push_back(sim.robotPos);
                if (sim.trail.size() > 800) sim.trail.erase(sim.trail.begin());
                if (sim.arrived) printf(">> 到达终点\n");
            }
        }

        // (c) 渲染：逐层 scene 缩放贴入大画布
        cv::Mat canvas(canvasH, canvasW, CV_8UC3, cv::Scalar(28, 28, 28));
        for (int f = 0; f < NUM_FLOORS; ++f)
        {
            cv::Mat scene, tile;
            drawFloor(sim, f, scene);
            cv::resize(scene, tile, cv::Size(TILE_W, sim.tileH), 0, 0, cv::INTER_AREA);
            tile.copyTo(canvas(cv::Rect(tileOrigin[f], cv::Size(TILE_W, sim.tileH))));
            // 楼层标签
            std::string lbl = "Floor " + std::to_string(f);
            cv::putText(canvas, lbl, cv::Point(tileOrigin[f].x + 4, MARGIN + LABEL_H - 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3);
            cv::putText(canvas, lbl, cv::Point(tileOrigin[f].x + 4, MARGIN + LABEL_H - 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
        }

        // (d) HUD（画布底部）
        std::string status = sim.arrived ? "ARRIVED" :
                             sim.blocked ? "BLOCKED (no detour)" :
                             sim.paused  ? "PAUSED" :
                             sim.running ? "RUNNING" : "IDLE";
        std::vector<std::string> hud = {
            "mode: [" + sim.mode + "]  (s/g/o + click on a floor map)",
            "status: " + status + "   replans: " + std::to_string(sim.replanCount) +
                "   obstacles: " + std::to_string(sim.obstacles.size()),
            "space=pause  c=clear obs  r=reset  q=quit",
        };
        int hudY = MARGIN + LABEL_H + sim.tileH + 6;
        for (size_t i = 0; i < hud.size(); ++i)
        {
            cv::putText(canvas, hud[i], cv::Point(MARGIN, hudY + (int)i * 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3);
            cv::putText(canvas, hud[i], cv::Point(MARGIN, hudY + (int)i * 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(235, 235, 235), 1);
        }

        cv::imshow(WIN, canvas);

        // (e) 按键
        int key = cv::waitKey(FRAME_DELAY_MS) & 0xFF;
        if (key == 27 || key == 'q') break;
        else if (key == 's') { sim.mode = "s"; printf("mode -> start\n"); }
        else if (key == 'g') { sim.mode = "g"; printf("mode -> goal\n"); }
        else if (key == 'o') { sim.mode = "o"; printf("mode -> obstacle\n"); }
        else if (key == ' ') { sim.paused = !sim.paused; printf(sim.paused ? "paused\n" : "resumed\n"); }
        else if (key == 'c') { sim.obstacles.clear(); printf("cleared obstacles\n"); }
        else if (key == 'r')
        {
            sim.pathV.clear(); sim.obstacles.clear(); sim.trail.clear();
            sim.hasStart = sim.hasGoal = false;
            sim.running = sim.arrived = sim.blocked = false;
            sim.replanCount = 0;
            printf("reset\n");
        }
    }

    cv::destroyAllWindows();
    return 0;
}
