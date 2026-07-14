#include <iostream>
#include <vector>
#include <cmath>

#include "PRMmap.h"
#include "PRMmultifloor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// 在指定图像上绘制路径
void drawPath(const std::list<BIpoint>& path, cv::Mat& image, cv::Scalar color = cv::Scalar(255, 255, 0), int thickness = 3)
{
    if (path.size() < 2) return;
    auto it = path.begin();
    cv::Point pt1((int)it->x, (int)it->y);
    ++it;
    for (; it != path.end(); ++it)
    {
        cv::Point pt2((int)it->x, (int)it->y);
        cv::line(image, pt1, pt2, color, thickness);
        pt1 = pt2;
    }
}

int main()
{
    printf("=== PRM Multi-Floor Path Planning ===\r\n\r\n");

    PRMMultiFloor planner;
    const int numFloors = 5;
    planner.initialize(numFloors);

    // 设置全局默认 PRM 参数（所有楼层通用）
    planner.setPRMParams(5000, 50.0);

    // 按楼层覆盖：大图（或重要楼层）用更多采样点 & 更密半径
    // 小图/不重要楼层用更少节点
    planner.setFloorPRMParams(0, 15000, 25.0);
    planner.setFloorPRMParams(1, 3000, 25.0);
    planner.setFloorPRMParams(2, 3000, 25.0);
    planner.setFloorPRMParams(3, 3000, 25.0);
    planner.setFloorPRMParams(4, 2000, 25.0);

    // planner.setFloorPRMParams(0, 10000, 20.0);
    // planner.setFloorPRMParams(1, 8000, 15.0);


    std::vector<BImap> floorMaps(numFloors);
    std::vector<cv::Mat> floorImages(numFloors);

    const char* floorPaths[] = {
        "/home/dow/DOW/PRM-3D/expMap/galileo_1.png",
        "/home/dow/DOW/PRM-3D/expMap/galileo_2.png",
        "/home/dow/DOW/PRM-3D/expMap/galileo_3.png",
        "/home/dow/DOW/PRM-3D/expMap/galileo_4.png",
        "/home/dow/DOW/PRM-3D/expMap/galileo_5.png"
    };

    // const char* floorPaths[] = {
    //     "/home/dow/DOW/PRM-3D/expMap/test_1.png",
    //     "/home/dow/DOW/PRM-3D/expMap/test_2.png"
    // };

    // 障碍物膨胀半径（像素），考虑机器狗体积
    const double Expansion_radius = 2.0;

    printf("Loading floor maps (direct PNG, no PRM server)...\r\n");
    for (int i = 0; i < numFloors; i++)
    {
        printf("  Floor %d: %s\r\n", i, floorPaths[i]);

        // 直接加载PNG地图，不依赖PRM服务器
        cv::Mat img = cv::imread(floorPaths[i], cv::IMREAD_GRAYSCALE);
        if (img.empty())
        {
            printf("error: cannot load %s!!\r\n", floorPaths[i]);
            return -1;
        }

        BImap& m = floorMaps[i];
        m.shapeX = img.cols;
        m.shapeY = img.rows;
        m.mapratio = 0.1;
        m.robotsize = 0.9;

        // 创建 BIimap (CV_16U): 可行走=0x0000, 障碍=0xC000
        // 白色(>128) = 可行走, 黑色(<=128) = 障碍
        m.BIimap = cv::Mat::zeros(img.rows, img.cols, CV_16U);
        cv::Mat obstacleMask = (img <= 128);
        cv::Mat expansionZone;  // 膨胀新增的障碍区域

        // 障碍物膨胀：黑色区域向外扩张 Expansion_radius 像素
        if (Expansion_radius > 0)
        {
            int kernelSize = (int)(2.0 * Expansion_radius + 1.0);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                        cv::Size(kernelSize, kernelSize));
            cv::Mat dilatedMask;
            cv::dilate(obstacleMask, dilatedMask, kernel);
            m.BIimap.setTo(cv::Scalar(0xC000), dilatedMask);

            // 膨胀新增的障碍区域 = 膨胀后障碍 & 非原始障碍
            cv::bitwise_and(dilatedMask, ~obstacleMask, expansionZone);
            printf("    obstacles dilated by %.1f px (kernel=%d)\r\n",
                   Expansion_radius, kernelSize);
        }
        else
        {
            m.BIimap.setTo(cv::Scalar(0xC000), obstacleMask);
        }

        // 创建 BIdmap 用于显示 (CV_8UC3)
        cv::cvtColor(img, m.BIdmap, cv::COLOR_GRAY2BGR);

        // 膨胀区域用红色标注
        if (Expansion_radius > 0 && cv::countNonZero(expansionZone) > 0)
        {
            m.BIdmap.setTo(cv::Scalar(0, 0, 255), expansionZone);  // BGR: 红色
        }

        // 创建空的 BIgraphList 用于兼容
        BIgraph emptyGraph;
        m.BIgraphList.push_back(emptyGraph);

        planner.loadFloorGraph(i, m.BIgraphList[0]);
        planner.setMapReference(i, &m);
    }

    planner.printGraphInfo();

    // 生成调试图像（已含红色膨胀标注）
    for (int i = 0; i < numFloors; i++)
    {
        floorImages[i] = floorMaps[i].BIdmap.clone();
    }

    // 拓扑连接边：{from, to, cost}
    struct ConnDef { BIpoint from, to; double cost; };

    std::vector<ConnDef> connectionPoints = {
        {{552.0, 204.0, 0}, {552.0, 204.0, 1}, 15},
        {{545.0, 231.0, 1}, {545.0, 231.0, 2}, 15},
        {{544.0, 197.0, 2}, {544.0, 197.0, 3}, 15},
        {{545.0, 232.0, 3}, {545.0, 232.0, 4}, 15}
    };

    // std::vector<ConnDef> connectionPoints = {
    //     {{318.0, 60.0, 0}, {318.0, 60.0, 1}, 15}
    // };

    // 起点和终点
    BIpoint start = {600.0, 419.0, 0};
    BIpoint goal  = {458.0, 204.0, 3};

    // BIpoint start = {118.2, 410.1, 0};
    // BIpoint goal  = {509.7, 534.1, 0};

    // ====== 验证起点/终点/连接点是否在膨胀后障碍物内，并在图上红色 X 标注 ======
    int invalidCount = 0;

    // 检查起点
    {
        uint16_t val = floorMaps[start.floor].BIimap.at<uint16_t>((int)start.y, (int)start.x);
        if ((val >> 14) & 0x3)
        {
            printf("ERROR: start (%.1f, %.1f) floor=%d is in dilated obstacle zone!!\r\n",
                   start.x, start.y, start.floor);
            cv::drawMarker(floorImages[start.floor], cv::Point((int)start.x, (int)start.y),
                           cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, 14, 2);
            invalidCount++;
        }
    }

    // 检查终点
    {
        uint16_t val = floorMaps[goal.floor].BIimap.at<uint16_t>((int)goal.y, (int)goal.x);
        if ((val >> 14) & 0x3)
        {
            printf("ERROR: goal (%.1f, %.1f) floor=%d is in dilated obstacle zone!!\r\n",
                   goal.x, goal.y, goal.floor);
            cv::drawMarker(floorImages[goal.floor], cv::Point((int)goal.x, (int)goal.y),
                           cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, 14, 2);
            invalidCount++;
        }
    }

    // 检查连接点
    for (const auto& cp : connectionPoints)
    {
        for (int j = 0; j < 2; j++)
        {
            const BIpoint& pt = (j == 0) ? cp.from : cp.to;
            BImap& fm = floorMaps[pt.floor];
            if (pt.x < 0 || pt.x >= fm.shapeX || pt.y < 0 || pt.y >= fm.shapeY)
            {
                printf("WARNING: connection point%s (%.1f, %.1f) floor=%d out of map bounds!!\r\n",
                       (j == 0) ? ".from" : ".to", pt.x, pt.y, pt.floor);
                invalidCount++;
                continue;
            }
            uint16_t val = fm.BIimap.at<uint16_t>((int)pt.y, (int)pt.x);
            if ((val >> 14) & 0x3)
            {
                printf("WARNING: connection point%s (%.1f, %.1f) floor=%d is in dilated obstacle zone!!\r\n",
                       (j == 0) ? ".from" : ".to", pt.x, pt.y, pt.floor);
                invalidCount++;
                // 在 floorImages 上画红色 X 标记无效连接点
                cv::drawMarker(floorImages[pt.floor], cv::Point((int)pt.x, (int)pt.y),
                               cv::Scalar(0, 0, 255), cv::MARKER_TILTED_CROSS, 12, 2);
            }
        }
    }

    if (invalidCount > 0)
        printf("  => %d point(s) invalid (in dilated obstacle), marked with red X on preview!\r\n", invalidCount);

    // ====== 膨胀地图预览（红色区域=膨胀新增障碍，红色 X=无效点） ======
    if (Expansion_radius > 0)
    {
        printf("\r\nExpanded map preview: red fill = dilated obstacle, red X = invalid point\r\n");
        for (int i = 0; i < numFloors; i++)
        {
            cv::namedWindow("Expanded Floor " + std::to_string(i), 0);
            cv::resizeWindow("Expanded Floor " + std::to_string(i), 800, 800);
            cv::imshow("Expanded Floor " + std::to_string(i), floorImages[i]);
        }
        printf("Press any key to continue planning...\r\n");
        cv::waitKey(0);
        for (int i = 0; i < numFloors; i++)
            cv::destroyWindow("Expanded Floor " + std::to_string(i));
    }

    if (invalidCount > 0)
    {
        printf("ERROR: %d point(s) in obstacle zone, please adjust coordinates!\r\n", invalidCount);
        return -1;
    }

    // 注册连接点到规划器
    for (const auto& cp : connectionPoints)
    {
        planner.addFloorConnection(cp.from, cp.to, cp.cost);
    }

    // 查找起点终点所在多边形
    int32_t startPoly = -1, goalPoly = -1;
    {
        uint16_t val = floorMaps[start.floor].BIimap.at<uint16_t>((int)start.y, (int)start.x);
        startPoly = val & 0x3FFF;
    }
    {
        uint16_t val = floorMaps[goal.floor].BIimap.at<uint16_t>((int)goal.y, (int)goal.x);
        goalPoly = val & 0x3FFF;
    }

    printf("Start: (%.1f, %.1f) floor=%d, poly=%d\r\n", start.x, start.y, start.floor, startPoly);
    printf("Goal:  (%.1f, %.1f) floor=%d, poly=%d\r\n", goal.x, goal.y, goal.floor, goalPoly);

    MultiFloorTask task;
    task.start = start;
    task.goal = goal;
    task.startPoly = startPoly;
    task.goalPoly = goalPoly;

    printf("\r\nPlanning...\r\n");
    int64_t t0 = utime_ns();
    double cost = planner.plan(task);
    int64_t t1 = utime_ns();

    // 绘制连接点（黄色）
    for (const auto& cp : connectionPoints)
    {
        int f1 = cp.from.floor;
        int f2 = cp.to.floor;
        if (f1 >= 0 && f1 < numFloors)
            cv::circle(floorImages[f1], cv::Point((int)cp.from.x, (int)cp.from.y), 2, cv::Scalar(0, 255, 255), -1);
        if (f2 >= 0 && f2 < numFloors)
            cv::circle(floorImages[f2], cv::Point((int)cp.to.x, (int)cp.to.y), 2, cv::Scalar(0, 255, 255), -1);
    }

    // 绘制起点（绿色）和终点（红色）
    cv::circle(floorImages[start.floor], cv::Point((int)start.x, (int)start.y), 3, cv::Scalar(0, 255, 0), -1);
    cv::putText(floorImages[start.floor], "S", cv::Point((int)start.x + 12, (int)start.y - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

    cv::circle(floorImages[goal.floor], cv::Point((int)goal.x, (int)goal.y), 3, cv::Scalar(0, 0, 255), -1);
    cv::putText(floorImages[goal.floor], "G", cv::Point((int)goal.x + 12, (int)goal.y - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

    if (cost > 0)
    {
        printf("\r\n=== SUCCESS ===\r\n");
        printf("Cost: %.2f\r\n", cost);
        printf("Time: %.2f ms\r\n", (t1 - t0) / 1e6);
        printf("Path points: %zu\r\n", task.path.size());

        // 按 floor 分段绘制
        std::list<BIpoint> currentSegment;
        int currentFloor = -1;
        for (const auto& pt : task.path)
        {
            if (currentSegment.empty() || pt.floor == currentFloor)
            {
                currentSegment.push_back(pt);
                currentFloor = pt.floor;
            }
            else
            {
                if (currentFloor >= 0 && currentFloor < numFloors)
                    drawPath(currentSegment, floorImages[currentFloor], cv::Scalar(0, 0, 255), 3);
                currentSegment.clear();
                currentSegment.push_back(pt);
                currentFloor = pt.floor;
            }
        }
        if (!currentSegment.empty() && currentFloor >= 0 && currentFloor < numFloors)
        {
            drawPath(currentSegment, floorImages[currentFloor], cv::Scalar(0, 0, 255), 3);
        }

        for (const auto& pt : task.path)
        {
            printf("  (%.1f, %.1f) floor=%d\r\n", pt.x, pt.y, pt.floor);
        }
    }
    else
    {
        printf("\r\n=== FAILED ===\r\n");
    }

    // 显示 3 个窗口
    for (int i = 0; i < numFloors; i++)
    {
        cv::namedWindow("Floor " + std::to_string(i), 0);
        cv::resizeWindow("Floor " + std::to_string(i), 800, 800);
        cv::imshow("Floor " + std::to_string(i), floorImages[i]);
    }

    printf("\r\nPress any key to exit.\r\n");
    cv::waitKey(0);

    return 0;
}
