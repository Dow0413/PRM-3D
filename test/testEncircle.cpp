#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "CDTmap.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// 回调函数，当鼠标左键按下时调用
int32_t updata = false;
BIpoint mousePoint = {-1, -1};
void onMouse(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        updata = true;
        mousePoint.x = x;
        mousePoint.y = y;
        // 在控制台输出鼠标左击位置的坐标
        std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }
}

int main()
{
    BImap cemap;

    // 创建窗口并注册鼠标回调函数
    cv::namedWindow("Video", 0);
    cv::resizeWindow("Video", 1200, 1200);
    cv::setMouseCallback("Video", onMouse);
    int64_t t0 = utime_ns();
    cemap.MaptoBInavi((char *)"/home/tzyr/WorkSpace/Encircle/CDT-Encircle/expMap/GAME.png", 0.1, 0.9);
    int64_t t1 = utime_ns();
    std::cout << "Itime(ns): " << (t1 - t0) << std::endl;
    cemap.DrawBImap(true);
    cv::Mat image = cemap.BIdmap.clone();
    uint32_t PolyIndex_p = 9;
    while (1)
    {
        int32_t key = cv::waitKey(60);

        if (updata)
        {
            updata = false;

            image = cemap.BIdmap.clone();
            uint32_t anum_p = cemap.BIamap.at<uint16_t>(mousePoint.y, mousePoint.x);
            PolyIndex_p = cemap.BIimap.at<uint16_t>(mousePoint.y, mousePoint.x);
            std::cout << "anum_p: " << anum_p << std::endl;
            std::cout << "PolyIndex_p: " << PolyIndex_p << std::endl;
            std::cout << "PolyIndex_p & 0x3FFF: " << (PolyIndex_p & 0x3FFF) << std::endl;
            if (anum_p != 0xFFFF && ((PolyIndex_p & 0xC000) != 0))
            {
                int32_t obs_index = PolyIndex_p & 0x3FFF;
                BIgraph &graph = cemap.BIgraphList[anum_p];
                std::vector<BIfreepolygon> &freepolygonList = graph.freepolygonList;
                BIobspolygon &obspolygon = graph.obspolygonList[obs_index];
                printf("polygonRlist: \r\n");
                for (int32_t freeIndex : obspolygon.polygonRlist)
                {
                    BIfreepolygon &polygon = freepolygonList[freeIndex];
                    std::stringstream ss;
                    ss << polygon.index;
                    std::string text = ss.str();
                    cv::putText(image, text, cv::Point(int(polygon.core.x), int(polygon.core.y)), cv::FONT_HERSHEY_COMPLEX,
                                1, cv::Scalar(255, 0, 0), 2);
                    printf("%3d, ", freeIndex);
                }
                printf("\r\n");
            }
        }
        if (key == 27)
            break;
        cv::imshow("Video", image);
    }

    EncirclePtask task;
    task.x_s = {456, 1375};
    task.x_g = {1080, 1901};
    task.obs_goal = PolyIndex_p & 0x3FFF;
    task.encodingPath.push_back(86);
    task.encodingPath.push_back(72);
    task.encodingPath.push_back(73);
    task.encodingPath.push_back(71);
    task.encodingPath.push_back(85);
    task.encodingPath.push_back(176);
    task.encodingPath.push_back(207);
    task.encodingPath.push_back(206);
    task.encodingPath.push_back(204);
    t0 = utime_ns();
    cemap.EncirclePlanner(task);
    t1 = utime_ns();
    std::cout << "Itime(ns): " << (t1 - t0) << std::endl;
    std::cout << "task.cost: " << task.cost << std::endl;
    drawBIpath(task.minpath, image);
    cv::imshow("Video", image);
    cv::waitKey();

    return 0;
}
