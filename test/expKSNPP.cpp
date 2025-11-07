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

struct testSNPPtask
{
    char *name;
    BIpoint x_init;
    BIpoint x_goal;
};

std::vector<testSNPPtask> testTasks = {
    {(char *)"./expMap/MESS.png", {1000, 1125}, {200, 150}},
    {(char *)"./expMap/MAZE.png", {1100, 2200}, {2360, 360}},
    {(char *)"./expMap/GAME.png", {180, 320}, {2400, 2350}},
    {(char *)"./expMap/FLOOR.png", {855, 1935}, {1592, 173}},
};

int32_t kNUM[] = {1, 10, 25, 50};
int main()
{

    // 创建窗口并注册鼠标回调函数
    cv::namedWindow("Video", 0);
    cv::resizeWindow("Video", 1200, 1200);
    cv::setMouseCallback("Video", onMouse);

    int32_t map_index = 0;
    std::cout << "Input map_index:";
    std::cin >> map_index;
    if (map_index >= testTasks.size())
        return 1;
    testSNPPtask &testTask = testTasks[map_index];

    BImap cemap;
    kSNPPtask task;
    int64_t t0 = utime_ns();
    cemap.MaptoBInavi(testTask.name, 0.1, 0.9);

    int64_t t1 = utime_ns();
    std::cout << "Itime(ns): " << (t1 - t0) << std::endl;
    cemap.DrawBImap(true);
    // std::cout << "freepolygonList: " << cemap.BIgraphList[2].freepolygonList.size() << std::endl;
    // std::cout << "cutlineList: " << cemap.BIgraphList[2].cutlineList.size() << std::endl;

    cv::Mat image;
    int count = 1000;
    for (size_t k = 0; k < 1; k++)
    {
        t0 = utime_ns();
        for (size_t i = 0; i < count; i++)
            cemap.kSNPPlanner(task, testTask.x_init, testTask.x_goal, kNUM[k]); //, false
        t1 = utime_ns();
        std::cout << k << " time(ns): " << (t1 - t0) / count << std::endl;
        std::cout << kNUM[k] << "cost kNUM[k]: " << task.Q_Costs.back() << std::endl;
    }

    // cemap.kSNPPlanner(task, testTask.x_init, testTask.x_goal, 5);
    // for (size_t i = 0; i < 4; i++)
    // {
    //     image = cemap.BIdmap.clone();
    //     drawBIpath(task.Q_kSNP[i], image, 0, cv::Scalar(0, 0, 255));
    //     std::stringstream ss;
    //     ss << "map_index" << map_index << i << ".png";
    //     std::string text = ss.str();
    //     cv::imwrite(text, image);
    // }
    // image = cemap.BIdmap.clone();
    // drawBIpath(task.Q_kSNP[0], image, 0, cv::Scalar(0, 0, 255));
    // drawBIpath(task.Q_kSNP[1], image, 0, cv::Scalar(0, 0, 255));
    // drawBIpath(task.Q_kSNP[2], image, 0, cv::Scalar(0, 0, 255));
    // drawBIpath(task.Q_kSNP[3], image, 0, cv::Scalar(0, 0, 255));
    // drawBIpath(task.Q_kSNP[4], image, 0, cv::Scalar(0, 0, 255));

    // std::stringstream ss;
    // ss << "mapdebug_index" << map_index << ".png";
    // std::string text = ss.str();
    // cv::imwrite(text, cemap.BIdmap.clone());

    // std::unordered_set<int32_t> InvPolygons;
    // cemap.ReduceBranches(task.graphIndex, task.xx_init, task.xx_goal, InvPolygons);
    // std::cout << "InvPolygons " << InvPolygons.size() << std::endl;

    // image = cemap.BIdmap.clone();

    // for (int32_t index : InvPolygons)
    // {
    //     BIfreepolygon &freepolygon = cemap.BIgraphList[task.graphIndex].freepolygonList[index];
    //     BIpolygon &polygon = freepolygon.polygon;
    //     cv::Point PointArray[polygon.size()];
    //     for (int32_t pnum = 0; pnum < polygon.size(); pnum++)
    //     {
    //         PointArray[pnum].x = polygon[pnum].x;
    //         PointArray[pnum].y = polygon[pnum].y;
    //     }
    //     cv::Scalar color(255, 255, 255);
    //     cv::fillConvexPoly(image, PointArray, polygon.size(), color);
    //     cv::fillConvexPoly(image, PointArray, polygon.size(), color);
    // }
    // std::stringstream ss2;
    // ss2 << "mapInvdebug_index" << map_index << ".png";
    // std::string text2 = ss2.str();
    // cv::imwrite(text2, image.clone());

    return 0;
}
