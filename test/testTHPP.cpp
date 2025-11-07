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

// map1 {1241,1241}

int main()
{
    BImap cemap;
    kSNPPtask task;

    // 创建窗口并注册鼠标回调函数
    cv::namedWindow("Video", 0);
    cv::resizeWindow("Video", 1200, 1200);
    cv::setMouseCallback("Video", onMouse, &task);
    int64_t t0 = utime_ns();
    // cemap.MaptoBInavi((char *)"/home/tzyh_subsys/WorkSpace/kSNPP/expMap/THPP_map1.png", 0.1, 0.9);
    // cemap.MaptoBInavi((char *)"/home/tzyh_subsys/WorkSpace/THPP/expMap/MESS.png", 0.1, 0.9);
    cemap.MaptoBInavi((char *)"/home/tzyh_subsys/WorkSpace/THPP/expMap/GAME.png", 0.1, 0.9);
    int64_t t1 = utime_ns();
    std::cout << "Itime(ns): " << (t1 - t0) << std::endl;
    cemap.DrawBImap(true);
    // cv::imwrite("/home/tzyh_subsys/WorkSpace/THPP/expMap/MESSk.png", cemap.BIfmap);
    // cv::imwrite("/home/tzyh_subsys/WorkSpace/THPP/expMap/GAMEk.png", cemap.BIfmap);

    cv::Mat image;
    // int32_t key = -1;
    // int32_t mod = -1;
    // BIpoint x_init = {1048, 1125};
    // BIpoint x_goal = {2244, 84};
    // BIpoint x_init = {1194, 1210};
    // BIpoint x_goal = {346, 461};
    BIpoint x_init = {478, 704};
    BIpoint x_goal = {2686, 1580};
    int count = 100;
    t0 = utime_ns();
    for (size_t i = 0; i < count; i++)
    {
        cemap.kSNPPlanner(task, x_init, x_goal, 50);//, false
    }
    t1 = utime_ns();
    std::cout << "time(ns): " << (t1 - t0) / count << std::endl;
    std::cout << "test_count: " << cemap.test_count/ count << std::endl;

    for (int32_t i = 0; i < task.Q_kSNP.size(); i++)
    {
        image = cemap.BIdmap.clone();
        BIpoint pold = task.Q_kSNP[i].front();
        double costSum = 0;
        for (auto pnew : task.Q_kSNP[i])
        {
            costSum += (pold % pnew);
            pold = pnew;
        }

        // printf("k%02d: %7.2lf,  %7.2lf\r\n", i + 1, task.Q_Costs[i], fabs(task.Q_Costs[i] - costSum));
        // drawBIpath(task.Q_kSNP[i], image, 0, cv::Scalar(0, 0, 255));

        // std::string output_filename = "output_" + std::to_string(i) + ".png";

        // std::stringstream output_filename;
        // output_filename << "output_" << std::setfill('0') << std::setw(3) << i << ".png"; // 4位数字
        // cv::resize(image, image, cv::Size(), 1.0 / 4.0, 1.0 / 4.0, cv::INTER_AREA);
        // cv::imwrite(output_filename.str(), image);
        // cv::waitKey(30);
    }

    // drawBIfreeID(cemap.BIgraphList[0], image);

    std::unordered_set<int32_t> InvPolygons;
    cemap.ReduceBranches(task.graphIndex, task.xx_init, task.xx_goal, InvPolygons);
    const std::vector<BIfreepolygon> &freepolygonList = cemap.BIgraphList[task.graphIndex].freepolygonList;
    // for (const BIfreepolygon &polygon : freepolygonList)
    for (int32_t index : InvPolygons)
    {
        const BIfreepolygon &polygon = freepolygonList[index];
        std::stringstream ss;
        ss << polygon.index;
        std::string text = ss.str();
        cv::putText(image, text, cv::Point(int(polygon.core.x), int(polygon.core.y)), cv::FONT_HERSHEY_COMPLEX,
                    1, cv::Scalar(0, 0, 255), 2);
    }

    // cv::imshow("Video", image);
    // cv::waitKey(0);


    return 0;
}
