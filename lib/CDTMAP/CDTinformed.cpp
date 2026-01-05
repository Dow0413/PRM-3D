#include "CDTmap.h"


bool BIinformed::comparePoint(const BIpoint &p1, const BIpoint &p2)
{
    if (p1.y < p2.y)
        return true;
    else if (p1.y == p2.y)
        return p1.x < p2.x;
    return false;
}

double BIinformed::crossProduct(const BIpoint &O, const BIpoint &A, const BIpoint &B)
{
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}
// 计算两点之间的距离
double BIinformed::distance(const BIpoint &p1, const BIpoint &p2)
{
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}
// 用于比较极角的函数
bool BIinformed::comparePolar(const BIpoint &base, const BIpoint &p1, const BIpoint &p2)
{
    double angle1 = atan2(p1.y - base.y, p1.x - base.x);
    double angle2 = atan2(p2.y - base.y, p2.x - base.x);
    if (angle1 < angle2)
        return true;
    if (angle1 == angle2)
        return distance(base, p1) < distance(base, p2);
    return false;
}
// Graham扫描算法计算凸包
double BIinformed::grahamScan(std::vector<BIpoint> &points, std::vector<BIpoint> &hull)
{
    if (points.size() < 3)
    {
        hull = points;
        double perimeter = 0.0;
        for (size_t i = 0; i < hull.size(); ++i)
            perimeter += distance(hull[i], hull[(i + 1) % hull.size()]);
        return perimeter;
    }
    std::sort(points.begin(), points.end(), &BIinformed::comparePoint); // 找到基点
    BIpoint base = points[0];
    std::sort(points.begin() + 1, points.end(), [base](const BIpoint &p1, const BIpoint &p2)
              { return comparePolar(base, p1, p2); }); // 按照极角排序
    hull.clear();
    hull = {points[0], points[1]};
    for (size_t i = 2; i < points.size(); ++i)
    {
        while (hull.size() >= 2 && crossProduct(hull[hull.size() - 2], hull.back(), points[i]) <= 0)
            hull.pop_back();
        hull.push_back(points[i]);
    }
    double perimeter = 0.0; // 计算凸包的周长
    for (size_t i = 0; i < hull.size(); ++i)
        perimeter += distance(hull[i], hull[(i + 1) % hull.size()]);
    return perimeter;
}

void BIinformed::Initialize(std::vector<BIpoint> &points)
{
    BIpolygon polygon;
    _baseperimeter = grahamScan(points, polygon);
    _core = {0, 0};
    for (const BIpoint &p : polygon)
    {
        _core = _core + p;
    }
    _core = _core / polygon.size();
    _polygon.resize(polygon.size());
    for (int32_t index = 0; index < polygon.size(); index++)
    {
        double angle = calculateAngle(_core, polygon[index]);
        _polarIndex[angle] = index;
        _polygon[index] = polygon[index];
    }
    _polarIndex[-1.0] = polygon.size() - 1; // 额外添加一个保守的下界做循环
}

double BIinformed::informedFun(BIpoint point)
{
    double polarAngle = calculateAngle(_core, point);
    auto it = _polarIndex.lower_bound(polarAngle);

    int32_t polygonSize = _polygon.size();
    int32_t downIndex = it->second + polygonSize; // 未归一下边界点引索,防止递减越界
    int32_t upIndex = downIndex + 1;              // 未归一上边界点引索
    BIpoint downpoint = _polygon[downIndex % polygonSize];
    BIpoint uppoint = _polygon[upIndex % polygonSize];
    double cross = crossProduct(downpoint, point, uppoint);

    if (cross <= 0)
        return 0; // return 0 //如果point在_polygon内部直接返回

    // 开始左右迭代至最后的可视点
    double InformedPerimeter = _baseperimeter - distance(downpoint, uppoint);
    // 查找下边界
    int32_t downIndex_old = downIndex;
    BIpoint downpoint_old = downpoint;
    for (;;)
    {
        downIndex--;
        downpoint = _polygon[downIndex % polygonSize];
        if (crossProduct(downpoint_old, point, downpoint) >= 0)
            break;
        InformedPerimeter -= distance(downpoint, downpoint_old);
        downIndex_old = downIndex;
        downpoint_old = downpoint;
    }
    InformedPerimeter += distance(point, downpoint_old);

    // 查找上边界
    int32_t upIndex_old = upIndex;
    BIpoint uppoint_old = uppoint;
    for (;;)
    {
        upIndex++;
        uppoint = _polygon[upIndex % polygonSize];
        if (crossProduct(uppoint_old, uppoint, point) >= 0)
            break;
        InformedPerimeter -= distance(uppoint, uppoint_old);
        upIndex_old = upIndex;
        uppoint_old = uppoint;
    }
    InformedPerimeter += distance(point, uppoint_old);

    // cv::Mat image = showmat.clone();
    // cv::line(image, cv::Point((int)(point.x), (int)(point.y)),
    //          cv::Point((int)(uppoint_old.x), (int)(uppoint_old.y)), cv::Scalar(255, 255, 0), 2);
    // cv::line(image, cv::Point((int)(point.x), (int)(point.y)),
    //          cv::Point((int)(downpoint_old.x), (int)(downpoint_old.y)), cv::Scalar(255, 255, 0), 2);
    // cv::imshow("Video", image);
    // cv::waitKey(10);
    return InformedPerimeter;
}

double BIinformed::informedFun(BIline line, double threshold)
{
    double low = 0.0;
    double high = 1.0;
    BIpoint S = line.S;
    BIpoint E = line.E;
    BIpoint S2E = E - S;

    // 若在保守环内部返回0
    double informedS = informedFun(S);
    if (informedS <= _baseperimeter)
        return 0;
    double informedM = informedFun(S + S2E * 0.5);
    if (informedM <= _baseperimeter)
        return 0;
    double informedE = informedFun(E);
    if (informedE <= _baseperimeter)
        return 0;

    // 若单调则返回端点
    if (informedS <= informedM && informedM <= informedE)
        return informedS;
    if (informedS >= informedM && informedM >= informedE)
        return informedE;

    for (;;)
    {
        double mid1 = low + (high - low) / 3.0;
        double mid2 = high - (high - low) / 3.0;

        double informed1 = informedFun(S + S2E * mid1);
        if (informed1 <= _baseperimeter)
            return 0;
        double informed2 = informedFun(S + S2E * mid2);
        if (informed2 <= _baseperimeter)
            return 0;
        if (informed1 < informed2)
            high = mid2;
        else
            low = mid1;
        if (fabs(informed1 - informed2) < threshold)
            break;
    }

    return informedFun(S + S2E * ((low + high) / 2.0));
}
