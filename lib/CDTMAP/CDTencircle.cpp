#include "CDTmap.h"
#include <math.h>

double BImap::EncircleHeuristicCost(EncirclePtask &task,
                                    std::map<int32_t, std::list<BIpoint>> &HistoricalSolutions,
                                    std::unordered_map<int32_t, int32_t> &EncodingTree,
                                    int32_t HEncoding)
{
    BIpoint &x_init = task.x_s;
    BIpoint &x_goal = task.x_g;
    int32_t obs_goal = task.obs_goal;
    std::vector<int32_t> &encodingPath = task.encodingPath;
    std::unordered_set<int32_t> &goalfreePolygons = task.goalfreePolygons;
    std::unordered_map<int32_t, int32_t> &EncodingIndexMap = task.EncodingIndexMap;

    BIgraph &graph = BIgraphList[task.graphIndex];
    std::vector<BIcutline> &cutlineList = graph.cutlineList;
    std::vector<BIfreepolygon> &freepolygonList = graph.freepolygonList;
    BIinvnodeMap &invnode2cutlineMap = graph.invnode2cutlineMap;
    BIobspolygon &obspolygon = graph.obspolygonList[obs_goal];
    std::vector<int32_t> &polygonRlist = obspolygon.polygonRlist;

    // 回溯EncodingTree
    std::vector<int32_t> polyPath_s;
    std::vector<int32_t> polyPath_e;
    int32_t HEncoding_temp = HEncoding;
    int32_t HEncoding_root;
    while (HEncoding_temp != -1)
    {
        int32_t PolygonIndex = 0x0000FFFF & HEncoding_temp;
        polyPath_s.push_back(PolygonIndex);
        polyPath_e.push_back(PolygonIndex);
        HEncoding_root = HEncoding_temp;
        HEncoding_temp = EncodingTree[HEncoding_temp];
    }
    int32_t rootIndex = EncodingIndexMap[HEncoding_root];
    for (int32_t i = rootIndex - 1; i >= 0; i--)
        polyPath_s.push_back(encodingPath[i]);
    for (int32_t i = rootIndex + 1; i < encodingPath.size(); i++)
        polyPath_e.push_back(encodingPath[i]);

    // 判断HEncoding是否为goal多边形
    int32_t EndPolygonIndex = 0x0000FFFF & HEncoding;

    if (goalfreePolygons.count(EndPolygonIndex)) // 计算真实成本
    {
        double Costtemp1, Costtemp2;
        std::list<BIpoint> Pathtemp1, Pathtemp2;
        {
            std::vector<int32_t> polyPath;
            polyPath.insert(polyPath.end(), polyPath_s.rbegin(), polyPath_s.rend());
            int32_t startObsIndex = obspolygon.polygonRmap[EndPolygonIndex];
            int32_t polygonRlistSize = polygonRlist.size();
            for (int32_t index = 1; index < polygonRlistSize; index++)
                polyPath.push_back(polygonRlist[(index + startObsIndex) % polygonRlistSize]);
            polyPath.insert(polyPath.end(), polyPath_e.begin(), polyPath_e.end());
            ReversePathClearing(polyPath);

            std::vector<BIline> cpath;
            cpath.emplace_back(x_goal, x_goal);
            for (int32_t i = polyPath.size() - 1; i > 0; i--)
            {
                int32_t cutlineIndex_Par2Now = invnode2cutlineMap[{polyPath[i], polyPath[i - 1]}];
                cpath.push_back(cutlineList[cutlineIndex_Par2Now].line);
            }
            cpath.emplace_back(x_init, x_init);
            GetLeastHomotopyPath(cpath, Pathtemp1, Costtemp1);
        }
        {
            std::vector<int32_t> polyPath;
            polyPath.insert(polyPath.end(), polyPath_s.rbegin(), polyPath_s.rend());
            int32_t startObsIndex = obspolygon.polygonRmap[EndPolygonIndex];
            int32_t polygonRlistSize = polygonRlist.size();
            for (int32_t index = polygonRlistSize - 1; index > 0; index--)
                polyPath.push_back(polygonRlist[(index + startObsIndex) % polygonRlistSize]);
            polyPath.insert(polyPath.end(), polyPath_e.begin(), polyPath_e.end());
            ReversePathClearing(polyPath);

            std::vector<BIline> cpath;
            cpath.emplace_back(x_goal, x_goal);
            for (int32_t i = polyPath.size() - 1; i > 0; i--)
            {
                int32_t cutlineIndex_Par2Now = invnode2cutlineMap[{polyPath[i], polyPath[i - 1]}];
                cpath.push_back(cutlineList[cutlineIndex_Par2Now].line);
            }
            cpath.emplace_back(x_init, x_init);
            GetLeastHomotopyPath(cpath, Pathtemp2, Costtemp2);
        }
        if (Costtemp1 < Costtemp2)
        {
            std::swap(HistoricalSolutions[HEncoding], Pathtemp1);
            return Costtemp1;
        }
        else
        {
            std::swap(HistoricalSolutions[HEncoding], Pathtemp2);
            return Costtemp2;
        }
    }
    else // 计算启发式成本
    {
        BIpoint polyPath_s_epoint;
        BIpoint polyPath_e_epoint;
        double Costtemp1, Costtemp2;
        double cutline1, cutline2;
        if (polyPath_s.size() != 1)
        {
            int32_t cutlineIndex_Par2Now0 = invnode2cutlineMap[{polyPath_s[0], polyPath_s[1]}];
            BIcutline &cutlineTemp = cutlineList[cutlineIndex_Par2Now0];
            polyPath_s_epoint = cutlineTemp.core;
            cutline1 = cutlineTemp.line.S % cutlineTemp.line.E;
            std::vector<BIline> cpath;
            cpath.emplace_back(polyPath_s_epoint, polyPath_s_epoint);
            for (int32_t i = 1; i < polyPath_s.size() - 1; i++)
            {
                int32_t cutlineIndex_Par2Now = invnode2cutlineMap[{polyPath_s[i], polyPath_s[i + 1]}];
                cpath.push_back(cutlineList[cutlineIndex_Par2Now].line);
            }
            cpath.emplace_back(x_init, x_init);
            std::list<BIpoint> Pathtemp;
            GetLeastHomotopyPath(cpath, Pathtemp, Costtemp1);
            // cv::Mat image = BIdmap.clone();
            // drawBIpath(Pathtemp, image);
            // cv::waitKey();
        }
        else
        {
            polyPath_s_epoint = x_init;
            Costtemp1 = 0;
            cutline1 = 0;
        }
        if (polyPath_e.size() != 1)
        {
            int32_t cutlineIndex_Par2Now0 = invnode2cutlineMap[{polyPath_e[0], polyPath_e[1]}];
            BIcutline &cutlineTemp = cutlineList[cutlineIndex_Par2Now0];
            polyPath_e_epoint = cutlineTemp.core;
            cutline2 = cutlineTemp.line.S % cutlineTemp.line.E;
            std::vector<BIline> cpath;
            cpath.emplace_back(polyPath_e_epoint, polyPath_e_epoint);
            for (int32_t i = 1; i < polyPath_e.size() - 1; i++)
            {
                int32_t cutlineIndex_Par2Now = invnode2cutlineMap[{polyPath_e[i], polyPath_e[i + 1]}];
                cpath.push_back(cutlineList[cutlineIndex_Par2Now].line);
            }
            cpath.emplace_back(x_goal, x_goal);
            std::list<BIpoint> Pathtemp;
            GetLeastHomotopyPath(cpath, Pathtemp, Costtemp2);
            // cv::Mat image = BIdmap.clone();
            // drawBIpath(Pathtemp, image);
            // cv::waitKey();
        }
        else
        {
            polyPath_e_epoint = x_goal;
            Costtemp2 = 0;
            cutline2 = 0;
        }

        double cutline1_half = cutline1 / 2;
        double cutline2_half = cutline2 / 2;
        if (polyPath_s_epoint == polyPath_e_epoint)
        {
            double informedCost = task.informed.informedFun(polyPath_s_epoint);
            // std::cout << "informedCost1: " << informedCost << std::endl;
            if (informedCost == 0)
                informedCost = task.informed._baseperimeter;
            informedCost = std::max(informedCost - cutline1_half - cutline2_half, 0.0);
            informedCost += std::max(Costtemp1 - cutline1_half, 0.0);
            informedCost += std::max(Costtemp2 - cutline2_half, 0.0);
            return informedCost;
        }
        else
        {
            double informedCost = task.informed.informedFun((polyPath_s_epoint + polyPath_e_epoint) / 2.0);
            if (informedCost == 0)
                informedCost = task.informed._baseperimeter;
            double polyPath_se_Interval = polyPath_s_epoint % polyPath_e_epoint;
            informedCost = std::max(informedCost - polyPath_se_Interval -
                                        cutline1_half - cutline2_half,
                                    0.0);
            // std::cout << "informedCost2: " << informedCost << std::endl;
            informedCost += std::max(Costtemp1 - cutline1_half, 0.0);
            informedCost += std::max(Costtemp2 - cutline2_half, 0.0);
            return informedCost;
        }
    }
}

void BImap::EncirclePlanner(EncirclePtask &task)
{
    task.minpath.clear();
    task.cost = doubleMax;

    BIpoint &x_init = task.x_s;
    BIpoint &x_goal = task.x_g;
    int32_t obs_goal = task.obs_goal;
    std::vector<int32_t> &encodingPath = task.encodingPath;
    std::unordered_set<int32_t> &goalfreePolygons = task.goalfreePolygons;
    std::unordered_map<int32_t, int32_t> &EncodingIndexMap = task.EncodingIndexMap;
    goalfreePolygons.clear();
    EncodingIndexMap.clear();

    uint32_t anum_Init = BIamap.at<uint16_t>(x_init.y, x_init.x);
    uint32_t anum_Goal = BIamap.at<uint16_t>(x_goal.y, x_goal.x);
    int32_t PolyIndex_Init = BIimap.at<uint16_t>(x_init.y, x_init.x);
    int32_t PolyIndex_Goal = BIimap.at<uint16_t>(x_goal.y, x_goal.x);
    if ((PolyIndex_Init & 0xC000) != 0 || (PolyIndex_Goal & 0xC000) != 0)
    {
        printf("error: robot can't be here!!\r\n");
        return;
    }

    task.graphIndex = anum_Init;
    BIgraph &graph = BIgraphList[anum_Init];

    if (obs_goal < 0 || obs_goal >= graph.obspolygonList.size())
    {
        printf("error: robot can't be here!!\r\n");
        return;
    }

    std::vector<BIcutline> &cutlineList = graph.cutlineList;
    std::vector<BIfreepolygon> &freepolygonList = graph.freepolygonList;
    BIinvnodeMap &invnode2cutlineMap = graph.invnode2cutlineMap;

    BIobspolygon &obspolygon = graph.obspolygonList[obs_goal];
    for (int32_t freeIndex : obspolygon.polygonRlist)
    {
        goalfreePolygons.insert(freeIndex);
    }
    task.informed.Initialize(obspolygon.polygon);

    std::map<int32_t, std::list<BIpoint>> HistoricalSolutions;
    std::priority_queue<std::pair<double, int32_t>,
                        std::vector<std::pair<double, int32_t>>,
                        std::greater<std::pair<double, int32_t>>>
        Q_var;

    std::vector<int32_t> EncodingSet;
    std::unordered_map<int32_t, int32_t> EncodingTree;
    EncodingSet.resize(freepolygonList.size(), -1);
    for (size_t i = 0; i < encodingPath.size(); ++i)
    {
        int32_t PolyIndex = encodingPath[i];
        EncodingSet[PolyIndex]++;
        int32_t HEncoding = (EncodingSet[PolyIndex] << 16) | PolyIndex;
        EncodingTree[HEncoding] = -1;
        EncodingIndexMap[HEncoding] = i;

        // 待修改HistoricalSolutions
        std::vector<int32_t> PathEncoding_0;
        double HeuristicCost = EncircleHeuristicCost(task, HistoricalSolutions, EncodingTree, HEncoding);
        Q_var.push({HeuristicCost, HEncoding});
        // std::cout << "HeuristicCost: " << HeuristicCost << std::endl;
    }

    while (Q_var.size())
    {
        double CostNow = Q_var.top().first;
        int32_t PathEncodingNow = Q_var.top().second;
        int32_t PolygonNow = PathEncodingNow & 0x0000FFFF;
        Q_var.pop();

        if (goalfreePolygons.count(PolygonNow))
        {
            task.cost = CostNow;
            std::swap(task.minpath, HistoricalSolutions[PathEncodingNow]);
            break;
        }
        int32_t PathEncodingPar = EncodingTree[PathEncodingNow];
        int32_t PolyIndexPar = PathEncodingPar & 0x0000FFFF;

        for (int32_t PnearIndex : freepolygonList[PolygonNow].polygonlink)
        {
            if (PnearIndex == PolyIndexPar)
                continue;
            // 根节点不可横向生长
            if (PathEncodingPar == -1)
            {
                int32_t enPathNow_Index = EncodingIndexMap[PathEncodingNow];
                if (enPathNow_Index > 0 && PnearIndex == (encodingPath[enPathNow_Index - 1] & 0x0000FFFF))
                    continue;
                if (enPathNow_Index < encodingPath.size() - 1 &&
                    PnearIndex == (encodingPath[enPathNow_Index + 1] & 0x0000FFFF))
                    continue;
            }

            EncodingSet[PnearIndex]++;
            int32_t PathEncodingSub = (EncodingSet[PnearIndex] << 16) | PnearIndex;
            EncodingTree[PathEncodingSub] = PathEncodingNow;

            std::vector<int32_t> PathEncoding_0;
            double HeuristicCost = EncircleHeuristicCost(task, HistoricalSolutions, EncodingTree, PathEncodingSub);
            Q_var.push({HeuristicCost, PathEncodingSub});
        }
    }
}
