from Edge import Edge
import Brute

input_file_name = '4_city.txt'

def readInputFile():
    lines = [line for line in open(input_file_name) if not line.startswith('#') and len(line.strip())]
    numOfNodes = int(lines[0].split("\n")[0])
    reliability = []
    cost = []
    with open(input_file_name, 'r') as file:
        lines = file.readlines()
    for line in lines[8:8 + numOfNodes - 1 + numOfNodes - 2]:
        if line.startswith('#') or line.strip() == "":
            continue
        elements = line.strip().split()
        row = [float(e) for e in elements]
        reliability.append(row)
    for line in lines[len(lines) - 1 - 2 - (numOfNodes - 1):]:
        if line.startswith('#') or line.strip() == "":
            continue
        elements = line.strip().split()
        row = [int(e) for e in elements]
        cost.append(row)
    reliability = [item for sublist in reliability for item in sublist]
    cost = [item for sublist in cost for item in sublist]
    # print("Number of nodes: ", numOfNodes)
    # print("Reliability:",reliability)
    # print("Cost:", cost)
    return [numOfNodes, reliability, cost]


def sortEdges(nodeNum, reliabilities, costs):
    Edges = []
    index = 0
    for i in range(nodeNum):
        for j in range(i + 1, nodeNum):
            temp = Edge(i, j)
            temp.set_reliability(reliabilities[index])
            temp.set_cost(costs[index])
            Edges.append(temp)
            index = index + 1
    Edges.sort(key=lambda x: (x.reliability, -x.cost), reverse=True)
    return Edges

def sortEdgesByCost(nodeNum, reliabilities, costs):
    Edges = []
    index = 0
    for i in range(nodeNum):
        for j in range(i + 1, nodeNum):
            temp = Edge(i, j)
            temp.set_reliability(reliabilities[index])
            temp.set_cost(costs[index])
            Edges.append(temp)
            index = index + 1
    Edges.sort(key=lambda x: (x.cost, -x.reliability), reverse=False)
    return Edges

def findMST(nodeNum, sortedEdges):
    nodes = []
    edges = []
    nodes.append(sortedEdges[0].get_cityA())
    nodes.append(sortedEdges[0].get_cityB())
    edges.append(sortedEdges[0])
    for i in range(1, len(sortedEdges)):
        if len(nodes) == nodeNum:
            break
        x = sortedEdges[i].get_cityA()
        y = sortedEdges[i].get_cityB()
        if x in nodes and y in nodes:
            continue
        else:
            if x in nodes:
                nodes.append(y)
            elif y in nodes:
                nodes.append(x)
            else:
                nodes.append(y)
                nodes.append(x)
            edges.append(sortedEdges[i])

    return edges


# find reliability of spanning tree
def findTotalReliabilityST(paths):
    reliability = 1
    for e in paths:
        temp = e.get_reliability()
        reliability = temp * reliability
    return reliability


# find total cost of the spanning tree
def findTotalCostST(path):
    cost = 0
    for e in path:
        cost += e.get_cost()
    return cost


def main():
    costGoal = int(input("Please Input Cost goal :"))
    method = int(input("1 for Brute force. 2 for optimized method:"))

    if method != 1 and method != 2:
        print("Invalid method input!")
        return
    inputValue = readInputFile()
    numOfNodes = inputValue[0]
    allReliabilty = inputValue[1]
    allCost = inputValue[2]
    feasible = False
    sortedEdgesByReli = sortEdges(numOfNodes, allReliabilty, allCost)
    sortEdgesCost = sortEdgesByCost(numOfNodes, allReliabilty, allCost)
    if method == 2:
            MST_COST = findMST(numOfNodes,sortEdgesCost)            #sort by cost from low to high
            MST = findMST(numOfNodes, sortedEdgesByReli)

            curR = findTotalReliabilityST(MST)
            curR_Cost = findTotalReliabilityST(MST_COST)

            curC = findTotalCostST(MST)
            curC_Cost = findTotalCostST(MST_COST)

            # print("total cost: %s, total reliability: %s when prioritize reliability"  % (curC, curR))
            # print("total cost: %s, total reliability: %s when prioritize cost" % (curC_Cost, curR_Cost))

            curEdges = MST.copy()
            curEdges_COST = MST_COST.copy()

            remainingEdges = []  # list of edges that can be potentially enhanced
            remainingEdges_COST = []

            max_reliability_by_reli = 0
            max_reliability_by_cost = 0

            for edge in sortedEdgesByReli:
                if edge not in MST:
                    remainingEdges.append(edge)
                if edge not in MST_COST:
                    remainingEdges_COST.append(edge)

            while costGoal - curC >= min([edge.cost for edge in remainingEdges]):
                feasible = True
                rUnadded = [0] * len(remainingEdges)
                cUnadded = [0] * len(remainingEdges)
                ratio = [0] * len(remainingEdges)
                haveSpace = [0] * len(remainingEdges)
                for i in range(len(remainingEdges)):
                    e = remainingEdges[i]
                    cloneList = curEdges.copy()
                    cloneList.append(e)
                    cUnadded[i] = findTotalCostST(cloneList)
                    if cUnadded[i] > costGoal:
                        ratio[i] = -1
                        continue
                    perfectList = []
                    rUnadded[i] = findGraphReliability(cloneList, perfectList, len(MST), numOfNodes)
                    ratio[i] = rUnadded[i] / cUnadded[i]
                    remainingEdgesCopy = remainingEdges.copy()
                    remainingEdgesCopy.remove(e)
                    haveSpace[i] = 1 if curC - findTotalCostST(cloneList) >= min([edge.cost for edge in remainingEdgesCopy]) else 0
                for i in ratio:
                    if i > 0:
                        feasible = True
                maxRatio = max(ratio)
                maxReliability = max(rUnadded)
                if ratio.index(maxRatio) == rUnadded.index(maxReliability):
                    maxIndex = ratio.index(maxRatio)
                else:
                    if haveSpace[rUnadded.index(maxReliability)] == 1:
                        maxIndex = rUnadded.index(maxReliability)
                    elif haveSpace[rUnadded.index(maxReliability)] == 0 and haveSpace[ratio.index(maxRatio)] == 1:
                        maxIndex = ratio.index(maxRatio)
                    else:
                        maxIndex = rUnadded.index(maxReliability)
                maxReliability = rUnadded[maxIndex]
                max_reliability_by_reli = max(curR,maxReliability)
                curEdges.append(remainingEdges[maxIndex])
                curC = findTotalCostST(curEdges)
               # print("One edge found and added. Current Reliability: %s, Current Cost: %s" % (maxReliability, curC))
                remainingEdges.remove(remainingEdges[maxIndex])

            while costGoal - curC_Cost >= min([edge.cost for edge in remainingEdges_COST]):
                feasible = True
                rUnadded_COST = [0] * len(remainingEdges_COST)
                cUnadded_COST = [0] * len(remainingEdges_COST)
                ratio = [0] * len(remainingEdges_COST)
                haveSpace = [0] * len(remainingEdges_COST)
                for i in range(len(remainingEdges_COST)):
                    e = remainingEdges_COST[i]
                    cloneList = curEdges_COST.copy()
                    cloneList.append(e)
                    cUnadded_COST[i] = findTotalCostST(cloneList)
                    if cUnadded_COST[i] > costGoal:
                        ratio[i] = -1
                        continue
                    perfectList = []
                    rUnadded_COST[i] = findGraphReliability(cloneList, perfectList, len(MST_COST), numOfNodes)
                    ratio[i] = rUnadded_COST[i] / cUnadded_COST[i]
                    remainingEdgesCopy = remainingEdges_COST.copy()
                    remainingEdgesCopy.remove(e)
                    haveSpace[i] = 1 if curC - findTotalCostST(cloneList) >= min([edge.cost for edge in remainingEdgesCopy]) else 0
                for i in ratio:
                    if i > 0:
                        feasible = True
                maxRatio = max(ratio)
                maxReliability = max(rUnadded_COST)
                if ratio.index(maxRatio) == rUnadded_COST.index(maxReliability):
                    maxIndex = ratio.index(maxRatio)
                else:
                    if haveSpace[rUnadded_COST.index(maxReliability)] == 1:
                        maxIndex = rUnadded_COST.index(maxReliability)
                    elif haveSpace[rUnadded_COST.index(maxReliability)] == 0 and haveSpace[ratio.index(maxRatio)] == 1:
                        maxIndex = ratio.index(maxRatio)
                    else:
                        maxIndex = rUnadded_COST.index(maxReliability)
                maxReliability = rUnadded_COST[maxIndex]
                max_reliability_by_cost = max(curR_Cost,maxReliability)
                curEdges_COST.append(remainingEdges_COST[maxIndex])
                curC_Cost = findTotalCostST(curEdges_COST)
                #print("One edge found and added. Current Reliability: %s, Current Cost: %s" % (maxReliability, curC_Cost))
                remainingEdges_COST.remove(remainingEdges_COST[maxIndex])

            if feasible:
                max_reliability_by_reli = max(curR,max_reliability_by_reli)
                max_reliability_by_cost = max(curR_Cost,max_reliability_by_cost)

                if max_reliability_by_reli > max_reliability_by_cost:
                    print("Under cost limit of %s, max reliability is %s" % (costGoal,max_reliability_by_reli))
                    curEdges.append(max_reliability_by_reli)
                    Brute.draw(curEdges,numOfNodes,costGoal)
                else:
                    print("Under cost limit of %s, max reliability is %s" % (costGoal,max_reliability_by_cost))
                    curEdges_COST.append(max_reliability_by_cost)
                    Brute.draw(curEdges_COST,numOfNodes,costGoal)
                print("No more improvements.")
            else:
                print("No feasible Solution. Quit")
                return
    elif method == 1:
        allCombination = Brute.findCombination(sortedEdgesByReli, numOfNodes, costGoal)
        exhaustive_result = Brute.exhaustive(allCombination, numOfNodes)
        if exhaustive_result == 0:
            print("No feasible Solution. Quit")
            return
        print("Under cost limit of %s, max reliability is %s" % (costGoal,exhaustive_result[-1]))
        Brute.draw(exhaustive_result, numOfNodes, costGoal)




# recursively find the reliability
# input: Edges: all Edges in the graph.
# input: numOfEdgesMST : number of edges in MST
# input: PerfectEdges: Edges that are reliable under the assumption.
def findGraphReliability(Edges, PerfectEdges, numOfEdgesMST, numNodes):
    totalReliabilty = 0
    sortedEdges = sorted(Edges, key=lambda x: x.get_cityA(), reverse=True)
    if len(sortedEdges) + len(PerfectEdges) == numOfEdgesMST and isConnect(sortedEdges, PerfectEdges, numNodes):
        return findTotalReliabilityST(Edges)
    else:
        if not isConnect(sortedEdges, PerfectEdges, numNodes):
            return 0
        if len(sortedEdges) > 0:
            e = sortedEdges[0]
            clonedEdges = sortedEdges.copy()
            clonedEdges.remove(e)
            totalReliabilty += (1 - e.get_reliability()) * findGraphReliability(clonedEdges, PerfectEdges,
                                                                                numOfEdgesMST, numNodes)
            PerfectEdges.append(e)
            totalReliabilty += e.get_reliability() * findGraphReliability(clonedEdges, PerfectEdges, numOfEdgesMST,
                                                                          numNodes)
            return totalReliabilty
        else:
            return 1


def isConnect(Edges, PerfectEdges, numOfNodes):
    connectivity = [0] * numOfNodes  # initialize as not connected
    connectivity[0] = 1
    redo = True
    while (redo):
        redo = False
        for edge in Edges:
            if connectivity[edge.get_cityA()] != connectivity[edge.get_cityB()]:
                connectivity[edge.get_cityA()] = 1
                connectivity[edge.get_cityB()] = 1
                redo = True
        for edge in PerfectEdges:
            if connectivity[edge.get_cityA()] != connectivity[edge.get_cityB()]:
                connectivity[edge.get_cityA()] = 1
                connectivity[edge.get_cityB()] = 1
                redo = True

    if 0 in connectivity:
        return False
    else:
        return True


if __name__ == "__main__":
    main()
