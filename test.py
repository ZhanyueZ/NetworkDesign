from Edge import Edge


def readInputFile():
    lines = [line for line in open('4_city.txt') if not line.startswith('#') and len(line.strip())]
    numOfNodes = int(lines[0].split("\n")[0])
    reliability = []
    cost = []
    with open('4_city.txt', 'r') as file:
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
    Edges.sort(key=lambda x: x.reliability, reverse=True)
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
    MST = findMST(numOfNodes, sortedEdgesByReli)
    curR = findTotalReliabilityST(MST)
    curC = findTotalCostST(MST)
    print("total cost: %s, total reliability: %s" % (curC, curR))
    curEdges = MST.copy()
    remainingEdges = []                          # list of edges that can be potentially enhanced
    for edge in sortedEdgesByReli:
        if edge not in MST:
            remainingEdges.append(edge)
    while curC < costGoal:
        rUnadded = [0]* len(remainingEdges)
        cUnadded = [0]* len(remainingEdges)
        ratio = [0]*len(remainingEdges)
        for i in range(len(remainingEdges)):
            e = remainingEdges[i]
            cloneList = curEdges.copy()
            cloneList.append(e)
            perfectList = []
            rUnadded[i] = findGraphReliability(cloneList,perfectList,len(MST),numOfNodes)
            cUnadded[i] = findTotalCostST(cloneList)
            ratio[i] = 1
            if(cUnadded[i] > costGoal):
                ratio[i] = -1
        feasible = True

    print("Max achievable Reliability after improvements:")





# recursively find the reliability
# input: Edges: all Edges in the graph.
# input: numOfEdgesMST : number of edges in MST
# input: PerfectEdges: Edges that are reliable under the assumption.
def findGraphReliability(Edges,PerfectEdges,numOfEdgesMST,numNodes):
    totalReliabilty = 0
    if len(Edges) + len(PerfectEdges) == numOfEdgesMST and isConnect(Edges,PerfectEdges,numNodes):
        return findTotalReliabilityST(Edges)
    else:
        if not isConnect(Edges,PerfectEdges,numNodes):
            return 0
        if len(Edges)>0:
            e = Edges[0]
            clonedEdges = Edges.copy()
            clonedEdges.remove(e)
            totalReliabilty += (1-e.get_reliability()) * findGraphReliability(clonedEdges,PerfectEdges,numOfEdgesMST,numNodes)
            PerfectEdges.append(e)
            totalReliabilty += e.get_reliability() * findGraphReliability(clonedEdges,PerfectEdges,numOfEdgesMST,numNodes)
            return totalReliabilty
        else:
            return 1


def isConnect(Edges,PerfectEdges,numOfNodes):
   connectivity = [0] * numOfNodes  # initialize as not connected
   connectivity[0] = 1
   redo = True
   while(redo):
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
