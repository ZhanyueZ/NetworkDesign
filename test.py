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
    sortedEdgesByReli = sortEdges(numOfNodes, allReliabilty, allCost)
    MST = findMST(numOfNodes, sortedEdgesByReli)
    curR = findTotalReliabilityST(MST)
    curC = findTotalCostST(MST)
    


if __name__ == "__main__":
    main()
