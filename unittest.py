from Edge import Edge
import NetworkDesigner


def main():
    # edge1 = Edge(0, 1)
    # edge1.set_reliability(0.94)
    # edge2 = Edge(1, 2)
    # edge2.set_reliability(0.94)
    # edge3 = Edge(2, 5)
    # edge3.set_reliability(0.94)
    # edge4 = Edge(0, 4)
    # edge4.set_reliability(0.93)
    # edge5 = Edge(0, 3)
    # edge5.set_reliability(0.96)
    # edge6 = Edge(1, 3)
    # edge6.set_reliability(0.97)
    # mst = [edge6,edge2,edge4,edge3,edge5,edge1]
    # pre = []
    # print(test.findGraphReliability(mst, pre, 5, 6))

    # edge1 = Edge(0, 1)
    # edge1.set_reliability(0.94)
    # edge2 = Edge(1, 2)
    # edge2.set_reliability(0.94)
    # edge3 = Edge(2, 5)
    # edge3.set_reliability(0.94)
    # edge4 = Edge(0, 4)
    # edge4.set_reliability(0.93)
    # edge5 = Edge(0, 3)
    # edge5.set_reliability(0.96)
    # edge6 = Edge(1, 3)
    # edge6.set_reliability(0.97)
    # edge7 = Edge(3, 5)
    # edge7.set_reliability(0.96)
    # mst = [edge1, edge5, edge2, edge3, edge4, edge7,edge6]
    # pre = []
    # print(test.findGraphReliability(mst, pre, 5, 6))

    edge1 = Edge(0, 1)
    edge1.set_reliability(0.94)
    edge2 = Edge(0, 3)
    edge2.set_reliability(0.96)
    edge3 = Edge(1, 2)
    edge3.set_reliability(0.94)
    edge4 = Edge(2, 5)
    edge4.set_reliability(0.94)
    edge5 = Edge(3, 4)
    edge5.set_reliability(0.93)
    edge6 = Edge(3, 5)
    edge6.set_reliability(0.96)
    edge7 = Edge(1, 3)
    edge7.set_reliability(0.97)
    mst = [edge7, edge2, edge6, edge1, edge3, edge4, edge5]
    pre = []
    print(NetworkDesigner.r_g(mst, pre, 5, 6))


if __name__ == "__main__":
    main()
