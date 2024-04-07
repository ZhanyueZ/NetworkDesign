from itertools import combinations
import numpy as np
import NetworkDesigner
import math
import matplotlib.pyplot as plt


def combination(edge, node, costGoal):
    list_combination = list()
    for i in range(node - 1, len(edge) + 1):
        sub_combination = list(combinations(edge, i))
        for j, row in enumerate(sub_combination):
            # check if connected
            Connected = checkConnection(row, node)
            cost = 0  # initial
            for k, item in enumerate(row):  # iterate all element
                cost += item.cost
            if cost <= costGoal and Connected:
                list_combination.append(list(row))
    return list_combination


def checkConnection(edges, node):
    checkList = np.zeros(node)  # list to check if all node connected
    change = True
    checkList[edges[1].get_cityA()] = 1
    checkList[edges[1].get_cityB()] = 1  # initial
    while change:
        copy_checklist = checkList[:]  # copy of array to check connected
        for i in edges:
            if checkList[i.get_cityA()] != checkList[i.get_cityB()]:
                checkList[i.get_cityA()] = 1
                checkList[i.get_cityB()] = 1
        if (copy_checklist == checkList).all():
            change = False
    result = 0 not in checkList
    return result


def exhaustive(edges_list, node):
    sorted_list = []

    for i in edges_list:
        null_list = []
        R_i = NetworkDesigner.graph_reliability(i, null_list, node - 1, node)
        i.append(R_i)
        sorted_list.append(i)
        # print(i[-1])
    sorted_list = sorted(sorted_list, key=lambda x: x[-1])  # sorted using R
    if sorted_list:
        return sorted_list[-1]
    else:
        print("infeasible")
        return 0


def draw(edges, num_node, cost, t):
    """
    GRAPH VISUALIZATION
        :param edges: a list contains all edges and the last element is the reliability under given edges
        :param num_node: number of nodes
        :param cost: total cost
        :param t: runtime
    """
    angle = 2 * math.pi / num_node
    points = []
    labels = []
    for i in range(num_node):
        x = math.cos(i * angle)
        y = math.sin(i * angle)
        points.append((x, y))
        labels.append(i)
    x, y = zip(*points)  # separate x and y
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='black')
    for i in edges:
        if type(i) != float:
            city_a_index = int(i.get_cityA())
            city_b_index = int(i.get_cityB())
            ax.plot([x[city_a_index], x[city_b_index]], [y[city_a_index], y[city_b_index]], marker='o', color='blue')
        else:
            plt.text(-1, -1, f"Under cost {cost}, MaxR={i}, Runtime={t} ms", fontsize=8, color='black')
    for i in range(len(labels)):
        label = labels[i]
        ax.text(x[i], y[i], label, fontsize=12, ha='right', va='bottom')
    ax.axis('off')
    plt.show()
