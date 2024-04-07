# POTENTIAL SHORTCUT: import networkx as nx
import time
import math
import numpy as np
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
from Edge import Edge

tester = './tester/6_city.txt'


# parse input text file
def read():
    reliability, cost = [], []
    with open(tester, 'r') as file:
        lines = file.readlines()
        num_node = None
        for line in lines:
            if not line.startswith('#') and line.strip():
                num_node = int(line.strip())
                break
        if num_node is None:
            raise ValueError("NUMBER OF NODES UNSPECIFIED IN THE PROVIDED FILE.")
        for line in lines[8: 2 * num_node + 5]:
            if not line.startswith('#') and line.strip():
                reliability.extend(map(float, line.strip().split()))
        for line in lines[- num_node - 2:]:
            if not line.startswith('#') and line.strip():
                cost.extend(map(int, line.strip().split()))
    return [num_node, reliability, cost]


def reorder(num_node, reliabilities, costs, *args):
    """
    SORT EDGES BY THE GIVEN CRITERIA
        :param num_node:      number of nodes in the graph;
        :param reliabilities: list of edge reliabilities;
        :param costs:         list of edge costs;
        Additional arguments specifying the sorting criteria:
            'reliability' to sort by descending reliability (if same, less cost precedes),
            'cost' to sort by ascending cost (if same, larger reliability precedes).
    """
    e = []
    index = 0
    for i in range(num_node):
        for j in range(i + 1, num_node):
            tmp = Edge(i, j)
            tmp.set_reliability(reliabilities[index])
            tmp.set_cost(costs[index])
            e.append(tmp)
            index = index + 1
    if 'reliability' in args:
        e.sort(key=lambda x: (x.reliability, -x.cost), reverse=True)
    elif 'cost' in args:
        e.sort(key=lambda x: (x.cost, -x.reliability))
    return e


# find the Minimum Spanning Tree of the given edges
def kruskal(num_node, sorted_edges):
    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
    nodes, e = set(), []
    parent = {i: i for i in range(num_node)}    # initialize parent dictionary for union-find
    for edge in sorted_edges:
        if len(e) == num_node - 1:
            break
        a, b = edge.get_city_a(), edge.get_city_b()
        if find(a) != find(b):
            union(a, b)
            e.append(edge)
    # TO BE OPTIMIZED: move cost limit check and replacement here
    return e


def r_total(path):
    r = 1
    for e in path:
        r *= e.get_reliability()
    return r


# DOUBLE CHECK: function refined by ChatGPT
def optimizer(e_rest, e_curr, cost, reliability, budget, num_node):
    max_r = 0
    feasible = sum(e.get_cost() for e in e_curr) <= budget
    while budget - cost >= min([edge.cost for edge in e_rest]):
        r_rest, c_rest, ratio, available = [[0] * len(e_rest) for _ in range(4)]
        for i, e in enumerate(e_rest):
            replica = e_curr.copy() + [e]
            c_rest[i] = sum(e.get_cost() for e in replica)
            if c_rest[i] > budget:
                ratio[i] = -1
                continue
            list_perfect = []
            r_rest[i] = reliability_graph(replica, list_perfect, num_node - 1, num_node)
            ratio[i] = r_rest[i] / c_rest[i]
            available[i] = 1 if (budget - sum(e.get_cost() for e in replica) >=
                                 min([edge.cost for edge in e_rest[:i] + e_rest[i + 1:]])) else 0
        max_ratio = max(ratio)
        max_r = max(r_rest)
        r = ratio.index(max_ratio)
        idx = r if r == r_rest.index(max_r) else (r if available[r] == 1 else r_rest.index(max_r))
        max_r = max(reliability, max_r)
        e_curr.append(e_rest[idx])
        cost = sum(e.get_cost() for e in e_curr)
        e_rest.pop(idx)
    return max(max_r, reliability), feasible


def reliability_graph(edges, edges_perfect, num_edge, num_node):
    """
    CALCULATE RELIABILITY OF THE GRAPH RECURSIVELY
        :param edges:         list of all edges in the graph;
        :param edges_perfect: list of edges that are considered reliable under the assumption;
        :param num_edge:      number of edges in the minimum spanning tree;
        :param num_node:      number of nodes in the graph.
    RETURNS: reliability of the graph
    """
    r = 0
    sorted_e = sorted(edges, key=lambda x: x.get_city_a(), reverse=True)
    if len(sorted_e) + len(edges_perfect) == num_edge and connected(sorted_e, edges_perfect, num_node):
        return r_total(edges)
    else:
        if not connected(sorted_e, edges_perfect, num_node):
            return 0
        if len(sorted_e) > 0:
            e = sorted_e[0]
            cloned = sorted_e.copy()
            cloned.remove(e)
            r += (1 - e.get_reliability()) * reliability_graph(cloned, edges_perfect, num_edge, num_node)
            edges_perfect.append(e)
            r += e.get_reliability() * reliability_graph(cloned, edges_perfect, num_edge, num_node)
            return r
        else:
            return 1


def draw(edges, num_node, c):
    angle = 2 * math.pi / num_node
    points, labels = [], []
    for i in range(num_node):
        x, y = math.cos(i * angle), math.sin(i * angle)
        points.append((x, y))
        labels.append(i)
    x, y = zip(*points)  # separate x and y
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='black')
    for i in edges:
        if not isinstance(i, float):
            idx_a, idx_b = int(i.get_city_a()), int(i.get_city_b())
            ax.plot([x[idx_a], x[idx_b]], [y[idx_a], y[idx_b]], marker='o', color='blue')
        else:
            plt.text(-1, -1, f"Under cost {c}, maxR {i:.6f}")
    for i in range(len(labels)):
        ax.text(x[i], y[i], labels[i], fontsize=12)
    ax.axis('off')
    plt.show()


"""
# alternative mathematical approach
def connected(edges, edges_perfect, num_node):
    connectivity = [0] * num_node   # initialize as not connected
    connectivity[0] = 1
    redo = True
    while redo:
        redo = False
        for e in edges + edges_perfect:  # merge sets
            if connectivity[e.get_city_a()] != connectivity[e.get_city_b()]:
                connectivity[e.get_city_a()], connectivity[e.get_city_b()] = 1, 1
                redo = True
    return sum(connectivity) == num_node
"""


def connected(edges, edges_perfect, num_node):
    def dfs(n, g, is_visited):
        """
        DEPTH-FIRST SEARCH TRAVERSAL
            :param n: current node being visited;
            :param g: adjacency list representation of the graph;
            :param is_visited: boolean list to record visited nodes.
        """
        is_visited[n] = True
        for neighbour in g[n]:
            if not is_visited[neighbour]:
                dfs(neighbour, g, is_visited)
    graph = [[] for _ in range(num_node)]
    for e in edges + edges_perfect:
        graph[e.get_city_a()].append(e.get_city_b())
        graph[e.get_city_b()].append(e.get_city_a())
    visited = [False] * num_node
    dfs(0, graph, visited)
    return all(visited)


def main():
    while True:
        try:
            limit = int(input("Please specify cost limit: "))
            assert limit > 0
            break
        except (ValueError, AssertionError) as e:
            print("Invalid input: ", e)
    while True:
        try:
            algo = int(input("Press 1 for exhaustive, 2 for advanced: "))
            assert algo in [1, 2]
            break
        except (ValueError, AssertionError) as e:
            print("Invalid input: ", e)
    num_node, matrix_r, matrix_c = read()
    e_by_r = reorder(num_node, matrix_r, matrix_c, 'reliability')
    e_by_c = reorder(num_node, matrix_r, matrix_c, 'cost')
    if algo == 1:
        start = time.time()
        valid_comb = list()
        pbar = tqdm(total=sum(math.comb(len(e_by_r), i) for i in range(num_node - 1, len(e_by_r) + 1)))
        for e in range(num_node - 1, len(e_by_r) + 1):
            for row in list(combinations(e_by_r, e)):
                cost = sum(item.cost for item in row)
                # TO BE OPTIMIZED: call connected() directly
                checklist = np.zeros(num_node)
                changed = True
                checklist[row[1].get_city_a()] = 1
                checklist[row[1].get_city_b()] = 1
                while changed:
                    replica = checklist[:]
                    for j in row:
                        if checklist[j.get_city_a()] != checklist[j.get_city_b()]:
                            checklist[j.get_city_a()] = 1
                            checklist[j.get_city_b()] = 1
                    if np.all(replica == checklist):
                        changed = False
                if cost <= limit and 0 not in checklist:
                    valid_comb.append(list(row))
                pbar.update(1)
        pbar.close()
        arr = []
        for e in valid_comb:
            list_perfect = []
            e.append(reliability_graph(e, list_perfect, num_node - 1, num_node))
            arr.append(e)
        arr = sorted(arr, key=lambda x: x[-1])
        result = arr[-1] if arr else 0  # non-empty list
        if result == 0:
            print("FEASIBLE SOLUTION NOT POSSIBLE. PROGRAM TERMINATED.")
            return
        print("Under cost limit of %s, max reliability is %s" % (limit, result[-1]))
        print("Runtime for advanced algorithm: %.4f ms" % ((time.time() - start) * 1000))
        draw(result, num_node, limit)
    else:
        start = time.time()
        mst_c = kruskal(num_node, e_by_c)
        mst = kruskal(num_node, e_by_r)
        curr_r = r_total(mst)
        curr_r_c = r_total(mst_c)
        curr_e = mst.copy()
        curr_e_c = mst_c.copy()
        max_r_by_r, r_feasible = optimizer([edge for edge in e_by_r if edge not in mst], curr_e,
                                           sum(e.get_cost() for e in mst), curr_r, limit, num_node)
        max_r_by_c, c_feasible = optimizer([edge for edge in e_by_c if edge not in mst_c], curr_e_c,
                                           sum(e.get_cost() for e in mst_c), curr_r_c, limit, num_node)
        print("Runtime for advanced algorithm: %.4f ms" % ((time.time() - start) * 1000))
        if r_feasible or c_feasible:
            max_r_by_r = max(curr_r, max_r_by_r)
            max_r_by_c = max(curr_r_c, max_r_by_c)
            if max_r_by_r > max_r_by_c:
                print("Under cost limit of %s, max reliability is %s" % (limit, max_r_by_r))
                curr_e.append(max_r_by_r)
                draw(curr_e, num_node, limit)
            else:
                print("Under cost limit of %s, max reliability is %s" % (limit, max_r_by_c))
                curr_e_c.append(max_r_by_c)
                draw(curr_e_c, num_node, limit)
            print("NO MORE IMPROVEMENT")
        else:
            print("FEASIBLE SOLUTION NOT POSSIBLE. PROGRAM TERMINATED.")
            return


if __name__ == "__main__":
    main()
