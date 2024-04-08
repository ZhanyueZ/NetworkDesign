# POTENTIAL SHORTCUT: import networkx as nx
import time
import math
import numpy as np  # TO BE OPTIMIZED: apply more numpy arrays
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
from Edge import Edge

tester = './tester/6_city.txt'


def reorder(num_node, reliabilities, costs, *args):
    """
    SORT EDGES BY THE GIVEN CRITERIA
        :param num_node:      number of nodes in the graph;
        :param reliabilities: list of edge reliabilities;
        :param costs:         list of edge costs;
        Additional argument specifying the sorting preference:
            'reliability' to sort by descending reliability (if same, less cost precedes),
            'cost' to sort by ascending cost (if same, larger reliability precedes).
    """
    edges = []
    idx = 0
    for i in range(num_node):
        for j in range(i + 1, num_node):
            e = Edge(i, j)
            e.set_reliability(reliabilities[idx])
            e.set_cost(costs[idx])
            edges.append(e)
            idx = idx + 1
    if 'reliability' in args:
        edges.sort(key=lambda x: (x.reliability, -x.cost), reverse=True)
    elif 'cost' in args:
        edges.sort(key=lambda x: (x.cost, -x.reliability))
    return edges


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
    parent = {i: i for i in range(num_node)}  # initialize parent dictionary for union-find
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


def optimizer(e_rest, e_curr, cost, reliability, budget, num_node):
    r_max = 0
    feasible = sum(e.get_cost() for e in e_curr) <= budget
    while budget - cost >= min([edge.cost for edge in e_rest]):
        r_rest, c_rest, ratio, available = [[0] * len(e_rest) for _ in range(4)]
        for i, e in enumerate(e_rest):
            replica = e_curr.copy() + [e]
            c_rest[i] = sum(e.get_cost() for e in replica)
            if c_rest[i] > budget:
                ratio[i] = -1
                continue
            list_reliable = []
            r_rest[i] = r_g(replica, list_reliable, num_node - 1, num_node)
            ratio[i] = r_rest[i] / c_rest[i]
            available[i] = 1 if (budget - sum(e.get_cost() for e in replica) >=
                                 min([edge.cost for edge in e_rest[:i] + e_rest[i + 1:]])) else 0
        r_max = max(r_rest)
        r = ratio.index(max(ratio))
        idx = r if r == r_rest.index(r_max) else (r if available[r] == 1 else r_rest.index(r_max))
        r_max = max(reliability, r_max)
        e_curr.append(e_rest[idx])
        cost = sum(e.get_cost() for e in e_curr)
        e_rest.pop(idx)
    return r_max, feasible


def r_g(edges, reliable, num_edge, num_node):
    """
    CALCULATE RELIABILITY OF THE GRAPH RECURSIVELY
        :param edges:    list of all edges in the graph;
        :param reliable: list of edges that are considered reliable under the assumption;
        :param num_edge: number of edges in the minimum spanning tree;
        :param num_node: number of nodes in the graph.
    RETURNS: reliability of the graph
    """
    sorted_e = sorted(edges, key=lambda x: x.get_city_a(), reverse=True)
    if len(sorted_e) + len(reliable) == num_edge and connected(sorted_e, reliable, num_node):
        return r_total(edges)
    else:
        if not connected(sorted_e, reliable, num_node):
            return 0
        if len(sorted_e) > 0:
            r = 0
            e = sorted_e[0]
            cloned = sorted_e.copy()
            cloned.remove(e)
            r += (1 - e.get_reliability()) * r_g(cloned, reliable, num_edge, num_node)
            reliable.append(e)
            r += e.get_reliability() * r_g(cloned, reliable, num_edge, num_node)
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
            plt.text(-1, -1, f"Under cost {c}, R_max {i:.6f}")
    for i in range(len(labels)):
        ax.text(x[i], y[i], labels[i], fontsize=12)
    ax.axis('off')
    plt.show()


"""
# alternative mathematical approach
def connected(edges, reliable, num_node):
    connectivity = [0] * num_node   # initialize as not connected
    connectivity[0] = 1
    redo = True
    while redo:
        redo = False
        for e in edges + reliable:  # merge sets
            if connectivity[e.get_city_a()] != connectivity[e.get_city_b()]:
                connectivity[e.get_city_a()], connectivity[e.get_city_b()] = 1, 1
                redo = True
    return sum(connectivity) == num_node
"""


def connected(edges, reliable, num_node):
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
    for e in edges + reliable:
        graph[e.get_city_a()].append(e.get_city_b())
        graph[e.get_city_b()].append(e.get_city_a())
    visited = [False] * num_node
    dfs(0, graph, visited)
    return all(visited)


def main():
    # requirements validation
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
    # parse input text file
    matrix_r, matrix_c = [], []
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
                matrix_r.extend(map(float, line.strip().split()))
        for line in lines[- num_node - 2:]:
            if not line.startswith('#') and line.strip():
                matrix_c.extend(map(int, line.strip().split()))
    e_r = reorder(num_node, matrix_r, matrix_c, 'reliability')
    e_c = reorder(num_node, matrix_r, matrix_c, 'cost')
    if algo == 1:
        start = time.time()
        valid_comb = list()
        pbar = tqdm(total=sum(math.comb(len(e_r), i) for i in range(num_node - 1, len(e_r) + 1)))
        for e in range(num_node - 1, len(e_r) + 1):
            for row in list(combinations(e_r, e)):
                matrix_c = sum(item.cost for item in row)
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
                if matrix_c <= limit and 0 not in checklist:
                    valid_comb.append(list(row))
                pbar.update(1)
        pbar.close()
        arr = []
        for e in valid_comb:
            list_reliable = []
            e.append(r_g(e, list_reliable, num_node - 1, num_node))
            arr.append(e)
        arr = sorted(arr, key=lambda x: x[-1])
        result = arr[-1] if arr else 0  # non-empty list
        if result == 0:
            print("FEASIBLE SOLUTION NOT POSSIBLE. PROGRAM TERMINATED.")
            return
        print("Under cost limit of %s, max reliability is %s" % (limit, result[-1]))
        print("Runtime for exhaustive algorithm: %.4f ms" % ((time.time() - start) * 1000))
        draw(result, num_node, limit)
    else:
        # TO BE OPTIMIZED: reduce redundancy
        # reliability-greedy part
        start = time.time()
        mst = kruskal(num_node, e_r)
        e_curr = mst.copy()
        r_curr = r_total(mst)
        r_max, r_feasible = optimizer([edge for edge in e_r if edge not in mst], e_curr,
                                      sum(e.get_cost() for e in mst), r_curr, limit, num_node)
        # cost-greedy part
        mst_c = kruskal(num_node, e_c)
        e_curr_c = mst_c.copy()
        r_curr_c = r_total(mst_c)
        r_max_c, c_feasible = optimizer([edge for edge in e_c if edge not in mst_c], e_curr_c,
                                        sum(e.get_cost() for e in mst_c), r_curr_c, limit, num_node)
        print("Runtime for advanced algorithm: %.4f ms" % ((time.time() - start) * 1000))
        if r_feasible or c_feasible:
            r_max = max(r_curr, r_max)
            r_max_c = max(r_curr_c, r_max_c)
            if r_max > r_max_c:
                print(f"Under cost limit of {limit}, max reliability is {r_max}")
                e_curr.append(r_max)
                draw(e_curr, num_node, limit)
            else:
                print(f"Under cost limit of {limit}, max reliability is {r_max_c}")
                e_curr_c.append(r_max_c)
                draw(e_curr_c, num_node, limit)
            print("NO MORE IMPROVEMENT")
        else:
            print("FEASIBLE SOLUTION NOT POSSIBLE. PROGRAM TERMINATED.")
            return


if __name__ == "__main__":
    main()
