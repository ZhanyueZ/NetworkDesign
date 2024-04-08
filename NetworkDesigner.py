# POTENTIAL SHORTCUT: import networkx as nx
import math
import time
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm import tqdm
from Edge import Edge

path = './tester/6_city.txt'


def reorder(num_node, reliabilities, costs, criteria='r'):
    edges = []
    idx = 0
    for i in range(num_node):
        for j in range(i + 1, num_node):
            e = Edge(i, j)
            e.set_reliability(reliabilities[idx])
            e.set_cost(costs[idx])
            edges.append(e)
            idx = idx + 1
    if criteria == 'r':
        edges.sort(key=lambda x: (x.reliability, -x.cost), reverse=True)
    elif criteria == 'cost':
        edges.sort(key=lambda x: (x.cost, -x.reliability))
    return edges


# find Minimum Spanning Tree of the given edges, utilizing disjoint-set data structure
def kruskal(num_node, sorted_e):
    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
    edges = []
    parent = {i: i for i in range(num_node)}  # initialize parent dictionary
    for e in sorted_e:
        if len(edges) == num_node - 1:
            break
        a, b = e.get_city_a(), e.get_city_b()
        if find(a) != find(b):
            union(a, b)
            edges.append(e)
    # TO BE OPTIMIZED: cost limit check and replacement / early termination
    return edges


# alternative mathematical approach
"""
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


def connected(edges, num_node):
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
    for e in edges:
        graph[e.get_city_a()].append(e.get_city_b())
        graph[e.get_city_b()].append(e.get_city_a())
    visited = [False] * num_node
    dfs(0, graph, visited)
    return all(visited)


def r_total(path):
    r = 1
    for e in path:
        r *= e.get_reliability()
    return r


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
    if len(sorted_e) + len(reliable) == num_edge and connected(sorted_e + reliable, num_node):
        return r_total(edges)
    else:
        if not connected(sorted_e + reliable, num_node):
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


def draw(edges, num_node, c, title='ADVANCED'):
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
    ax.set_title(title)
    plt.show()


def optimizer(num_node, e_rest, e_curr, reliability, cost, budget):
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
            r_rest[i] = r_g(replica, [], num_node - 1, num_node)
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


def main():
    # 1. REQUIREMENT VALIDATION
    while True:
        try:
            budget = int(input("Please specify cost limit: "))
            assert budget > 0
            break
        except (ValueError, AssertionError) as err:
            print("Invalid input: ", err)

    # 2. PARSE INPUT TEXT FILE
    matrix_r, matrix_c = [], []
    with open(path, 'r') as file:
        lines = file.readlines()
        num_node = None
        for line in lines:  # expecting number of nodes at the first non-comment appearance
            if not line.startswith('#') and line.strip():
                num_node = int(line)
                break
        if num_node is None or num_node <= 0:
            raise ValueError("VOID NUMBER OF NODES IN THE PROVIDED FILE.")
        for line in lines[8: 2 * num_node + 5]:
            if not line.startswith('#') and line.strip():
                matrix_r.extend(map(float, line.split()))
        for line in lines[- num_node - 2:]:
            if not line.startswith('#') and line.strip():
                matrix_c.extend(map(int, line.split()))
    e_r = reorder(num_node, matrix_r, matrix_c)
    e_c = reorder(num_node, matrix_r, matrix_c, 'cost')

    # 3. ADVANCED ALGORITHM
    # reliability-greedy part
    start1 = time.time()
    mst = kruskal(num_node, e_r)
    e_curr = mst.copy()
    r_max, r_feasible = optimizer(num_node, [edge for edge in e_r if edge not in mst], e_curr,
                                  r_total(mst), sum(e.get_cost() for e in mst), budget)
    # cost-greedy part
    mst_c = kruskal(num_node, e_c)
    e_curr_c = mst_c.copy()
    r_max_c, c_feasible = optimizer(num_node, [edge for edge in e_c if edge not in mst_c], e_curr_c,
                                    r_total(mst_c), sum(e.get_cost() for e in mst_c), budget)
    if r_feasible or c_feasible:
        rt1 = (time.time() - start1) * 1000
        if r_max > r_max_c:
            e_curr.append(r_max)
            draw(e_curr, num_node, budget)
        else:
            e_curr_c.append(r_max_c)
            draw(e_curr_c, num_node, budget)
        print(f"Runtime for guided search: {rt1:.4f} ms\nNO FURTHER IMPROVEMENTS\n")
    else:
        print("INFEASIBLE CASE. PROGRAM TERMINATED.")
        return

    # 4. SIMPLE ALGORITHM
    start2 = time.time()
    valid_comb = list()
    pbar = tqdm(total=sum(math.comb(len(e_r), i) for i in range(num_node - 1, len(e_r) + 1)))
    for i in range(num_node - 1, len(e_r) + 1):
        for row in list(combinations(e_r, i)):
            if sum(item.cost for item in row) <= budget and connected(list(row), num_node):
                valid_comb.append(list(row))
            pbar.update(1)
    pbar.close()
    arr = []
    for e in valid_comb:
        e.append(r_g(e, [], num_node - 1, num_node))
        arr.append(e)
    arr = sorted(arr, key=lambda x: x[-1])
    result = arr[-1] if arr else 0  # non-empty list
    if result == 0:
        print("INFEASIBLE CASE. PROGRAM TERMINATED.")
        return
    rt2 = (time.time() - start2) * 1000
    print(f"Runtime for exhaustive search: {rt2:.4f} ms")
    if rt1 != 0:
        print("Relative speed disadvantage: %.2f%%" % ((rt1 - rt2) / rt1 * 100))
    else:
        print("NETWORK TOO SIMPLE. MACRO PERFORMANCE DISCREPANCY NEGLIGIBLE.")
    draw(result, num_node, budget, 'SIMPLE')


if __name__ == "__main__":
    main()
