# OPTIMIZATIONS:
# 1. upgrade data structures, e.g. numpy, set
# 2. exhaustive: multiprocessing / range limitation
# 3. efficient: budget check + replacement / early termination
# 4. import networkx for connectivity and mst

import math
import time
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm import tqdm
from Edge import Edge

path = './tester/5_city.txt'
NUM_NODE = 5


def connected(edges):
    def dfs(n, g, visited):
        """
        DEPTH-FIRST SEARCH TRAVERSAL
            :param n:       current node being visited;
            :param g:       adjacency list representation of the graph;
            :param visited: boolean list to record visited nodes.
        """
        visited[n] = True
        for neighbour in g[n]:
            if not visited[neighbour]:
                dfs(neighbour, g, visited)

    graph = [[] for _ in range(NUM_NODE)]
    is_visited = [False] * NUM_NODE
    for e in edges:
        graph[e.get_city_a()].append(e.get_city_b())
        graph[e.get_city_b()].append(e.get_city_a())
    dfs(0, graph, is_visited)
    return all(is_visited)


# RETURNS RELIABILITY OF THE GRAPH RECURSIVELY
def r_g(edges, reliable):
    e_sorted = sorted(edges, key=lambda x: x.get_city_a(), reverse=True)
    if len(e_sorted) + len(reliable) == NUM_NODE - 1 and connected(e_sorted + reliable):
        return math.prod(e.get_reliability() for e in edges)
    else:
        if not connected(e_sorted + reliable):
            return 0
        if len(e_sorted) > 0:
            r = 0
            e = e_sorted[0]
            cloned = e_sorted.copy()
            cloned.remove(e)
            r += (1 - e.get_reliability()) * r_g(cloned, reliable)
            reliable.append(e)
            r += e.get_reliability() * r_g(cloned, reliable)
            return r
        else:
            return 1


def draw(edges, c, title='ADVANCED'):
    """last element of edges is graph reliability"""
    angle = 2 * math.pi / NUM_NODE
    points, labels = [], []
    for i in range(NUM_NODE):
        x, y = math.cos(i * angle), math.sin(i * angle)
        points.append((x, y))
        labels.append(i + 1)
    x, y = zip(*points)  # separate x & y
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i in edges[:-1]:
        idx_a, idx_b = i.get_city_a(), i.get_city_b()
        ax.plot([x[idx_a], x[idx_b]], [y[idx_a], y[idx_b]], marker='o', color='blue')
    plt.text(-1, -1, f"Under cost {c}, R_max {edges[-1]:.6f}")
    for i in range(len(labels)):
        ax.text(x[i], y[i], labels[i], fontsize=12)
    ax.axis('off')
    ax.set_title(title)
    plt.show()


def optimizer(edges, budget):
    # KRUSKAL'S: MINIMUM SPANNING TREE WITH DISJOINT-SET DATA STRUCTURE, by ChatGPT
    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    mst = []
    parent = {i: i for i in range(NUM_NODE)}  # init parent dictionary
    for e in edges:
        if len(mst) == NUM_NODE - 1:
            break
        root_a, root_b = find(e.get_city_a()), find(e.get_city_b())
        if root_a != root_b:
            parent[root_a] = root_b
            mst.append(e)
    e_curr = mst.copy()
    cost = sum(e.get_cost() for e in mst)
    e_rest = [e for e in edges if e not in mst]
    r_max = 0
    feasible = sum(e.get_cost() for e in e_curr) <= budget
    while budget - cost >= min([e.cost for e in e_rest]):
        r_rest, c_rest, ratio, available = [[0] * len(e_rest) for _ in range(4)]
        for i, e in enumerate(e_rest):
            replica = e_curr.copy() + [e]
            c_rest[i] = sum(e.get_cost() for e in replica)
            if c_rest[i] > budget:
                ratio[i] = -1
                continue
            r_rest[i] = r_g(replica, [])
            ratio[i] = r_rest[i] / c_rest[i]
            available[i] = 1 if (budget - sum(e.get_cost() for e in replica) >=
                                 min([e.cost for e in e_rest[:i] + e_rest[i + 1:]])) else 0
        r_max = max(r_rest)
        r = ratio.index(max(ratio))
        idx = r if r == r_rest.index(r_max) else (r if available[r] == 1 else r_rest.index(r_max))
        r_max = max(math.prod(e.get_reliability() for e in mst), r_max)
        e_curr.append(e_rest[idx])
        cost = sum(e.get_cost() for e in e_curr)
        e_rest.pop(idx)
    return e_curr, r_max, feasible


def main():
    # 1. REQUIREMENT VALIDATION
    while True:
        try:
            budget = int(input("Please specify cost limit: "))
            assert budget > 0
            break
        except (ValueError, AssertionError) as err:
            print("Invalid input: ", err)

    # 2. PARSE TESTER TEXT
    edges, matrix_r, matrix_c = [], [], []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines[8: 2 * NUM_NODE + 5]:
            if not line.startswith('#') and line.strip():
                matrix_r.extend(map(float, line.split()))
        for line in lines[- NUM_NODE - 2:]:
            if not line.startswith('#') and line.strip():
                matrix_c.extend(map(int, line.split()))
    for i in range(NUM_NODE):
        for j in range(i + 1, NUM_NODE):
            idx = i * NUM_NODE - i * (i + 3) // 2 + j - 1   # or just increment
            e = Edge(i, j)
            e.set_reliability(matrix_r[idx])
            e.set_cost(matrix_c[idx])
            edges.append(e)
    l = len(edges)  # number of distinct city pairs <=> NUM_NODE * (NUM_NODE - 1) // 2
    e_r = sorted(edges, key=lambda x: (x.reliability, -x.cost), reverse=True)
    e_c = sorted(edges, key=lambda x: (x.cost, -x.reliability))

    # 3. ADVANCED ALGO
    start1 = time.time()
    mst, r_max, r_feasible = optimizer(e_r, budget)        # reliability-greedy part
    mst_c, r_max_c, c_feasible = optimizer(e_c, budget)    # cost-greedy part
    if r_feasible or c_feasible:
        rt1 = (time.time() - start1) * 1000
        print(f"Runtime for guided search: {rt1:.4f} ms\nNO FURTHER IMPROVEMENTS\n")
        draw(mst + [r_max] if r_max > r_max_c else mst_c + [r_max_c], budget)
    else:
        raise ValueError("INFEASIBLE CASE. PROGRAM TERMINATED.")

    # 4. SIMPLE ALGO
    start2 = time.time()
    valid_combs = []
    pbar = tqdm(total=sum(math.comb(l, i) for i in range(NUM_NODE - 1, l + 1)))
    for k in range(NUM_NODE - 1, l + 1):
        for comb in combinations(e_c, k):
            if sum(e.cost for e in comb) <= budget and connected(comb):
                valid_combs.append(list(comb) + [r_g(comb, [])])
            pbar.update(1)
    pbar.close()
    valid_combs.sort(key=lambda x: x[-1])
    if valid_combs:  # expecting feasible
        rt2 = (time.time() - start2) * 1000
        print(f"Runtime for exhaustive search: {rt2:.4f} ms\n"
              f"Relatively {((rt2 - rt1) / rt1 * 100):.2f}% slower" if rt1 != 0
              else "SIMPLE NETWORK. MACRO PERFORMANCE DISCREPANCY NEGLIGIBLE.")
        draw(valid_combs[-1], budget, 'SIMPLE')
    else:
        raise ValueError("INFEASIBLE CASE. PROGRAM TERMINATED.")


if __name__ == "__main__":
    main()
