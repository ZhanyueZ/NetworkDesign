import time
import numpy as np  # TO BE OPTIMIZED: use numpy for faster array operations
from Edge import Edge
import Brute

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
            raise ValueError("NUMBER OF NODES NOT SPECIFIED IN THE PROVIDED FILE.")
        reliability_lines = lines[8: 5 + 2 * num_node]
        for line in reliability_lines:
            if not line.startswith('#') and line.strip():
                reliability.extend(map(float, line.strip().split()))
        cost_lines = lines[- 2 - num_node:]
        for line in cost_lines:
            if not line.startswith('#') and line.strip():
                cost.extend(map(int, line.strip().split()))
    return [num_node, reliability, cost]


# sort the edges by descending reliability / ascending cost; less cost precedes if same reliability
def sort_edge(num_node, reliabilities, costs, *args):
    edges = []
    index = 0
    for i in range(num_node):
        for j in range(i + 1, num_node):
            temp = Edge(i, j)
            temp.set_reliability(reliabilities[index])
            temp.set_cost(costs[index])
            edges.append(temp)
            index = index + 1
    if 'reliability' in args:
        edges.sort(key=lambda x: (x.reliability, -x.cost), reverse=True)
    elif 'cost' in args:
        edges.sort(key=lambda x: (x.cost, -x.reliability))
    else:
        raise ValueError("Valid sorting parameter: 'reliability' or 'cost'.")
    return edges


# find the Minimum Spanning Tree of the given edges
def kruskal(num_node, sortedEdges, cost_limit):
    nodes, edges = set(), []

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    sortedEdges.sort(key=lambda x: x.reliability, reverse=True)
    parent = {i: i for i in range(num_node)}  # initialize parent dictionary for union-find
    for edge in sortedEdges:
        if len(edges) == num_node - 1:
            break
        a, b = edge.get_cityA(), edge.get_cityB()
        if find(a) != find(b):
            union(a, b)
            edges.append(edge)
    mst_cost = sum(edge.cost for edge in edges)
    # TO BE OPTIMIZED: if total cost exceeds limit, replace edge(s) with one(s) with lower reliability
    while mst_cost > cost_limit:
        old = min(edges, key=lambda x: x.reliability)
        unused = [edge for edge in sortedEdges if edge not in edges]
        new = None
        for new_edge in unused:
            if find(new_edge.get_cityA()) != find(new_edge.get_cityB()):
                new = new_edge
                break
        if new is None:
            break
        edges.remove(old)
        edges.append(new)
        mst_cost = mst_cost - old.cost + new.cost
    return edges


def total_reliability(paths):
    r = 1
    for e in paths:
        temp = e.get_reliability()
        r = temp * r
    return r


def total_cost(path):
    c = 0
    for e in path:
        c += e.get_cost()
    return c


def process_edge(remaining_edges, cur_edges, cur_c, cur_r, cost_limit, num_node):
    max_reliability = 0
    feasible = False
    if total_cost(cur_edges) <= cost_limit: feasible = True
    while cost_limit - cur_c >= min([edge.cost for edge in remaining_edges]):
        feasible = True
        r_unadded, c_unadded, ratio, have_space = [[0] * len(remaining_edges) for _ in range(4)]
        for i in range(len(remaining_edges)):
            e = remaining_edges[i]
            clone_list = cur_edges.copy()
            clone_list.append(e)
            c_unadded[i] = total_cost(clone_list)
            if c_unadded[i] > cost_limit:
                ratio[i] = -1
                continue
            perfect_list = []
            r_unadded[i] = graph_reliability(clone_list, perfect_list, num_node - 1, num_node)
            ratio[i] = r_unadded[i] / c_unadded[i]
            remaining_edges_copy = remaining_edges.copy()
            remaining_edges_copy.remove(e)
            have_space[i] = 1 if cost_limit - total_cost(clone_list) >= min(
                [edge.cost for edge in remaining_edges_copy]) else 0
        for i in ratio:
            if i > 0:
                feasible = True
        max_ratio = max(ratio)
        max_reliability = max(r_unadded)
        if ratio.index(max_ratio) == r_unadded.index(max_reliability):
            idx = ratio.index(max_ratio)
        else:
            if have_space[r_unadded.index(max_reliability)] == 1:
                idx = r_unadded.index(max_reliability)
            elif have_space[r_unadded.index(max_reliability)] == 0 and have_space[ratio.index(max_ratio)] == 1:
                idx = ratio.index(max_ratio)
            else:
                idx = r_unadded.index(max_reliability)
        max_reliability = r_unadded[idx]
        max_reliability = max(cur_r, max_reliability)
        cur_edges.append(remaining_edges[idx])
        cur_c = total_cost(cur_edges)
        remaining_edges.remove(remaining_edges[idx])
    return max(max_reliability,cur_r), feasible


def run():
    print("ATTENTION: Beware of your cost estimation before trying a tester.")
    cost_limit = int(input("Please Input Cost Goal: "))
    method = int(input("Press 1 for exhaustive, 2 for advanced: "))
    if method != 1 and method != 2:
        print("INVALID INDEX. PROGRAM TERMINATED.")
        return
    input_value = read()
    num_node = input_value[0]
    all_reliability = input_value[1]
    all_cost = input_value[2]
    feasible = False
    sorted_edges_by_reli = sort_edge(num_node, all_reliability, all_cost, 'reliability')
    sort_edges_cost = sort_edge(num_node, all_reliability, all_cost, 'cost')
    if method == 2:
        start = time.time()
        mst_c = kruskal(num_node, sort_edges_cost, cost_limit)
        mst = kruskal(num_node, sorted_edges_by_reli, cost_limit)
        cur_r = total_reliability(mst)
        cur_r_c = total_reliability(mst_c)
        cur_c = total_cost(mst)
        cur_c_c = total_cost(mst_c)
        cur_e = mst.copy()
        cur_e_c = mst_c.copy()
        remaining_e = [edge for edge in sorted_edges_by_reli if edge not in mst]
        remaining_e_c = [edge for edge in sort_edges_cost if edge not in mst_c]
        max_r_by_r, feasible_reli = process_edge(remaining_e, cur_e, cur_c, cur_r, cost_limit, num_node)
        max_r_by_c, feasible_cost = process_edge(remaining_e_c, cur_e_c, cur_c_c, cur_r_c, cost_limit, num_node)
        runtime = (time.time() - start) * 1000
        print("Runtime for advanced algorithm: ", runtime, "ms")
        if feasible_reli or feasible_cost:
            max_r_by_r = max(cur_r, max_r_by_r)
            max_r_by_c = max(cur_r_c, max_r_by_c)
            if max_r_by_r > max_r_by_c:
                print("Under cost limit of %s, max reliability is %s" % (cost_limit, max_r_by_r))
                cur_e.append(max_r_by_r)
                Brute.draw(cur_e, num_node, cost_limit, runtime)
            else:
                print("Under cost limit of %s, max reliability is %s" % (cost_limit, max_r_by_c))
                cur_e_c.append(max_r_by_c)
                Brute.draw(cur_e_c, num_node, cost_limit, runtime)
            print("No more improvements.")
        else:
            print("No feasible Solution. Quit")
            return
    elif method == 1:
        start = time.time()
        result = Brute.exhaustive(Brute.combination(sorted_edges_by_reli, num_node, cost_limit), num_node)
        if result == 0:
            print("FEASIBLE SOLUTION NOT POSSIBLE. PROGRAM TERMINATED.")
            return
        print("Under cost limit of %s, max reliability is %s" % (cost_limit, result[-1]))
        runtime = (time.time() - start) * 1000
        print("Runtime for exhaustive algorithm: ", runtime, "ms")
        Brute.draw(result, num_node, cost_limit, runtime)


def graph_reliability(Edges, PerfectEdges, numOfEdgesMST, num_node):
    """
    CALCULATE RELIABILITY OF THE GRAPH RECURSIVELY
        :param Edges:         A list of all edges in the graph;
        :param numOfEdgesMST: The number of edges in the minimum spanning tree;
        :param PerfectEdges:  A list of edges that are considered reliable under the assumption;
        :param num_node:      The number of nodes in the graph.
    RETURNS: reliability of the graph
    """
    r = 0
    sortedEdges = sorted(Edges, key=lambda x: x.reliability, reverse=True)
    if len(sortedEdges) + len(PerfectEdges) == numOfEdgesMST and connected(sortedEdges, PerfectEdges, num_node):
        return total_reliability(Edges)
    else:
        if not connected(sortedEdges, PerfectEdges, num_node):
            return 0
        if len(sortedEdges) > 0:
            e = sortedEdges[0]
            cloned_edges = sortedEdges.copy()
            cloned_edges.remove(e)
            r += (1 - e.get_reliability()) * graph_reliability(cloned_edges, PerfectEdges, numOfEdgesMST, num_node)
            PerfectEdges.append(e)
            r += e.get_reliability() * graph_reliability(cloned_edges, PerfectEdges, numOfEdgesMST, num_node)
            return r
        else:
            return 1


# TO BE OPTIMIZED:
def connected(Edges, PerfectEdges, num_node):
    connectivity = [0] * num_node  # initialize as not connected
    connectivity[0] = 1
    redo = True
    while redo:
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
    run()
