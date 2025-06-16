from dspgToLuxTest import Graph
import numpy as np
import time
from matplotlib import pyplot as plt
import pickle


def GetAllPaths(graph: Graph):
    # Returns a list of all paths(as graph) from source to the target
    source = graph.source
    target = graph.target
    adj_matrix = graph.adj

    def dfs(node, path):
        if node == target:
            all_paths_list.append(path)
            return
        visited[node] = True
        for neighbor in range(len(adj_matrix)):
            if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor, path + [neighbor])
        visited[node] = False

    all_paths_list = []
    visited = [False] * len(adj_matrix)
    dfs(source, [source])
    res = []
    for path in all_paths_list:
        graph_path = Graph(mat=len(graph.name))
        for i in range(len(path)-1):
            graph_path.add_edge_new(path[i], path[i+1])
        res.append(graph_path)
    return res


def find_path(synthesizing_graph: Graph, node_src, node_target):
    # returns path from node_src to node_target if exists in synthesizing_graph
    # otherwise returns None
    tmp = synthesizing_graph.check_path(
        node_src, node_target, synthesizing_graph.adj)
    if (len(tmp) == 0):
        return None
    return tmp


def nodes_between(adj_matrix, start_node, end_node):
    def dfs(node, visited):
        visited[node] = True
        for neighbor in range(len(adj_matrix)):
            if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                nodes_between_list.append(neighbor)
                dfs(neighbor, visited)

    if start_node < 0 or start_node >= len(adj_matrix) or end_node < 0 or end_node >= len(adj_matrix):
        return []

    nodes_between_list = []
    visited = [False] * len(adj_matrix)

    dfs(start_node, visited)

    if visited[end_node]:
        return nodes_between_list
    else:
        return []


def contains(graph: Graph, key, node):
    # returns True if node is source, target or intermediate node of pps or if pps=None
    if key not in graph.nodes_to_pps.keys():
        return True
    pps = graph.nodes_to_pps[key]
    if (node == pps[0] or node == pps[1]):
        return True
    tmp = nodes_between(graph.adj, pps[0], pps[1])
    if (node in tmp):
        return True
    return False

 # Global Variable to update in to_dspg function


def dspg_edge_check(synthesizing_graph: Graph, node_src, node_target):
    path = find_path(synthesizing_graph, node_src, node_target)
    if (path == None):
        return None
    if (not contains(synthesizing_graph, node_src, node_target)):
        return None
    if (not contains(synthesizing_graph, node_target, node_src)):
        return None
    return path  # .intermediate_nodes # intermediate nodes are nodes which are not source or target


def getNodes(processing_edges):
    # Returns a list of all intermediate nodes in processing_edges
    n = len(processing_edges)
    return [processing_edges[i][0] for i in range(n-1)]


def to_dspg(graph):
    path_to_process = GetAllPaths(graph)
    path_to_process.sort(key=lambda x: len(x.edges))
    # print([len(path_to_process[i].edges) for i in range(len(path_to_process))])
    # print(len(path_to_process))
    # path_to_process.sort() #length smaller to bigges
    synthesizing_graph = path_to_process.pop()
    synthesizing_graph.nodes_to_pps = {}
    while (len(path_to_process) != 0):
        # if(len(path_to_process)%1000==0):
        #     print(len(path_to_process)//1000)
        processing_path = path_to_process.pop()
        processing_path.index = 0
        processing_edges = []
        end_adding_path = False
        while (not end_adding_path):
            # edge = processing_path.next()
            edge = processing_path.edges[processing_path.index]
            processing_path.index += 1

            # end_adding_path = processing_path.no_more_edge()
            if (processing_path.index == len(processing_path.edges)):
                end_adding_path = True

            # if(edge.sink not in synthesizing_graph.nodes):
            if (edge[1] not in synthesizing_graph.nodes):
                processing_edges.append(edge)
            elif (len(processing_edges) != 0 or edge not in synthesizing_graph.edges):
                processing_edges.append(edge)
                node_src = processing_edges[0][0]
                node_target = processing_edges[-1][1]
                intermediate_nodes = dspg_edge_check(
                    synthesizing_graph, node_src, node_target)
                if (intermediate_nodes != None):
                    synthesizing_graph.add_edges(processing_edges)
                    for node in intermediate_nodes:
                        # if node!= graph.source and node!= graph.target:
                            synthesizing_graph.nodes_to_pps[node] = (
                            node_src, node_target)
                    for node in getNodes(processing_edges):
                        synthesizing_graph.nodes_to_pps[node] = (
                            node_src, node_target)
                    processing_edges = []
                else:
                    end_adding_path = True
    return synthesizing_graph


def graph11():
    Gr = Graph(mat=6)
    Gr.add_edge_new(1, 3)
    Gr.add_edge_new(3, 4)
    Gr.add_edge_new(4, 2)

    Gr.add_edge_new(3, 2)
    Gr.add_edge_new(1, 5)

    Gr.add_edge_new(5, 2)
    Gr.add_edge_new(5, 4)
    
    Gr.source = 1
    Gr.target = 2
    return Gr 

def graph2():
    Gr = Graph(mat=9)
    Gr.add_edge_new(1, 8)
    Gr.add_edge_new(1, 4)
    Gr.add_edge_new(1, 3)
    Gr.add_edge_new(1, 7)
    Gr.add_edge_new(1, 2)
    Gr.add_edge_new(1, 6)

    Gr.add_edge_new(3, 2)
    Gr.add_edge_new(3, 5)

    Gr.add_edge_new(4, 6)
    Gr.add_edge_new(4, 3)

    Gr.add_edge_new(5, 2)

    Gr.add_edge_new(6, 3)
    Gr.add_edge_new(6, 5)
    Gr.add_edge_new(6, 7)

    Gr.add_edge_new(7, 2)
    Gr.add_edge_new(7, 5)

    Gr.add_edge_new(8, 4)
    Gr.add_edge_new(8, 3)
    Gr.add_edge_new(8, 6)
    Gr.add_edge_new(8, 7)

    Gr.source = 1
    Gr.target = 2
    return Gr


def test():
    Gr = graph2()
    Gr.show()
    print(len(Gr.edges))
    print(Gr.adj)
    s = to_dspg(Gr)
    print(s.adj)
    print(len(s.edges))
    s.show()


def test_data(directory, delimiter=None):
    from os import listdir
    from tqdm import tqdm
    g_files = listdir(directory)
    # for i in tqdm(g_files):
    table = []
    for i in g_files:
        print("processing >>>", i)
        fname = directory+'/'+str(i)
        s, gr, duration = process(fname, str(i), delimiter=delimiter)
        # print(len(gr.edges))
        # print(len(s.edges)/len(gr.edges))
        table.append({"n_edges": len(gr.edges), "compression_rate": len(
            s.edges)/len(gr.edges), "time": duration})

    print(table)
    file = open('important', 'wb')
    pickle.dump(table, file)
    file.close()


def process(filename, alias, delimiter=None):
    Gr = read_graph_file(filename, delimiter=delimiter)
    debut = time.time()
    s = to_dspg(Gr)
    duration = time.time() - debut
    tmp = s.adj.astype(int)
    np.savetxt('output_path/'+alias, tmp,
               delimiter='\t', newline='\n', fmt='%d')
    return s, Gr, duration


def read_graph_file(filename, delimiter=None):
    mat = np.loadtxt(filename, delimiter=delimiter)
    # print(mat)
    t = np.where(np.sum(mat, axis=1) == 0)[0][0]
    s = np.where(np.sum(mat, axis=0) == 0)[0][0]
    n = np.shape(mat)[0]
    Gr = Graph(n)
    for i in range(n):
        for j in range(n):
            if (mat[i][j] > 0):
                Gr.add_edge_new(i, j)

    Gr.source = s
    Gr.target = t
    # Gr.show()
    return Gr


if __name__ == '__main__':
    # test_data('./data')
    G = graph11()
    G.show()
    a = to_dspg(G)
    a.show()
    a.source = 1
    a.target = 2
    tt = to_dspg(a)
    tt.show()

   
    