import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from collections import deque


def dfs(graph, start, end, visited, path):
    visited[start] = True
    path.append(start)
    if start == end:
        return True
    for neighbor, connected in enumerate(graph[start]):
        if connected and not visited[neighbor]:
            if dfs(graph, neighbor, end, visited, path):
                return True
    path.pop()
    return False


def find_path(adj_matrix, start_node, end_node):
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    path = []
    if start_node < 0 or start_node >= num_nodes or end_node < 0 or end_node >= num_nodes:
        return []
    if dfs(adj_matrix, start_node, end_node, visited, path):
        return path
    else:
        return []


class Graph:
   

    def add_edge_new(self, a, b, v=None):
        if(a not in self.nodes):
            self.nodes.append(a)
        if(b not in self.nodes): 
            self.nodes.append(b)

        self.edges.append([a, b])
        if (v == None):
            # self.adj[a][b] = random.randint(1, 10)
            self.adj[a][b] = 1
        else:
            self.adj[a][b] = v
        return
    def add_edges(self, edges):
        for edge in edges:
            a = edge[0]
            b = edge[1]
            self.add_edge_new(a, b)

    def __init__(self, mat, name=None) -> None:


        self.adj = np.zeros((mat, mat))
        self.edges = []
        self.name = [str(i) for i in range(mat)]
        self.name = list(range(mat))
        self.nodes=[]
        # ##Check:
        # n = len(name)
        # assert np.shape(mat)==(n,n)

        # self.adj = mat
        # self.name = name
        # self.G = None
        # self.source = name[0]
        # self.target = name[-1]
        # self.edges = []
        # self.inb = None
        # self.outb = None
        # self.edges_nesting = None
        # self.nodes_nesting = None
        # self.path = {}
        # for i in range(n):
        #     for j in range(n):
        #         if(self.adj[i][j] !=0):
        #             self.edges.append([self.name[i], self.name[j]])

    def add_node(self, a):
        n = len(self.name)
        self.name.append(a)
        self.adj = np.append(self.adj, np.zeros((1, n)), axis=0)
        self.adj = np.append(self.adj, np.zeros((n+1, 1)), axis=1)
        # self.adj = np.pad(self.adj, ((1,1)), 'constant', constant_values=(0))

    def add_edge(self, a, b, v=None):
        self.edges.append([a, b])
        a = self.name.index(a)
        b = self.name.index(b)
        if (v == None):
            # self.adj[a][b] = random.randint(1, 10)
            self.adj[a][b] = 1
        else:
            self.adj[a][b] = v

        return

    def to_graph(self, with_nesting=False):
        self.G = nx.Graph()

        n = np.shape(self.adj)[0]
        for i in range(n):
            for j in range(n):
                if (self.adj[i][j] > 0):
                    a = self.name[i]
                    b = self.name[j]
                    # print(i, a)
                    # print(j, b)
                    if(with_nesting):
                        self.G.add_edge(a, b, weight=self.adj[i][j], nesting=self.edges_nesting[str([self.name[i], self.name[j]])])
                    else:
                        self.G.add_edge(a, b, weight=self.adj[i][j])

                    

                    # self.G.edges[(a,b)]["label"] = str(self.adj[i][j])
        return

    def show(self, with_nesting=False):
        # if(self.G==None):
        self.to_graph(with_nesting)
        # nx.draw(self.G, arrows=True)
        options = {
            # # 'node_color': 'blue',
            # # 'node_size': 100,
            # # 'width': 3,
            # 'arrowstyle': '-|>',
            # 'arrowsize': 10,
        }

        pos = nx.spring_layout(self.G)

        nx.draw_networkx(self.G, pos, with_labels=True, **options)
        nx.draw_networkx_edges(self.G, pos, self.edges, arrows=True, arrowstyle= '-|>', arrowsize= 10, )
        if(with_nesting):
            labels = nx.get_edge_attributes(self.G, 'nesting')
        else:
            labels = nx.get_edge_attributes(self.G, 'weight')

        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
        plt.show()

    def compute_inout(self):
        self.inb = [-1]*len(self.name)
        self.outb = [-1]*len(self.name)
        for i in range(len(self.name)):
            self.outb[i] = len(np.where(self.adj[i] > 0)[0])
        for i in range(len(self.name)):
            self.inb[i] = len(np.where(self.adj[:, i] > 0)[0])

    def compute_edges_nesting(self):
        self.edges_nesting = {}
        self.edges_nesting[str(['A', 'B'])] = 0
        self.edges_nesting[str(['B', 'C'])] = 0
        self.edges_nesting[str(['C', 'F'])] = 1
        self.edges_nesting[str(['F', 'X'])] = 1

        self.edges_nesting[str(['C', 'G'])] = 2
        self.edges_nesting[str(['G', 'J'])] = 2
        self.edges_nesting[str(['J', 'X'])] = 1

        self.edges_nesting[str(['C', 'E'])] = 2
        self.edges_nesting[str(['E', 'J'])] = 3

        self.edges_nesting[str(['E', 'H'])] = 3
        self.edges_nesting[str(['H', 'J'])] = 3

    def compute_vertices_nesting_rec(self, curr):
        # print(np.where(self.adj[curr]>0))
        for i in np.where(self.adj[curr] > 0)[0]:
            if (self.inb[i] > 0):
                self.vertices_nesting[i] = self.edges_nesting[str(
                    [self.name[curr], self.name[i]])]
            elif (self.outb[i] > 0):
                tmp = np.where(self.adj[i] > 0)[0][0]
                self.vertices_nesting[i] = self.edges_nesting[str(
                    [self.name[i], self.name[tmp]])]
            elif (self.outb[i] == 0):
                self.vertices_nesting[i] = 0
            else:

                print([self.edges_nesting[str([self.name[i], self.name[tmp]])]
                      for tmp in np.where(self.adj[i] > 0)[0]])
                self.vertices_nesting[i] = np.min([self.edges_nesting[str(
                    [self.name[i], self.name[tmp]])] for tmp in np.where(self.adj[i] > 0)[0]])-1
            self.compute_vertices_nesting_rec(i)

    def compute_vertices_nesting(self):
        self.vertices_nesting = [-1]*len(self.name)
        self.vertices_nesting[0] = 0
        curr = 0
        if (self.inb == None):
            self.compute_inout()
        if (self.edges_nesting == None):
            self.compute_edges_nesting()
        self.compute_vertices_nesting_rec(curr)
        print(self.vertices_nesting)

    def dspg_first_path(self, adj_to_process, edges_to_process, verbose=False):
        n = np.shape(self.adj)[0]
        
        curr_graph = Graph(np.zeros((n,n)), self.name)
        curr_graph.source = self.source
        curr_graph.target = self.target
        curr_graph.existing_nodes = [self.source, self.target]
        curr_node = self.name.index(self.source)
        cont = True
        # print(curr_graph.adj)
        while (cont):
            next_node = np.argmax(self.adj[curr_node])
            if(verbose):
                print(
                    "Adding edge {}->{}".format(self.name[curr_node], self.name[next_node]))
                print("edge to remove (because added): ")
                print(self.name[curr_node], self.name[next_node])
                print(edges_to_process)
                print(adj_to_process)
                print("Updated edges_to_process: ", edges_to_process)
            self.dspg_remove_edge(edges_to_process, adj_to_process, [curr_node, next_node])
                

            
            if (self.name[next_node] != self.target):
                # curr_graph.add_node(self.name[next_node])
                curr_graph.existing_nodes.append(self.name[next_node])
            if (self.name[next_node] == self.target):
                cont = False
            curr_graph.add_edge(
                self.name[curr_node], self.name[next_node], v=self.adj[curr_node][next_node])
            curr_node = next_node
            # print(curr_graph.adj)
        return curr_graph

    def check_path(self, src_node, target_node, search_mat, verbose=False):  # Check whether there is a path
        # BFS or DFS
        tmp = find_path(search_mat, src_node, target_node)
        if(verbose):
            print("TMP: ", tmp)
            print("TMP: ", self.adj)
            print("TMP: ", search_mat)
        
        return tmp  # list of all int node

    def dspg_check_NL(self, src_node, target_node, verbose=False):  # Check whether there is a path
        # BFS or DFS
        if(verbose):
            print(self.nodes_nesting[src_node])
            print(self.nodes_nesting[target_node])

        return self.nodes_nesting[src_node] == self.nodes_nesting[target_node]

    def dspg_check_int_NL(self, val, inter, verbose=False):
        # BFS or DFS
        for node in inter[1: len(inter) -1]:
            # if (self.nodes_nesting[node] < self.nodes_nesting[src_node]):
            if(verbose):
                print(val)
                print(node, self.nodes_nesting[node])
            if (self.nodes_nesting[node] < val):
                return False
        return True

    ## Complexity O(E+V)
    def dspg_check_edge(self, src_node, target_node, search_mat, verbose=False, acc=False):

        ## Complexity O(E+V)
        tmp = self.check_path(src_node, target_node, search_mat)
        if(verbose):
            print(tmp)
        if len(tmp) == 0:
            if(verbose):
                print("check path failed")
            return False, None  # BFS or DFS

        # ## Complexity O(1)
        # if(not acc):
        #     if not self.dspg_check_NL(src_node, target_node):
        #         if(verbose):
        #             print("check NL failed")
        #         return False, None  # BFS or DFS

        # val = max([self.nodes_nesting[src_node], self.nodes_nesting[target_node]])
        # ## Complexity O(V)
        # if not self.dspg_check_int_NL(val, tmp):
        #     if(verbose):
        #         print("check NL int failed")
        #     return False, None  # BFS or DFS
        # # del (tmp[0])


        a = self.name[src_node]
        b = self.name[target_node]
        # if 𝑎 is an intermediate node of a PPS p then b is inside a p
        # print("self pps", self.nodes_to_pps)
        # input()
        if a in self.nodes_to_pps.keys():
            # print("a=", a)
            pps = self.nodes_to_pps[a]
            
            # print(pps)
            # print(b)
    
            if b!=pps[0] and b!=pps[1]:
             
                if self.name.index(b) not in self.getAllNodes(pps):
                    # print("fa")
                    return False, None     
     

        # if 𝑏 is an intermediate node of a PPS p then a is inside a p
        if b in self.nodes_to_pps.keys():
            # print("b=", b)
            pps = self.nodes_to_pps[b]
            # print(pps)
            if a!=pps[0] and  a!=pps[1]:
    
                if self.name.index(a) not in self.getAllNodes(pps):
                    # print("fb")
                    return False, tmp    
        return True, tmp


    def BFS(src, adj):
        # Mark all the vertices as not visited
        # Create a queue for BFS
        #a =  visited
        queue = deque()
        n = np.shape(adj)[0]
        visited = [0]*(n+1)
        queue.append(src)
    
        # Assign Component Number
        visited[src] = 1
    
        # Vector to store all the reachable
        # nodes from 'src'
        reachableNodes = []
        #print("0:",visited)
    
        while (len(queue) > 0):
            
            # Dequeue a vertex from queue
            u = queue.popleft()
    
            reachableNodes.append(u)
    
            # Get all adjacent vertices of the dequeued
            # vertex u. If a adjacent has not been visited,
            # then mark it visited and enqueue it
            for itr in np.where(adj[u]!=0)[0]:
                if (visited[itr] == 0):
                    
                    # Assign Component Number to all the
                    # reachable nodes
                    visited[itr] = 1
                    queue.append(itr)
 
        return reachableNodes

    

    def IS(self, node):
        res = Graph.BFS(self.name.index(node), self.adj)
        # print("IS: ", res)
        return res

    def OS(self, node):
        res = Graph.BFS(self.name.index(node), self.adj.T)
        # print("OS: ", res)
        return res

    def getAllNodes(self, pps):
        res = list(set(self.IS(pps[0])) & set(self.OS(pps[1])))
        # print("All: ", res)
        return res
    
    def dspg_remove_edge(self, edges_to_process, adj_to_process, edge_index_list):

        edges_to_process.remove([self.name[edge_index_list[0]], self.name[edge_index_list[1]]])
        adj_to_process[edge_index_list[0]][edge_index_list[1]] = 0
        return 

    def dspg_increment_NL(self, edge, new):
        return 
