import sys
import math 
from sklearn.metrics import pairwise_distances
import numpy as np

class Graph():
    def __init__(self, V, dist):
        self.V = V
        self.dist = dist
    def MST(self):
        # 'Shortest' distance to reach a node from current MST
        key = np.asarray([sys.maxsize for i in range(self.V)])
        # Parent of each vertex in MST
        self.mst = [None for i in range(self.V)] 
        # Check existence in MST
        in_mst = np.asarray([False for i in range(self.V)])
 		
 		# Root node 0
        key[0] = 0
        self.mst[0] = -1 
 		
        for i in range(self.V):
            # Pick the nearest node not in MST
            u = np.argmin(key[~in_mst])
            in_mst[u] = True

            # Update key for the node not in MST yet
            for v in range(self.V):
                if not in_mst[v] and self.dist[u][v] < key[v]:
                    key[v] = self.dist[u][v]
                    self.mst[v] = u
 
def good_path(dim, dist):
    g = Graph(dim, dist)
    g.MST()
    l = len(g.mst)
    child = [[] for i in range(l)]
    inpath = [False for i in range(l)]
    for i in range(l):
        child[g.mst[i]].append(i)
    inpath[0] = True
    def DFS(node, path = []):
        path += [node]
        if child[node] == []:
            return
        for c in child[node]:
            if not inpath[c]:
                inpath[c] = True
                DFS(c, path = path)
        return path
    return np.asarray(DFS(0) + [0])