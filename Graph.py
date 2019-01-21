from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
import math
import sys
from collections import defaultdict


dim_x = 40
dim_y = 40
dim_z = 40


def find_weight(face1, face2, distanceGrid):
    if (face1 == (-1, -1, -1)) or (face2 == (-1, -1, -1)) or (face1 == (dim_x, dim_y, dim_z)) or (face2 == (dim_x, dim_y, dim_z)):
        return 0

    x, y, z = face1
    x2, y2, z2 = face2
    voxel_x = max(floor(x), floor(x2))
    voxel_y = max(floor(y), floor(y2))
    voxel_z = max(floor(z), floor(z2))
    return pow(distanceGrid[voxel_x][voxel_y][voxel_z], 4) + 0.00001

class Graph(object):

    def __init__(self, distanceGrid):
        self.dico = {(-1,-1,-1):[],(dim_x,dim_y,dim_z):[]}
        self.distanceGrid = distanceGrid
        self.distOrigin = {}
        self.distGrid = {}
        self.visited = {(-1, -1, -1): False, (dim_x, dim_y, dim_z): False}
        self.parent = {(-1, -1, -1): (-5,-5,-5), (dim_x, dim_y, dim_z): (-5, -5, -5)}
        for x in range(dim_x):
            for y in range(dim_y):
                for z in range(dim_z):
                    self.dico[(x - 0.5, y, z)] = []
                    self.dico[(x + 0.5, y, z)] = []
                    self.dico[(x, y + 0.5, z)] = []
                    self.dico[(x, y - 0.5, z)] = []
                    self.dico[(x, y, z + 0.5)] = []
                    self.dico[(x, y, z - 0.5)] = []

                    self.visited[(x - 0.5, y, z)] = False
                    self.visited[(x + 0.5, y, z)] = False
                    self.visited[(x, y + 0.5, z)] = False
                    self.visited[(x, y - 0.5, z)] = False
                    self.visited[(x, y, z + 0.5)] = False
                    self.visited[(x, y, z - 0.5)] = False

                    self.parent[(x - 0.5, y, z)] = (-5,-5,-5)
                    self.parent[(x + 0.5, y, z)] = (-5,-5,-5)
                    self.parent[(x, y + 0.5, z)] = (-5,-5,-5)
                    self.parent[(x, y - 0.5, z)] = (-5,-5,-5)
                    self.parent[(x, y, z + 0.5)] = (-5,-5,-5)
                    self.parent[(x, y, z - 0.5)] = (-5,-5,-5)


    def __fillGraph__(self, tabVoxels):

        for x in range(dim_x):
            for y in range(dim_y):
                for z in range(dim_z):

                    if tabVoxels[x][y][z] == 1: # Valeur Crust

                        w = pow(self.distanceGrid[voxel_x][voxel_y][voxel_z], 4) + 0.00001
                        wSource = 0
                        wSink = 0

                        self.dico[(x - 0.5, y, z)] += [(x, y, z + 0.5), (x, y, z - 0.5), (x, y + 0.5, z), (x, y - 0.5, z)]
                        self.dico[(x + 0.5, y, z)] += [(x, y, z + 0.5), (x, y, z - 0.5), (x, y + 0.5, z), (x, y - 0.5, z)]
                        self.dico[(x, y + 0.5, z)] += [(x, y, z + 0.5), (x, y, z - 0.5), (x + 0.5, y, z), (x - 0.5, y, z)]
                        self.dico[(x, y - 0.5, z)] += [(x, y, z + 0.5), (x, y, z - 0.5), (x + 0.5, y, z), (x - 0.5, y, z)]
                        self.dico[(x, y, z + 0.5)] += [(x, y + 0.5, z), (x, y - 0.5, z), (x + 0.5, y, z), (x - 0.5, y, z)]
                        self.dico[(x, y, z - 0.5)] += [(x, y + 0.5, z), (x, y - 0.5, z), (x + 0.5, y, z), (x - 0.5, y, z)]

                        # Faire les sink et source
                        if tabVoxels[x-1][y][z] == 0:
                            self.dico[(dim_x, dim_y, dim_z)] += [(x - 0.5, y, z)]
                            self.distGrid[((dim_x, dim_y, dim_z), (x - 0.5, y, z))] = wSink
                            self.distGrid[((x - 0.5, y, z), (dim_x, dim_y, dim_z))] = wSink
                        if tabVoxels[x+1][y][z] == 0:
                            self.dico[(dim_x, dim_y, dim_z)] += [(x + 0.5, y, z)]
                            self.distGrid[((dim_x, dim_y, dim_z), (x + 0.5, y, z))] = wSink
                            self.distGrid[((x + 0.5, y, z), (dim_x, dim_y, dim_z))] = wSink
                        if tabVoxels[x][y-1][z] == 0:
                            self.dico[(dim_x, dim_y, dim_z)] += [(x, y - 0.5, z)]
                            self.distGrid[((dim_x, dim_y, dim_z), (x, y - 0.5, z))] = wSink
                            self.distGrid[((x, y - 0.5, z), (dim_x, dim_y, dim_z))] = wSink
                        if tabVoxels[x][y+1][z] == 0:
                            self.dico[(dim_x, dim_y, dim_z)] += [(x, y + 0.5, z)]
                            self.distGrid[((dim_x, dim_y, dim_z), (x, y + 0.5, z))] = wSink
                            self.distGrid[((x, y + 0.5, z), (dim_x, dim_y, dim_z))] = wSink
                        if tabVoxels[x][y][z-1] == 0:
                            self.dico[(dim_x, dim_y, dim_z)] += [(x, y, z - 0.5)]
                            self.distGrid[((dim_x, dim_y, dim_z), (x, y, z - 0.5))] = wSink
                            self.distGrid[((x, y, z - 0.5), (dim_x, dim_y, dim_z))] = wSink
                        if tabVoxels[x][y][z+1] == 0:
                            self.dico[(dim_x, dim_y, dim_z)] += [(x, y, z + 0.5)]
                            self.distGrid[((dim_x, dim_y, dim_z), (x, y, z + 0.5))] = wSink
                            self.distGrid[((x, y, z + 0.5), (dim_x, dim_y, dim_z))] = wSink

                        if tabVoxels[x-1][y][z] == 2:
                            self.dico[(-1, -1, -1)] += [(x - 0.5, y, z)]
                            self.distGrid[((-1, -1, -1), (x - 0.5, y, z))] = wSource
                            self.distGrid[((x - 0.5, y, z), (-1, -1, -1))] = wSource
                        if tabVoxels[x+1][y][z] == 2:
                            self.dico[(-1, -1, -1)] += [(x + 0.5, y, z)]
                            self.distGrid[((-1, -1, -1), (x + 0.5, y, z))] = wSource
                            self.distGrid[((x + 0.5, y, z), (-1, -1, -1))] = wSource
                        if tabVoxels[x][y-1][z] == 2:
                            self.dico[(-1, -1, -1)] += [(x, y - 0.5, z)]
                            self.distGrid[((-1, -1, -1), (x, y - 0.5, z))] = wSource
                            self.distGrid[((x, y - 0.5, z), (-1, -1, -1))] = wSource
                        if tabVoxels[x][y+1][z] == 2:
                            self.dico[(-1, -1, -1)] += [(x, y + 0.5, z)]
                            self.distGrid[((-1, -1, -1), (x, y + 0.5, z))] = wSource
                            self.distGrid[((x, y + 0.5, z), (-1, -1, -1))] = wSource
                        if tabVoxels[x][y][z-1] == 2:
                            self.dico[(-1, -1, -1)] += [(x, y, z - 0.5)]
                            self.distGrid[((-1, -1, -1), (x, y, z - 0.5))] = wSource
                            self.distGrid[((x, y, z - 0.5), (-1, -1, -1))] = wSource
                        if tabVoxels[x][y][z+1] == 2:
                            self.dico[(-1, -1, -1)] += [(x, y, z + 0.5)]
                            self.distGrid[((-1, -1, -1), (x, y, z + 0.5))] = wSource
                            self.distGrid[((x, y, z + 0.5), (-1, -1, -1))] = wSource

                        # Dist Grid
                        self.distGrid[((x - 0.5, y, z), (x, y, z + 0.5))] = w
                        self.distGrid[((x - 0.5, y, z), (x, y, z - 0.5))] = w
                        self.distGrid[((x - 0.5, y, z), (x, y + 0.5, z))] = w
                        self.distGrid[((x - 0.5, y, z), (x, y - 0.5, z))] = w

                        self.distGrid[((x + 0.5, y, z), (x, y, z + 0.5))] = w
                        self.distGrid[((x + 0.5, y, z), (x, y, z - 0.5))] = w
                        self.distGrid[((x + 0.5, y, z), (x, y + 0.5, z))] = w
                        self.distGrid[((x + 0.5, y, z), (x, y - 0.5, z))] = w

                        self.distGrid[((x, y + 0.5, z), (x, y, z + 0.5))] = w
                        self.distGrid[((x, y + 0.5, z), (x, y, z - 0.5))] = w
                        self.distGrid[((x, y + 0.5, z), (x + 0.5, y, z))] = w
                        self.distGrid[((x, y + 0.5, z), (x - 0.5, y, z))] = w

                        self.distGrid[((x, y - 0.5, z), (x, y, z + 0.5))] = w
                        self.distGrid[((x, y - 0.5, z), (x, y, z - 0.5))] = w
                        self.distGrid[((x, y - 0.5, z), (x + 0.5, y, z))] = w
                        self.distGrid[((x, y - 0.5, z), (x - 0.5, y, z))] = w

                        self.distGrid[((x, y, z + 0.5), (x, y + 0.5, z))] = w
                        self.distGrid[((x, y, z + 0.5), (x, y - 0.5, z))] = w
                        self.distGrid[((x, y, z + 0.5), (x + 0.5, y, z))] = w
                        self.distGrid[((x, y, z + 0.5), (x - 0.5, y, z))] = w

                        self.distGrid[((x, y, z - 0.5), (x, y + 0.5, z))] = w
                        self.distGrid[((x, y, z - 0.5), (x, y - 0.5, z))] = w
                        self.distGrid[((x, y, z - 0.5), (x + 0.5, y, z))] = w
                        self.distGrid[((x, y, z - 0.5), (x - 0.5, y, z))] = w


        self.distOrigin = self.distGrid
        for node in self.dico[(dim_x, dim_y, dim_z)]:
            self.disco[node] += (dim_x, dim_y, dim_z)
        for node in self.dico[(-1,-1,-1)]:
            self.disco[node] += (-1,-1,-1)


    def BFS(self, source, sink):

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(source)
        visited[source] = True

        # Standard BFS Loop
        while queue:

            # Dequeue a vertex from queue and print it
            u = queue.pop(0)

            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for node in (self.dico[u]):
                if visited[node] == False and self.distGrid[(u, node)] > 0:
                    queue.append(node)
                    visited[node] = True
                    parent[node] = u

                    # If we reached sink in BFS starting from source, then return
        # true, else false
        return True if visited[sink] else False

        # Returns the min-cut of the given graph
    def minCut(self, source, sink):

        max_flow = 0  # There is no flow initially

        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink):

            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.distGrid[(parent[s],s)])
                s = parent[s]

            # Add path flow to overall flow
            max_flow += path_flow

            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while (v != source):
                u = parent[v]
                self.distGrid[(u, v)] -= path_flow
                self.distGrid[(v, u)] += path_flow
                v = parent[v]

                    # print the edges which initially had weights
        # but now have 0 weight
        for i in self.distGrid:
            if self.distGrid[i] == 0 and self.distOrigint[i] > 0:
                        print i





