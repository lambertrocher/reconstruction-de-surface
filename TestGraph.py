from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
import sys


dim_x = 40
dim_y = 40
dim_z = 40

class FaceNode(object):

    def __init__(self):
        self.x = -1
        self.y = -1
        self.z = -1

    def __setX__(self, x):
        self.x = x

    def __setY__(self, y):
        self.y = y

    def __setZ__(self, z):
        self.z = z


class Graph(object):

    def __init__(self):
        self.dico = {}
        self.source = FaceNode()
        self.sink = FaceNode()
        self.listPoids = {}

    def __fillGraph__(self, tabVoxels):
        for x in range(dim_x):
            for y in range(dim_y):
                for z in range(dim_z):
                    self.dico[(x - 0.5, y, z)] = [(x, y, z + 0.5), (x, y, z - 0.5), (x, y + 0.5, z), (x, y - 0.5, z)]
                    self.dico[(x + 0.5, y, z)] = [(x, y, z + 0.5), (x, y, z - 0.5), (x, y + 0.5, z), (x, y - 0.5, z)]
                    self.dico[(x, y + 0.5, z)] = [(x, y, z + 0.5), (x, y, z - 0.5), (x + 0.5, y, z), (x - 0.5, y, z)]
                    self.dico[(x, y - 0.5, z)] = [(x, y, z + 0.5), (x, y, z - 0.5), (x + 0.5, y, z), (x - 0.5, y, z)]
                    self.dico[(x, y, z + 0.5)] = [(x, y + 0.5, z), (x, y - 0.5, z), (x + 0.5, y, z), (x - 0.5, y, z)]
                    self.dico[(x, y, z - 0.5)] = [(x, y + 0.5, z), (x, y - 0.5, z), (x + 0.5, y, z), (x - 0.5, y, z)]


                    if x > 0:
                        self.dico[(x - 0.5, y, z)] += [(x - 1, y, z + 0.5), (x - 1, y, z - 0.5), (x - 1, y + 0.5, z),
                                                       (x - 1, y - 0.5, z)]
                        self.dico[(x + 0.5, y, z)] += [(x - 1, y, z + 0.5), (x - 1, y, z - 0.5), (x - 1, y + 0.5, z),
                                                       (x - 1, y - 0.5, z)]
                    if y > 0:
                        self.dico[(x, y + 0.5, z)] += [(x, y - 1, z + 0.5), (x, y - 1, z - 0.5), (x + 0.5, y - 1, z),
                                                       (x - 0.5, y - 1, z)]
                        self.dico[(x, y - 0.5, z)] += [(x, y - 1, z + 0.5), (x, y - 1, z - 0.5), (x + 0.5, y - 1, z),
                                                       (x - 0.5, y - 1, z)]
                    if z > 0:
                        self.dico[(x, y, z + 0.5)] += [(x, y + 0.5, z - 1), (x, y - 0.5, z - 1), (x + 0.5, y, z - 1),
                                                       (x - 0.5, y, z - 1)]
                        self.dico[(x, y, z - 0.5)] += [(x, y + 0.5, z - 1), (x, y - 0.5, z - 1), (x + 0.5, y, z - 1),
                                                       (x - 0.5, y, z - 1)]

        return 0



def min_cut_graph(graph):
    return 0


def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if not graph.has_key(start):
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath: return newpath
    return None

def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if not graph.has_key(start):
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

def find_shortest_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if not graph.has_key(start):
        return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest

