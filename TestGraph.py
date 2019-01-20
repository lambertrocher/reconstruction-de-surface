from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
import math
import sys


dim_x = 40
dim_y = 40
dim_z = 40

class Graph(object):

    def __init__(self):
        self.dico = {}
        self.source = (-1, -1, -1)
        self.sink = (40, 40, 40)
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
                    if x < dim_x:
                        self.dico[(x + 0.5, y, z)] += [(x - 1, y, z + 0.5), (x - 1, y, z - 0.5), (x - 1, y + 0.5, z),
                                                       (x - 1, y - 0.5, z)]
                    if y > 0:
                        self.dico[(x, y + 0.5, z)] += [(x, y - 1, z + 0.5), (x, y - 1, z - 0.5), (x + 0.5, y - 1, z),
                                                       (x - 0.5, y - 1, z)]
                    if y < dim_y:
                        self.dico[(x, y - 0.5, z)] += [(x, y - 1, z + 0.5), (x, y - 1, z - 0.5), (x + 0.5, y - 1, z),
                                                       (x - 0.5, y - 1, z)]
                    if z > 0:
                        self.dico[(x, y, z + 0.5)] += [(x, y + 0.5, z - 1), (x, y - 0.5, z - 1), (x + 0.5, y, z - 1),
                                                       (x - 0.5, y, z - 1)]
                    if z < dim_z:
                        self.dico[(x, y, z - 0.5)] += [(x, y + 0.5, z - 1), (x, y - 0.5, z - 1), (x + 0.5, y, z - 1),
                                                       (x - 0.5, y, z - 1)]

                    # Faire les sink et source et liste poids



def find_weight(face1, face2, distanceGrid):
    x, y, z = face1
    x2, y2, z2 = face2
    


def min_cut_graph(graph, coupe, poids, start, fin):

    coupe = coupe + [start]


    if start == fin:
        return poids, coupe

    if not graph.dico.has_key(start):
        return None

    for node in graph.dico[start]:
        minPoid = 1000000
        minCoupe = []
        if node not in coupe:
            # calcul du poids prochain ici
            newPoid = 0
            poidAux, newCoupe = min_cut_graph(graph, coupe, newPoid, node, fin)
            if poidAux < minPoid:
                minCoupe = newCoupe

    return minPoid, minCoupe





