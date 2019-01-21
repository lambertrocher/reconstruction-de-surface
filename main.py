from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
import sys
import math
from collections import defaultdict


# This class represents a directed graph using adjacency matrix representation
class Graph:

    def __init__(self, graph):
        self.graph = graph  # residual graph
        # self.org_graph = [i[:] for i in graph]
        self.org_graph = graph.copy()
        self.ROW = len(graph)
        self.COL = len(graph[0])

    '''Returns true if there is a path from source 's' to sink 't' in 
    residual graph. Also fills parent[] to store the path '''

    def BFS(self, s, t, parent):

        # print("bfs", s, t, parent)

        # Mark all the vertices as not visited
        visited = [False] * (self.ROW)

        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:

            # Dequeue a vertex from queue and print it
            u = queue.pop(0)

            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

                    # If we reached sink in BFS starting from source, then return
        # true, else false
        return True if visited[t] else False

    # Returns the min-cut of the given graph
    def minCut(self, source, sink):

        cut_edges = []

        # This array is filled by BFS and to store path
        parent = [-1] * (self.ROW)

        max_flow = 0  # There is no flow initially

        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent):

            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

                # Add path flow to overall flow
            max_flow += path_flow

            # print("flow max", max_flow)


            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]


        print("terminé")


        # print the edges which initially had weights
        # but now have 0 weight
        for i in range(self.ROW):
            for j in range(self.COL):
                if self.graph[i][j] == 0 and self.org_graph[i][j] > 0:
                    # print(str(i) + " - " + str(j))
                    cut_edges += [(i,j)]

                # Create a graph given in the above diagram
        return cut_edges


def read_off(file):
    if 'OFF' != file.readline().strip():
        print('Header OFF invalide')
    n_vertices, n_faces, _ = [int(s) for s in file.readline().strip().split(' ')]
    vertices = [[float(s) for s in file.readline().strip().split(' ')] for i_vertex in range(n_vertices)]
    # faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    print("nombre de points : ", n_vertices)
    # print(vertices)
    return vertices


def init_voxels_grid(vertices):
    grid = numpy.zeros((dim_x, dim_y, dim_z))

    x_min, x_max, y_min, y_max, z_min, z_max = vertices[0] * 2

    for vertex in vertices:
        if x_min >= vertex[0]:
            x_min = vertex[0]
        if x_max <= vertex[0]:
            x_max = vertex[0]
        if y_min >= vertex[1]:
            y_min = vertex[1]
        if y_max <= vertex[1]:
            y_max = vertex[1]
        if z_min >= vertex[2]:
            z_min = vertex[2]
        if z_max <= vertex[2]:
            z_max = vertex[2]

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    print("x range", x_range)
    print("y range", y_range)
    print("z range", z_range)

    for vertex in vertices:
        x = int((vertex[0] - x_min) / x_range * (dim_x-1))
        y = int((vertex[1] - y_min) / y_range * (dim_x-1))
        z = int((vertex[2] - z_min) / z_range * (dim_x-1))
        # print("x, y, z", x, y, z)
        grid[x][y][z] = 1

    #print("grille de voxels", grid)
    return grid


def get_distances_grid(voxels_grid, v_crust):
    distance_grid = voxels_grid.copy()

    x, y, z = [0]*3
    for x in range(dim_x):
        for y in range(dim_y):
            for z in range(dim_z):
                if distance_grid[x][y][z] == 1:
                    distance_grid[x][y][z] = 0
                else:
                    distance_grid[x][y][z] = 1
    #print("grille des distances", distance_grid)

    for x in range(voxels_grid.shape[0]):
        for y in range(voxels_grid.shape[1]):
            for z in range(voxels_grid.shape[2]):
                pass

    return distance_grid


def dilate_voxels_grid(voxels_grid):
    dilated_grid = voxels_grid.copy()
    for x in range(dim_x):
        for y in range(dim_y):
            for z in range(dim_z):
                if voxels_grid[x][y][z] == 1:
                    if x+1 < dim_x:
                        dilated_grid[x+1][y][z] = 1
                    if x-1 >= 0:
                        dilated_grid[x-1][y][z] = 1
                    if y + 1 < dim_y:
                        dilated_grid[x][y+1][z] = 1
                    if y - 1 >= 0:
                        dilated_grid[x][y-1][z] = 1
                    if z + 1 < dim_z:
                        dilated_grid[x][y][z+1] = 1
                    if z - 1 >= 0:
                        dilated_grid[x][y][z-1] = 1
    return dilated_grid


def flood_fill(voxels_grid, x, y, z, col_cible, col_rep):
    filled_grid = voxels_grid.copy()
    pile = []
    if voxels_grid[x][y][z] != col_cible:
        return
    else:
        pile.append([x,y,z])
        while pile:
            n = pile.pop()

            x = n[0]
            y = n[1]
            z = n[2]

            filled_grid[x][y][z] = col_rep

            if x + 1 < dim_x:
                if filled_grid[x+1][y][z] == col_cible:
                    pile.append([x+1,y,z])
            if x - 1 >= 0:
                if filled_grid[x-1][y][z] == col_cible:
                    pile.append([x-1,y,z])
            if y + 1 < dim_y:
                if filled_grid[x][y+1][z] == col_cible:
                    pile.append([x,y+1,z])
            if y - 1 >= 0:
                if filled_grid[x][y-1][z] == col_cible:
                    pile.append([x,y-1,z])
            if z + 1 < dim_z:
                if filled_grid[x][y][z+1] == col_cible:
                    pile.append([x,y,z+1])
            if z - 1 >= 0:
                if filled_grid[x][y][z-1] == col_cible:
                    pile.append([x,y,z-1])
        #print("filled_grid", filled_grid)
    return filled_grid


def color_voxels_grid(filled_voxels_grid):
    colors = numpy.empty(filled_voxels_grid.shape, dtype=object)
    for x in range(filled_voxels_grid.shape[0]):
        for y in range(filled_voxels_grid.shape[1]):
            for z in range(filled_voxels_grid.shape[2]):
                if filled_voxels_grid[x][y][z] == 2:
                    colors[x][y][z] = "green"
                if filled_voxels_grid[x][y][z] == 1:
                    colors[x][y][z] = "yellow"
                if filled_voxels_grid[x][y][z] == 0:
                    colors[x][y][z] = "blue"
                    print("blue", x, y, z)
    return colors


def get_v_ext(filled_voxels_grid):
    v_ext = numpy.zeros(filled_voxels_grid.shape)
    for x in range(filled_voxels_grid.shape[0]):
        for y in range(filled_voxels_grid.shape[1]):
            for z in range(filled_voxels_grid.shape[2]):
                ext = False
                if filled_voxels_grid[x][y][z] == 1:
                    if x + 1 < dim_x:
                        if filled_voxels_grid[x + 1][y][z] == 2:
                            ext = True
                    else:
                        ext = True
                    if x - 1 >= 0:
                        if filled_voxels_grid[x - 1][y][z] == 2:
                            ext = True
                    else:
                        ext = True
                    if y + 1 < dim_y:
                        if filled_voxels_grid[x][y + 1][z] == 2:
                            ext = True
                    else:
                        ext = True
                    if y - 1 >= 0:
                        if filled_voxels_grid[x][y - 1][z] == 2:
                            ext = True
                    else:
                        ext = True
                    if z + 1 < dim_z:
                        if filled_voxels_grid[x][y][z + 1] == 2:
                            ext = True
                    else:
                        ext = True
                    if z - 1 >= 0:
                        if filled_voxels_grid[x][y][z - 1] == 2:
                            ext = True
                    else:
                        ext = True
                    if ext:
                        v_ext[x][y][z] = 1
                    else:
                        pass
                        # print("crust mais pas ext", x,y, z)
    return v_ext


def get_v_int(filled_voxels_grid):
    v_int = numpy.zeros(filled_voxels_grid.shape)
    for x in range(filled_voxels_grid.shape[0]):
        for y in range(filled_voxels_grid.shape[1]):
            for z in range(filled_voxels_grid.shape[2]):
                interior = False
                if filled_voxels_grid[x][y][z] == 0:
                    if x + 1 < dim_x:
                        if filled_voxels_grid[x + 1][y][z] == 1:
                            interior = True
                    if x - 1 >= 0:
                        if filled_voxels_grid[x - 1][y][z] == 1:
                            interior = True
                    if y + 1 < dim_y:
                        if filled_voxels_grid[x][y + 1][z] == 1:
                            interior = True
                    if y - 1 >= 0:
                        if filled_voxels_grid[x][y - 1][z] == 1:
                            interior = True
                    if z + 1 < dim_z:
                        if filled_voxels_grid[x][y][z + 1] == 1:
                            interior = True
                    if z - 1 >= 0:
                        if filled_voxels_grid[x][y][z - 1] == 1:
                            interior = True
                    if interior:
                        # print("dans int", x, y, z)
                        v_int[x][y][z] = 1
                    else:
                        pass
                        # print("crust mais pas int", x,y, z)
    return v_int


def construct_mesh(s_opt):
    for x in range(s_opt.shape[0]-1):
        for y in range(s_opt.shape[1]-1):
            for z in range(s_opt.shape[2]-1):
                n_voxels_in_s_opt = s_opt[x][y][z][0] + s_opt[x+1][y][z][0] + s_opt[x][y+1][z][0] + s_opt[x+1][y+1][z][0] + s_opt[x][y][z+1][0] + s_opt[x+1][y][z+1][0] + s_opt[x][y+1][z+1][0] + s_opt[x+1][y+1][z+1][0]
                if n_voxels_in_s_opt >= 3:
                    v = []
                    if s_opt[x][y][z][0] == 1:
                        v = [x, y, z]
                    elif s_opt[x+1][y][z][0] == 1:
                        v = [x+1, y, z]
                    elif s_opt[x][y+1][z][0] == 1:
                        v = [x, y+1, z]
                    elif s_opt[x+1][y+1][z][0] == 1:
                        v = [x+1, y+1, z]
                    elif s_opt[x][y][z+1][0] == 1:
                        v = [x, y, z+1]
                    elif s_opt[x+1][y][z+1][0] == 1:
                        v = [x+1, y, z+1]
                    elif s_opt[x][y+1][z+1][0] == 1:
                        v = [x, y+1, z+1]
                    elif s_opt[x+1][y+1][z+1][0] == 1:
                        v = [x+1, y+1, z+1]

def find_weight(sommet1, sommet2, distances_grid):
    if (sommet1 == "sink") or (sommet1 == "source") or (sommet2 == "sink") or (sommet2 == "source"):
        return 100000

    x, y, z = sommet1
    x2, y2, z2 = sommet2
    voxel_x = max(math.floor(x), math.floor(x2))
    voxel_y = max(math.floor(y), math.floor(y2))
    voxel_z = max(math.floor(z), math.floor(z2))
    return 0.00001 + (distances_grid[voxel_x][voxel_y][voxel_z])**4

def construct_graph(v_crust, distances_grid, v_int, v_ext):

    vertices_and_edges = {}
    vertices_and_edges["source"] = []
    vertices_and_edges["sink"] = []


    a = .00001
    s = 4

    for x in range(v_crust.shape[0]):
        for y in range(v_crust.shape[1]):
            for z in range(v_crust.shape[2]):
                if v_crust[x][y][z] == 1:
                    if (x - 0.5, y, z) in vertices_and_edges.keys():
                        vertices_and_edges[(x - 0.5, y, z)] += [(x, y, z + 0.5), (x, y, z - 0.5), (x, y + 0.5, z), (x, y - 0.5, z)]
                    else:
                        vertices_and_edges[(x - 0.5, y, z)] = [(x, y, z + 0.5), (x, y, z - 0.5), (x, y + 0.5, z), (x, y - 0.5, z)]

                    if x > 0:
                        if ((v_int[x][y][z] == 1) and (v_crust[x - 1][y][z] == 1)) or (
                                (v_crust[x][y][z] == 1) and (v_int[x - 1][y][z] == 1)):
                            vertices_and_edges["source"].append((x - 0.5, y, z))
                            vertices_and_edges[(x - 0.5, y, z)].append("source")
                        if ((v_ext[x][y][z] == 1) and (v_crust[x - 1][y][z] == 1)) or (
                                (v_crust[x][y][z] == 1) and (v_ext[x - 1][y][z] == 1)):
                            vertices_and_edges["sink"].append((x - 0.5, y, z))
                            vertices_and_edges[(x - 0.5, y, z)].append("sink")

                    if (x + 0.5, y, z) in vertices_and_edges.keys():
                        vertices_and_edges[(x + 0.5, y, z)] += [(x, y, z + 0.5), (x, y, z - 0.5), (x, y + 0.5, z), (x, y - 0.5, z)]
                    else:
                        vertices_and_edges[(x + 0.5, y, z)] = [(x, y, z + 0.5), (x, y, z - 0.5), (x, y + 0.5, z), (x, y - 0.5, z)]

                    if x+1 < v_crust.shape[0]:
                        if ((v_int[x][y][z] == 1) and (v_crust[x + 1][y][z] == 1)) or (
                                (v_crust[x][y][z] == 1) and (v_int[x + 1][y][z] == 1)):
                            vertices_and_edges["source"].append((x + 0.5, y, z))
                            vertices_and_edges[(x + 0.5, y, z)].append("source")
                        if ((v_ext[x][y][z] == 1) and (v_crust[x - 1][y][z] == 1)) or (
                                (v_crust[x][y][z] == 1) and (v_ext[x - 1][y][z] == 1)):
                            vertices_and_edges["sink"].append((x + 0.5, y, z))
                            vertices_and_edges[(x + 0.5, y, z)].append("sink")

                    if (x, y + 0.5, z) in vertices_and_edges.keys():
                        vertices_and_edges[(x, y + 0.5, z)] += [(x, y, z + 0.5), (x, y, z - 0.5), (x + 0.5, y, z), (x - 0.5, y, z)]
                    else:
                        vertices_and_edges[(x, y + 0.5, z)] = [(x, y, z + 0.5), (x, y, z - 0.5), (x + 0.5, y, z), (x - 0.5, y, z)]

                    if y+1 < v_crust.shape[1]:
                        if ((v_int[x][y][z] == 1) and (v_crust[x][y+1][z] == 1)) or (
                                (v_crust[x][y][z] == 1) and (v_int[x][y+1][z] == 1)):
                            vertices_and_edges["source"].append((x, y + 0.5, z))
                            vertices_and_edges[(x, y + 0.5, z)].append("source")
                        if ((v_ext[x][y][z] == 1) and (v_crust[x][y+1][z] == 1)) or (
                                (v_crust[x][y][z] == 1) and (v_ext[x][y+1][z] == 1)):
                            vertices_and_edges["sink"].append((x, y + 0.5, z))
                            vertices_and_edges[(x, y + 0.5, z)].append("sink")

                    if (x, y - 0.5, z) in vertices_and_edges.keys():
                        vertices_and_edges[(x, y - 0.5, z)] += [(x, y, z + 0.5), (x, y, z - 0.5), (x + 0.5, y, z), (x - 0.5, y, z)]
                    else:
                        vertices_and_edges[(x, y - 0.5, z)] = [(x, y, z + 0.5), (x, y, z - 0.5), (x + 0.5, y, z), (x - 0.5, y, z)]

                    if y > 0:
                        if ((v_int[x][y][z] == 1) and (v_crust[x][y-1][z] == 1)) or (
                                (v_crust[x][y][z] == 1) and (v_int[x][y-1][z] == 1)):
                            vertices_and_edges["source"].append((x, y - 0.5, z))
                            vertices_and_edges[(x, y - 0.5, z)].append("source")
                        if ((v_ext[x][y][z] == 1) and (v_crust[x][y-1][z] == 1)) or (
                                (v_crust[x][y][z] == 1) and (v_ext[x][y-1][z] == 1)):
                            vertices_and_edges["sink"].append((x, y - 0.5, z))
                            vertices_and_edges[(x, y - 0.5, z)].append("sink")

                    if (x, y, z + 0.5) in vertices_and_edges.keys():
                        vertices_and_edges[(x, y, z + 0.5)] += [(x, y + 0.5, z), (x, y - 0.5, z), (x + 0.5, y, z), (x - 0.5, y, z)]
                    else:
                        vertices_and_edges[(x, y, z + 0.5)] = [(x, y + 0.5, z), (x, y - 0.5, z), (x + 0.5, y, z), (x - 0.5, y, z)]

                    if z+1 < v_crust.shape[2]:
                        if ((v_int[x][y][z] == 1) and (v_crust[x][y][z+1] == 1)) or (
                                (v_crust[x][y][z] == 1) and (v_int[x][y][z+1] == 1)):
                            vertices_and_edges["source"].append((x, y, z + 0.5))
                            vertices_and_edges[(x, y, z + 0.5)].append("source")
                        if ((v_ext[x][y][z] == 1) and (v_crust[x][y][z+1] == 1)) or (
                                (v_crust[x][y][z] == 1) and (v_ext[x][y][z+1] == 1)):
                            vertices_and_edges["sink"].append((x, y, z + 0.5))
                            vertices_and_edges[(x, y, z + 0.5)].append("sink")

                    if (x, y, z - 0.5) in vertices_and_edges.keys():
                        vertices_and_edges[(x, y, z - 0.5)] += [(x, y + 0.5, z), (x, y - 0.5, z), (x + 0.5, y, z), (x - 0.5, y, z)]
                    else:
                        vertices_and_edges[(x, y, z - 0.5)] = [(x, y + 0.5, z), (x, y - 0.5, z), (x + 0.5, y, z), (x - 0.5, y, z)]

                    if z > 0:
                        if ((v_int[x][y][z] == 1) and (v_crust[x][y][z-1] == 1)) or (
                                (v_crust[x][y][z] == 1) and (v_int[x][y][z-1] == 1)):
                            vertices_and_edges["source"].append((x, y, z - 0.5))
                            vertices_and_edges[(x, y, z - 0.5)].append("source")
                        if ((v_ext[x][y][z] == 1) and (v_crust[x][y][z-1] == 1)) or (
                                (v_crust[x][y][z] == 1) and (v_ext[x][y][z-1] == 1)):
                            vertices_and_edges["sink"].append((x, y, z - 0.5))
                            vertices_and_edges[(x, y, z - 0.5)].append("sink")

    print(vertices_and_edges)
    # print(edges_weight)

    print(len((vertices_and_edges["sink"])))
    print(len((vertices_and_edges["source"])))
    print(len(vertices_and_edges))

    ind = {k: i-1 for i, k in enumerate(vertices_and_edges.keys())}
    inverse_ind = {i-1: k for i, k in enumerate(vertices_and_edges.keys())}
    ind["source"] = 0
    ind["sink"] = len(vertices_and_edges)-1

    inverse_ind[0] = "source"
    inverse_ind[len(vertices_and_edges)-1] = "sink"

    graph = numpy.zeros((len(vertices_and_edges),len(vertices_and_edges)))

    for sommet, aretes in vertices_and_edges.items():
        for sommet2 in aretes:
            graph[ind[sommet]][ind[sommet2]] = find_weight(sommet, sommet2, distances_grid)
            graph[ind[sommet2]][ind[sommet]] = find_weight(sommet, sommet2, distances_grid)
            # graph[ind[sommet]][ind[sommet2]] = 1
            # graph[ind[sommet2]][ind[sommet]] = 1

    return graph, vertices_and_edges, inverse_ind


def get_cut_edges(cut_edges, graph_dict, inverse_index):
    resultat = []
    for arc in cut_edges:
        print(arc[0])
        print(arc[1])
        resultat.append((inverse_index[arc[0]], inverse_index[arc[1]]))
    return resultat

dim_x = 10
dim_y = 10
dim_z = 10


def main():
    off_file = open("dragon.OFF", "r")
    vertices = read_off(off_file)

    voxels_grid = init_voxels_grid(vertices)

    # voxels_grid = dilate_voxels_grid(voxels_grid)
    # voxels_grid = dilate_voxels_grid(voxels_grid)
    # voxels_grid = dilate_voxels_grid(voxels_grid)
    # voxels_grid = dilate_voxels_grid(voxels_grid)

    filled_voxels_grid = flood_fill(voxels_grid, 0, 0, 0, 0, 2)


    # ax.voxels(filled_voxels_grid[:][:][0:8], facecolors=color_voxels_grid(filled_voxels_grid)[:][:][0:8], edgecolor="k")

    v_int = get_v_int(filled_voxels_grid)
    # print("v_int", v_int)

    v_ext = get_v_ext(filled_voxels_grid)
    # print("v_ext", v_ext)

    for x in range(filled_voxels_grid.shape[0]):
        for y in range(filled_voxels_grid.shape[1]):
            for z in range(filled_voxels_grid.shape[2]):
                if filled_voxels_grid[x][y][z] == 2:
                    filled_voxels_grid[x][y][z] = 2
                elif filled_voxels_grid[x][y][z] == 1:
                    filled_voxels_grid[x][y][z] = 1
                elif filled_voxels_grid[x][y][z] == 0:
                    filled_voxels_grid[x][y][z] = 10

    v_crust = voxels_grid.copy()

    distances_grid = get_distances_grid(voxels_grid, v_crust)

    graph, graph_dictionnary, inverse_index = construct_graph(v_crust, distances_grid, v_int, v_ext)

    # graph = [[0, 16, 13, 0, 0, 0],
    #          [0, 0, 10, 12, 0, 0],
    #          [0, 4, 0, 0, 14, 0],
    #          [0, 0, 9, 0, 0, 20],
    #          [0, 0, 0, 7, 0, 4],
    #          [0, 0, 0, 0, 0, 0]]

    g = Graph(graph)

    source = 0
    sink = len(graph)-1

    cut_edges = g.minCut(source, sink)
    cut_edges = get_cut_edges(cut_edges, graph_dictionnary, inverse_index)

    print("cut-edges", cut_edges)

    # parent = [-1]*len(graph)
    # res = g.BFS(source, sink, parent)
    # print(res)
    # print(parent)

    # v_ext = get_v_ext(filled_voxels_grid)
    # print("v_ext", v_ext)

    colors = color_voxels_grid(filled_voxels_grid[:][:][:])
    #print(colors)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')
    # # ax.voxels(filled_voxels_grid[:][:][:], facecolors=colors)
    # ax.voxels(v_ext, edgecolor="k")
    # plt.show()

    # plt.ion()


    plt.figure(1)
    # plt.imshow(v_ext[:][:][70], cmap='gray', interpolation='none')

    plt.figure(2)
    # plt.imshow(v_int[:][:][70], cmap='gray', interpolation='none')


    plt.show()


main()


