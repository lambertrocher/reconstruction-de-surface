from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt
import sys


def read_off(file):
    if 'OFF' != file.readline().strip():
        print('Header OFF invalide')
    n_vertices, n_faces, _ = [int(s) for s in file.readline().strip().split(' ')]
    vertices = [[float(s) for s in file.readline().strip().split(' ')] for i_vertex in range(n_vertices)]
    # faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    print("nombre de points : ", n_vertices)
    print(vertices)
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


def init_distances_grid(voxels_grid):
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
        print("filled_grid", filled_grid)
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
                        print("crust mais pas ext", x,y, z)
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
                        print("dans int", x, y, z)
                        v_int[x][y][z] = 1
                    else:
                        print("crust mais pas int", x,y, z)
    return v_int


dim_x = 40
dim_y = 40
dim_z = 40


def main():
    off_file = open("dragon.OFF", "r")
    vertices = read_off(off_file)

    voxels_grid = init_voxels_grid(vertices)
    distances_grid = init_distances_grid(voxels_grid)

    # voxels_grid = dilate_voxels_grid(voxels_grid)
    # voxels_grid = dilate_voxels_grid(voxels_grid)
    # voxels_grid = dilate_voxels_grid(voxels_grid)
    # voxels_grid = dilate_voxels_grid(voxels_grid)

    filled_voxels_grid = flood_fill(voxels_grid, 0, 0, 0, 0, 2)



    # ax.voxels(filled_voxels_grid[:][:][0:8], facecolors=color_voxels_grid(filled_voxels_grid)[:][:][0:8], edgecolor="k")

    v_int = get_v_int(filled_voxels_grid)
    print("v_int", v_int[:][:][15])

    v_ext = get_v_ext(filled_voxels_grid)
    print("v_ext", v_ext)

    for x in range(filled_voxels_grid.shape[0]):
        for y in range(filled_voxels_grid.shape[1]):
            for z in range(filled_voxels_grid.shape[2]):
                if filled_voxels_grid[x][y][z] == 2:
                    filled_voxels_grid[x][y][z] = 2
                elif filled_voxels_grid[x][y][z] == 1:
                    filled_voxels_grid[x][y][z] = 1
                elif filled_voxels_grid[x][y][z] == 0:
                    filled_voxels_grid[x][y][z] = 10

    # v_ext = get_v_ext(filled_voxels_grid)
    # print("v_ext", v_ext)

    colors = color_voxels_grid(filled_voxels_grid[:][:][:])
    print(colors)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')
    # # ax.voxels(filled_voxels_grid[:][:][:], facecolors=colors)
    # ax.voxels(v_ext, edgecolor="k")
    # plt.show()

    # plt.ion()


    plt.figure(1)
    plt.imshow(v_ext[:][:][28], cmap='gray', interpolation='none')


    plt.show()


main()


