from mpl_toolkits.mplot3d import Axes3D
import numpy
import matplotlib.pyplot as plt

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

    print("grille de voxels", grid)
    return grid


def init_distances_grid(voxels_grid):
    distance_grid = voxels_grid

    x, y, z = [0]*3
    for x in range(dim_x):
        for y in range(dim_y):
            for z in range(dim_z):
                if distance_grid[x][y][z] == 1:
                    distance_grid[x][y][z] = 0
                else:
                    distance_grid[x][y][z] = 1
    print("grille des distances", distance_grid)

    return distance_grid


off_file = open("mushroom.off", "r")
vertices = read_off(off_file)


dim_x = 50
dim_y = 50
dim_z = 50

voxels_grid = init_voxels_grid(vertices)
# distances_grid = init_distances_grid(voxels_grid)

N1 = 4
N2 = 4
N3 = 4
ma = numpy.random.choice([0,1], size=(N1,N2,N3), p=[0.90, 0.10])
print("test", ma)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')

ax.voxels(voxels_grid, edgecolor="k")

plt.show()