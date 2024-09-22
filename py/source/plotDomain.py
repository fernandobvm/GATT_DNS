import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#TODO: Não está rodando, o gpt parece nao conseguir corrigir
class Mesh:
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z

class Boundary:
    def __init__(self, wall, corners):
        self.wall = wall
        self.corners = corners

class Wall:
    def __init__(self, front, back, up, down, right, left):
        self.front = front
        self.back = back
        self.up = up
        self.down = down
        self.right = right
        self.left = left

class Corners:
    def __init__(self, limits, dir):
        self.limits = limits
        self.dir = dir

def plot_domain(mesh, boundary):
    colors = plt.cm.get_cmap('tab10', 6)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    walls = [boundary.wall.front, boundary.wall.back, boundary.wall.up, boundary.wall.down, boundary.wall.right, boundary.wall.left]

    for i, wallToPlot in enumerate(walls):
        if wallToPlot.size > 0:
            if i in [0, 1]:  # front and back
                X = mesh.X[wallToPlot[:, [0, 0, 0, 0]]]
                Y = mesh.Y[wallToPlot[:, [2, 3, 3, 2]]]
                Z = mesh.Z[wallToPlot[:, [4, 4, 5, 5]]]
            elif i in [2, 3]:  # up and down
                X = mesh.X[wallToPlot[:, [0, 1, 1, 0]]]
                Y = mesh.Y[wallToPlot[:, [2, 2, 2, 2]]]
                Z = mesh.Z[wallToPlot[:, [4, 4, 5, 5]]]
            else:  # right and left
                X = mesh.X[wallToPlot[:, [0, 1, 1, 0]]]
                Y = mesh.Y[wallToPlot[:, [2, 2, 3, 3]]]
                Z = mesh.Z[wallToPlot[:, [4, 4, 4, 4]]]

            for j in range(X.shape[0]):
                verts = np.array([X[j, :], Y[j, :], Z[j, :]]).T  # Organizando corretamente como lista de vértices
                ax.add_collection3d(Poly3DCollection([verts], facecolors=colors(i), edgecolor='k'))

    corners = boundary.corners.limits
    cornerDir = boundary.corners.dir
    for i in range(corners.shape[0]):
        if np.sum(np.abs(cornerDir[i, :])) == 2:
            ax.plot3D(mesh.X[corners[i, [0, 1]]], mesh.Z[corners[i, [4, 5]]], mesh.Y[corners[i, [2, 3]]], 'r', linewidth=2)
        else:
            ax.plot3D([mesh.X[corners[i, 0]]], [mesh.Z[corners[i, 4]]], [mesh.Y[corners[i, 2]]], 'ro')

        X = np.mean(mesh.X[corners[i, [0, 1]]])
        Y = np.mean(mesh.Y[corners[i, [2, 3]]])
        Z = np.mean(mesh.Z[corners[i, [4, 5]]])

        scale = 0.1

        ax.plot3D([X, X + cornerDir[i, 0] * scale], [Z, Z + cornerDir[i, 2] * scale], [Y, Y + cornerDir[i, 1] * scale], 'g', linewidth=2)

    ax.set_box_aspect([1, 1, 1])  # Aspect ratio
    ax.view_init(elev=30, azim=30)
    plt.tight_layout()
    plt.show()

# Exemplo de uso:
X = np.random.rand(10, 6)
Y = np.random.rand(10, 6)
Z = np.random.rand(10, 6)
mesh = Mesh(X, Y, Z)

wall = Wall(
    front=np.random.randint(0, 10, (5, 6)),
    back=np.random.randint(0, 10, (5, 6)),
    up=np.random.randint(0, 10, (5, 6)),
    down=np.random.randint(0, 10, (5, 6)),
    right=np.random.randint(0, 10, (5, 6)),
    left=np.random.randint(0, 10, (5, 6))
)

corners = Corners(
    limits=np.random.randint(0, 10, (4, 6)),
    dir=np.random.randint(-1, 2, (4, 3))
)

boundary = Boundary(wall, corners)

plot_domain(mesh, boundary)
