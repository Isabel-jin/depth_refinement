from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import Harmonic

lk = [-20,
-20,
-20,
-20,
-20,
-20,
-20,
-20,
-20]

def rgb_to_hex(r, g, b):
    return ('{:02X}' * 3).format(r, g, b)

def show_sphere(lk):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    size = 100
    phi = np.linspace(0, np.pi * 2, size)
    theta = np.linspace(0, np.pi, size)


    x = np.zeros(size * size)
    y = np.zeros(size * size)
    z = np.zeros(size * size)
    color = np.zeros(size * size, np.uint8)
    colorer = []

    for i in range(size):
        for j in range(size):
            n = size * i + j
            x[n] = np.cos(phi[i]) * np.sin(theta[j])
            y[n] = np.sin(phi[i]) * np.sin(theta[j])
            z[n] = np.cos(theta[j])
            color[n] = Harmonic.Render(theta[j], phi[i], lk)
            colorer.append('#' + str(rgb_to_hex(color[n],color[n],color[n])))


    ax = plt.axes(projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')


    ax.scatter3D(x, y, z, c=colorer)

    plt.show()

if __name__ == '__main__':
    show_sphere(lk)

