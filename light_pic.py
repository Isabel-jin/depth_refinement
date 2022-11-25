import Harmonic
import numpy as np
from scipy import ndimage, misc
import sys, math, os
from PIL import Image
import matplotlib.pyplot as plt

#球谐函数系数
lk = [94.678201,
-3.423619,
13.471551,
25.164422,
-0.318307,
-8.479666,
4.288890,
24.088781,
-9.017026]

def cubemap(lk):
    SIZE = 1024
    HSIZE = SIZE / 2.0

    side_im = np.zeros((SIZE, SIZE), np.uint8)
    color = np.zeros((SIZE, SIZE), np.uint8)
    for i in range(0, 6):
        #  This is numpy's way of visiting each point in an ndarray, I guess its fast...
        it = np.nditer(side_im, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            # Axis
            axA = it.multi_index[0]
            axB = it.multi_index[1]
            # Color is an axis, so we visit each point 3 times for R,G,B actually...

            # Here for each face we decide what each axis represents, x, y or z.
            z = -axA + HSIZE

            if i == 0:
                x = HSIZE
                y = -axB + HSIZE
            elif i == 1:
                x = -HSIZE
                y = axB - HSIZE
            elif i == 2:
                x = axB - HSIZE
                y = HSIZE
            elif i == 3:
                x = -axB + HSIZE
                y = -HSIZE
            elif i == 4:
                z = HSIZE
                x = axB - HSIZE
                y = axA - HSIZE
            elif i == 5:
                z = -HSIZE
                x = axB - HSIZE
                y = -axA + HSIZE

            # Now that we have x,y,z for point on plane, convert to spherical
            r = math.sqrt(float(x * x + y * y + z * z))
            theta = math.acos(float(z) / r)
            phi = -math.atan2(float(y), x)

            color[axA, axB] = Harmonic.Render(theta, phi, lk)
            it.iternext()

        # Save output image using prefix, type and index info.
        pimg = Image.fromarray(color)
        pimg.save(os.path.join('./result/light', "%s%d.%s" % ('side_', i, 'jpg')), quality=85)
        plt.figure()
        plt.hist(np.array(pimg).ravel(), 256, [0, 255])

        plt.savefig(os.path.join('./result/light', "%s%d.%s" % ('hist_side_', i, 'jpg')))

if __name__ == "__main__":
    cubemap(lk)

