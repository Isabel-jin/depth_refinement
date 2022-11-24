import numpy as np
from PIL import Image
import math
D=np.array(Image.open('RGBD_single/0_000000.png'))

def gaussian_filter(img, K_size=3, sigma=1.3):
 
 
    H, W=img.shape
    C=1
    ## Zero padding
 
    pad = K_size // 2
 
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float64)
 
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float64)
 
    ## prepare Kernel
 
    K = np.zeros((K_size, K_size), dtype=np.float64)
 
    for x in range(-pad, -pad + K_size):
 
        for y in range(-pad, -pad + K_size):
 
            K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
 
    K /= (2 * np.pi * sigma * sigma)
 
    K /= K.sum()
 
    tmp = out.copy()
 
    # filtering
 
    for y in range(H):
 
        for x in range(W):
 
            for c in range(C):
 
                out[pad + y, pad + x] = np.sum(K * tmp[y: y + K_size, x: x + K_size])
 
    out = out[pad: pad + H, pad: pad + W].astype(np.uint16)
    return out

D_f=gaussian_filter(D)
np.savetxt(fname="D_filtered.csv",X=D_f,fmt="%f",delimiter=",")
D_I=Image.fromarray(D_f)
D_I.save('D_filtered.png')
