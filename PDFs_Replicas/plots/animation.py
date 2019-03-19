from PIL import Image
import numpy as np
import imageio

ipath = "iterations/iteration_%d.png"

images = []
for i in range(11):
    images.append(imageio.imread(ipath%(i*1000)))
imageio.mimsave('iterations.gif', images, fps=1)
