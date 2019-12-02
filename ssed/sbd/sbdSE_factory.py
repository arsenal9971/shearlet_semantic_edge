"""
The :mod:`sbd` module implements a class which handles the loading and processing of the SBD (Semantic Boundary Dataset)"""
# Author: Hector Loarca (help of Ingo Guehring from another joint project,
#                        and help of Jonas Adler for point sampling)

# next two lines might work/be necessary only for mac
import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.image as img
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2
import math
import cmocean

import scipy.io

__all__ = ('gen_sembound_SBD', 'sbd_SE', 'plot_SBDcat')

def gen_sembound_SBD(file_path, size):
    bounds = scipy.io.loadmat(file_path)['GTinst']['Boundaries'][0][0]
    cats = scipy.io.loadmat(file_path)['GTinst']['Categories'][0][0]
    sembound = np.zeros([size,size])
    for i in range(bounds.shape[0]):
        bound= bounds[i][0].todense()
        cat = cats[i][0]
        sembound += cv2.resize(bound, dsize=(size, size), interpolation=cv2.INTER_CUBIC)*cat
    return sembound


def sbd_SE(size, img_path, img_file):
    # Get the image
    image = img.imread(img_path+img_file)[:,:,0]
    image = cv2.resize(image, dsize = (size, size), interpolation = cv2.INTER_CUBIC)

    # Get the bound
    bound_path = img_path.replace('/img/','/inst/')
    bound_file = img_file.replace('.jpg','.mat')
    sembounds = gen_sembound_SBD(bound_path+bound_file, size)
    return image, sembounds

def plot_SBDcat(sembound):
    cmap = cmap = plt.cm.get_cmap('tab20')
    cmap.set_bad(color='black')
    return plt.imshow(np.ma.masked_where(sembound == 0, sembound),
                      cmap = cmap)
