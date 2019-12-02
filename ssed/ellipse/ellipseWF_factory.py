"""
The :mod:`ellipses` module implements a class which handles the creation of the
random ellipses data"""
# Author: Hector Loarca (help of Ingo Guehring from another joint project,
#                        and help of Jonas Adler for point sampling)

# next two lines might work/be necessary only for mac
import matplotlib
#matplotlib.use('TkAgg')

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2
import math
import cmocean

__all__ = ('random_phantom', 'random_phantom_dataset', 'random_ellipses', 'random_ellipses_dataset', '_fig2data', '_ellipse_gen', '_ellipse_grad_gen', '_center_origin_gen', '_width_height_side_gen', '_angle_gen', '_opacity_gen', '_ellipse_random', '_phantom_outer_parameters', '_phantom_small_inner_parameters', '_phantom_big_inner_parameters', '_phantom_inner_ellipse_random', 'angles_toclasses', 'rotate', 'Wavefrontset_ellipse_classes', 'WFupdate', 'WFupdate_sino', 'plot_WF')

def random_phantom(size, nEllipses, dirBias, nClasses):
    """Create a `size` x `size` image with `nEllipses` phantom with random
    ellipses

    Parameters
    -----------
    size : integer, size of image

    nEllipses : integer, the number of ellipses in the image

    dirBias : integer, the center angle of the directional bias

        Returns
    -----------
    phantom : numpy array, `size` x `size` image with `nEllipses`
         phantom with random ellipses
    """
    # Create the WFimage, WFpoints and WF_classes
    WFimage = np.zeros((size,size))
    WFpoints_all = []
    WFclasses_all = []

    # Create the outer ellipses hull
    center, width, height, angle, opacity = _phantom_outer_parameters(size,dirBias, big=1)
    grad_level = rnd.uniform(-2,2)
    if grad_level>=0:
        big_ellipse= _ellipse_grad_gen(center, width, height, angle, size, opacity,grad_level)
    else:
        big_ellipse= _ellipse_gen(center, width, height, angle, size, opacity)

    # Update WFimage, WFpoints and WFclasses
    WFpoints, WFclasses = Wavefrontset_ellipse_classes(center, width, height, angle, nClasses)

    WFpoints_all += list(WFpoints)
    WFclasses_all += list(WFclasses)
    WFimage = WFupdate(WFpoints, WFclasses, WFimage)

    _, width, height, angle, opacity = _phantom_outer_parameters(size,dirBias, big=0)
    center = [center[0]+rnd.randint(-2,2),center[1]+rnd.randint(-2,2)]
    grad_level = rnd.uniform(-2,2)
    if grad_level>=0:
        small_ellipse= _ellipse_grad_gen(center, width, height, angle, size, opacity, grad_level)
    else:
        small_ellipse= _ellipse_gen(center, width, height, angle, size, opacity)

    # Update WFimage
    WFpoints, WFclasses = Wavefrontset_ellipse_classes(center, width, height, angle, nClasses)

    WFpoints_all += list(WFpoints)
    WFclasses_all += list(WFclasses)
    WFimage = WFupdate(WFpoints, WFclasses, WFimage)

    phantom = (1-big_ellipse)*(small_ellipse)
    phantom = phantom/phantom.max()
    phantom = np.interp(phantom,
                        (phantom.min(), phantom.max()),
                        (0, 1))

    # Create the inner ellipses hull

    # Small ellipses
    inner_ellipses = np.zeros((size,size))
    for i in range(nEllipses):
        center, width, height, angle, opacity = _phantom_small_inner_parameters(size,dirBias)
        grad_level = rnd.uniform(-2,2)
        if grad_level <= 0:
            inner_ellipses += _ellipse_gen(center, width, height, angle, size, opacity)
        else:
            inner_ellipses += _ellipse_grad_gen(center, width, height, angle, size, opacity, grad_level)
        # Update WFimage
        WFpoints, WFclasses = Wavefrontset_ellipse_classes(center, width, height, angle, nClasses)

        WFpoints_all += list(WFpoints)
        WFclasses_all += list(WFclasses)
        WFimage = WFupdate(WFpoints, WFclasses, WFimage)

    # Big ellipses
    for i in range(int(nEllipses/2)):
        center, width, height, angle, opacity = _phantom_big_inner_parameters(size,dirBias)
        grad_level = rnd.uniform(-2,2)
        if grad_level <= 0:
            inner_ellipses += _ellipse_gen(center, width, height, angle, size, opacity)
        else:
            inner_ellipses += _ellipse_grad_gen(center, width, height, angle, size, opacity, grad_level)
        # Update WFimage
        WFpoints, WFclasses = Wavefrontset_ellipse_classes(center, width, height, angle, nClasses)

        WFpoints_all += list(WFpoints)
        WFclasses_all += list(WFclasses)
        WFimage = WFupdate(WFpoints, WFclasses, WFimage)

    # Sum the hull and the inner ellipses and normalize
    inner_ellipses = 1 - inner_ellipses/inner_ellipses.max()
    inner_ellipses = np.interp(inner_ellipses, (inner_ellipses.min(), inner_ellipses.max()), (0, 1))

    phantom += inner_ellipses
    phantom = phantom/phantom.max()
    phantom = np.interp(phantom, (phantom.min(), phantom.max()), (0, 1))

    return phantom, np.array(WFpoints_all), WFclasses_all, WFimage

def random_phantom_dataset(size, nEllipses, nImages,dirBias, nClasses):
    """Create a 3D numpy array with images of random ellipses

    Parameters
    -----------
    size : integer, size of image

    nEllipses : integer, the number of ellipses in the image

    nImages : integer, number of Images in list

    Returns
    -----------
    dataset : numpy array of dimension `nImages` x `size` x `size`.
        Each image has a random number of up to `nEllipses` random ellipses
    """
    print('Start generating ellipse data...')
    dataset_phantoms = np.zeros((nImages, size, size))
    dataset_WFs = np.zeros((nImages, size, size))
    for i in range(nImages):
        if (i + 1) % 5 == 0:
            print('Finished {} of {}.'.format(i + 1, nImages))
        dataset_phantoms[i, :, :], dataset_WFs[i, :, :]  = random_phantom(size,
                                                                          rnd.randint(1, nEllipses),
                                                                          dirBias, nClasses)
    return dataset_phantoms, dataset_WFs

def random_ellipses(size, nEllipses):
    """Create a `size` x `size` image with `nEllipses` random ellipses

    Parameters
    -----------
    size : integer, size of image

    nEllipses : integer, the number of ellipses in the image

    Returns
    -----------
    ellipses : numpy array, `size` x `size` image with `nEllipses`
         random ellipses
    """
    ellipses = sum([_ellipse_random(size) for i in range(nEllipses)])
    ellipses = 1 - ellipses/ellipses.max()
    ellipses = np.interp(ellipses, (ellipses.min(), ellipses.max()), (0, 1))
    return ellipses


def random_ellipses_dataset(size, nEllipses, nImages):
    """Create a 3D numpy array with images of random ellipses

    Parameters
    -----------
    size : integer, size of image

    nEllipses : integer, the number of ellipses in the image

    nImages : integer, number of Images in list

    Returns
    -----------
    dataset : numpy array of dimension `nImages` x `size` x `size`.
        Each image has a random number of up to `nEllipses` random ellipses
    """
    print('Start generating ellipse data...')
    dataset = np.zeros((nImages, size, size))
    for i in range(nImages):
        if (i + 1) % 5 == 0:
            print('Finished {} of {}.'.format(i + 1, nImages))
        dataset[i, :, :] = random_ellipses(size, rnd.randint(1, nEllipses))
    return dataset


def _fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels
    and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel
    # to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def _ellipse_gen(center, width, height, angle, size, opacity=1):
    """Function that generates the data of the ellipse
    """
    # Generate the Ellipse figure
    fig = plt.figure(0, frameon=False, figsize=(1, 1), dpi=size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    e = Ellipse([center[0],size-center[1]], width, height, angle)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(None)
    e.set_facecolor(np.zeros(3))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    fig.add_axes(ax)
    plt.axis('off')
    # Convert figure to data
    data = _fig2data(fig)
    plt.close(fig)
    # Take just the first color entry
    data = data[:, :, 1]
    # Normalize the data
    data = data/data.max()
    # data = np.flip(data, 0)
    return ((data-1)*opacity)+1


def _ellipse_grad_gen(center, width, height, angle, size, opacity, grad_level):
    """Function that generates the data of the ellipse with color gradient
    """
    # Compute the needed parameters
    h,k = center[0],center[1]
    a,b = width/2, height/2
    theta = -math.radians(angle)
    # Precalculate constants
    st, ct =  math.sin(theta), math.cos(theta)
    aa, bb = a**2, b**2

    # Generate (x,y) coordinate arrays
    y,x = np.mgrid[-k:size-k,-h:size-h]
    # Calculate the weight for each pixel
    ellipse = (((x * ct + y * st) ** 2) / aa) + (((x * st - y * ct) ** 2) / bb)
    ellipse = np.clip(1.0 - ellipse, 0,grad_level)*opacity/grad_level

    return 1-ellipse

def _center_origin_gen(size):
    return [rnd.randint(low=0+int(size/10), high=size-int(size/10)),
            rnd.randint(low=0+int(size/10), high=size-int(size/10))]


def _width_height_side_gen(size):
    return [rnd.randint(low=10, high=int(size/3)),
            rnd.randint(low=10, high=int(size/3))]


def _angle_gen():
    return rnd.randint(low=0, high=180)


def _opacity_gen():
    return rnd.uniform(0.2, 1.0)


def _ellipse_random(size):
    # Random parameters for the ellipse
    center = _center_origin_gen(size)
    width, height = _width_height_side_gen(size)
    angle = _angle_gen()
    opacity = _opacity_gen()
    return _ellipse_gen(center, width, height, angle, size, opacity)

def _phantom_outer_parameters(size,dirBias, big=1):
    if big:
        width, height = [rnd.randint(low=size-int(size/8), high=size-int(size/16)),
                        rnd.randint(low=size-int(size/12), high=size-int(size/20))]
        opacity = rnd.uniform(0.2, 0.3)
    else:
        width, height = [rnd.randint(low=size-int(size/8), high=size-int(size/10)),
                        rnd.randint(low=size-int(size/8), high=size-int(size/15))]
        opacity = rnd.uniform(0.6, 1.0)
    center = [int(size/2)+rnd.randint(-5,5),
              int(size/2)+rnd.randint(-5,5)]
    angle = (dirBias+rnd.randint(-5,5))%180
    return center, width, height, angle, opacity

def _phantom_small_inner_parameters(size,dirBias):
    width, height = [rnd.randint(low=4, high=int(size/18)),
                        rnd.randint(low=4, high=int(size/15))]
    opacity = rnd.uniform(0.2, 1.)
    center = [int(size/2)+rnd.randint(-int(size/4),int(size/4)),
              int(size/2)+rnd.randint(-int(size/4),int(size/4))]
    angle = (dirBias+rnd.randint(-10,10))%180
    return center, width, height, angle, opacity

def _phantom_big_inner_parameters(size,dirBias):
    # Compute the ellipse
    width, height = [rnd.randint(low=15, high=int(size/6)),
                        rnd.randint(low=20, high=int(size/4))]
    opacity = rnd.uniform(0.2, 1.)
    center = [int(size/2)+rnd.randint(-int(size/4),int(size/4)),
              int(size/2)+rnd.randint(-int(size/4),int(size/4))]
    angle = (dirBias+rnd.randint(-10,10))%180
    return center, width, height, angle, opacity

def _phantom_inner_ellipse_random(size, dirBias,small = 1, grad_level = 0):
    # Random parameters for the ellipse
    if small:
        center, width, height, angle, opacity = _phantom_small_inner_parameters(size,dirBias)
    else:
        center, width, height, angle, opacity = _phantom_big_inner_parameters(size,dirBias)
    if grad_level <= 0:
        ellipse = _ellipse_gen(center, width, height, angle, size, opacity)
    else:
        ellipse = _ellipse_grad_gen(center, width, height, angle, size, opacity,grad_level)
    return ellipse

def angles_toclasses(nClasses,angle):
    # Function that generates the angles given by the number of scales and take
    # Version that doesnt depend on the number of scales
    return ((np.floor((angle%180)/(180/nClasses)))+1).astype(int)

def rotate(origin, point, angle):
    # Rotation function
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    # Angle in radians in the other direction (counter clock wise)
    rad_angle = angle*np.pi/180

    qx = ox + np.cos(rad_angle) * (px - ox) - np.sin(rad_angle) * (py - oy)
    qy = oy + np.sin(rad_angle) * (px - ox) + np.cos(rad_angle) * (py - oy)
    return qx, qy


def Wavefrontset_ellipse_classes(center, width, height, angle, nClasses):
    # Function that generates the points and class of angles (in degrees) of directions in the Wavefrontset of a ellipse
    # Version that doesnt deppend on number of scales
    # Compute a and b (semiaxis)
    a = width/2
    b = height/2

    # Sample twice as many points as needed for the circumference (overestimated here)
    n_points = 2 * np.pi * max(width, height)

    # Generate angles and x, y coordinates in closed form
    angles = np.linspace(0, 2 * np.pi, n_points)
    x = a * np.cos(angles)
    y = b * np.sin(angles)
    angles = 180*angles/np.pi

    # Rotated angles
    rot_angles_classes = angles_toclasses(nClasses,(180-angles+angle)%180)
    rot_angles_classes = [np.array([rot_anglei]) for rot_anglei in rot_angles_classes]
    # Rotated and translated points
    rot_trans_points = np.array([np.array(rotate([0,0],[x[i],y[i]],180-angle))+np.array(center) for i in range(len(x))])
     # Lets randomize the points and its angles to eleminate any structure given by construction
    permutations = np.random.permutation(len(rot_angles_classes))
    rot_trans_points = rot_trans_points[permutations]
    rot_angles_classes = [rot_angles_classes[permutation] for permutation in permutations]
    return rot_trans_points, rot_angles_classes

def WFupdate(WFpoints, WFclasses, WFimage):
    # Function to update the Wavefront set classes with points and classes
    size = WFimage.shape[0]
    WFimage[WFpoints.astype(int)[:,1],WFpoints.astype(int)[:,0]] = np.array(WFclasses)[:,0]
    return WFimage

def WFupdate_sino(WFpoints, WFclasses, WFimage):
    # Function to update the sinogram Wavefront set classes with points and classes
    size = WFimage.shape[0]
    WFimage[WFpoints.astype(int)[:,1],WFpoints.astype(int)[:,0]] = np.array(WFclasses)[:,0]
    return WFimage

def plot_WF(WFimage):
    # Plotting function for wavefront set images
    cmap = cmocean.cm.phase
    cmap.set_bad(color='black')
    return plt.imshow(np.ma.masked_where(WFimage == 0, WFimage),
                      cmap = cmap)

