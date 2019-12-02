"""
The :mod:`canon_relation` moduole implements a set of functions that perform the digital
canonical relation for the ray transform"""
# Author: Hector Andrade Loarca

# Needed modules
import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy as np

__all__ = ('point_img2sino', 'class_img2sino', 'CanRel_img2sino', 'point_sino2img', 'CanRel_sino2img')

# Forward canonical relation

def point_img2sino(x, phi, size, sinogram_shape):
    """Compute the mapping of the position entry of the Wavefront set from images to
    sinograms

    Parameters
    ----------
    x : numpy array, position of the singular point in the image

    phi : float, orientation angle of the singular point in the image

    size : integer, size of the image

    sinogram_shape : integer, shape of the sinogram

        Returns
    ___________
    np.array([s, theta]) : numpy array, distance to center entry `s` and angle entry `theta`
                            of the corresponding singular point in the sinogram
    """
    # Compute the angle in radians
    rad_phi = ((phi[0])*np.pi)/180
    # In the images the coordinates are switched and moved
    x01, x02 = x[1]-int(size/2), x[0]-int(size/2)
    # Compute the distance to center component
    if phi <= 90:
        dist_center = -x01*np.sin(rad_phi)+x02*np.cos(rad_phi)+int(sinogram_shape[1]/2)
    else:
        dist_center = x01*np.sin(rad_phi)-x02*np.cos(rad_phi)+int(sinogram_shape[1]/2)

    return np.array([dist_center,(phi+90)%180])


def class_img2sino(x, phi, size, sinogram_shape):
    """Compute the mapping of the orientation class entry of the Wavefront set from images to
    sinograms

    Parameters
    ----------
    x : numpy array, position of the singular point in the image

    phi : float, orientation angle of the singular point in the image

    size : integer, size of the image

    sinogram_shape : tuple, shape of the sinogram

        Returns
    ___________
    np.array([class]) : numpy array, with the class entry of the corresponding singular
                        point
    """

    # Compute the angle in radians
    rad_phi = ((phi[0])*np.pi)/180
    # In the images the coordinates are switched and moved
    x01, x02 = x[1]-int(size/2), x[0]-int(size/2)
    # Compute the class
    if phi <= 90:
        classe = -x01*np.cos(rad_phi)-x02*np.sin(rad_phi)
    else:
        classe = x01*np.cos(rad_phi)+x02*np.sin(rad_phi)

    return np.array([((np.arctan(classe))*180/np.pi)%180+1])

def CanRel_img2sino(WFpoints, WFclasses, size, sinogram_shape, num_angles):
    """ Compute the Wavefront set points and classes of the sinogram from an image

    Parameters
    ----------
    WFpoints: numpy array, singular points positions of the image

    WFclasses: list, singiular points orientations classes of the image

    size: integer, size of the image

    sinogram_shape: tuple, shape of the sinogram of the image

    num_angles: integer, number of angle classes of the wavefront set

        Returns:
    ------------
    WFpoints_sino, WFclasses_sino: numpy array and list, wavefront set points and wavefront set classes
                                    of the sinogram of the image
    """
    # Compute the WF points in the sinogram
    WFpoints_sino = []
    WFclasses_sino = []
    for i in range(WFpoints.shape[0]):
        if (WFclasses[i][0] % int(180/num_angles) == 0):
            WFpoints_sino.append(point_img2sino(WFpoints[i], WFclasses[i]-1, size, sinogram_shape))
            WFclasses_sino.append(class_img2sino(WFpoints[i], WFclasses[i]-1, size, sinogram_shape))

    return np.array(WFpoints_sino), WFclasses_sino

# Inverse canonical relation

def point_sino2img(y, varphi, size, sinogram_shape):
    """Compute the mapping of the position entry of the Wavefront set from sinograms to
    images

    Parameters
    ----------
    y : numpy array, position of the singular point in the sinogram

    varphi : float, orientation angle of the singular point in the sinogram

    size : integer, size of the image

    sinogram_shape : integer, shape of the sinogram

        Returns
    ___________
    x : numpy array, position of the corresponding singular point in the sinogram
    """
    rad_varphi = ((varphi[0])*np.pi)/180
    s, phi =  y
    s = s - sinogram_shape[1]/2
    rad_phi =(phi*np.pi)/180
    x01 = s*np.cos(rad_phi)-np.tan(rad_varphi)*np.sin(rad_phi)
    x02 = s*np.sin(rad_phi)+np.tan(rad_varphi)*np.cos(rad_phi)

    return np.array([x02+int(size/2), x01+int(size/2)])

def CanRel_sino2img(WFpoints_sino, WFclasses_sino, size, sinogram_shape, num_angles):
    """ Compute the Wavefront set points and classes of the image from an sinogram

    Parameters
    ----------
    WFpoints_sino: numpy array, singular points positions of the image

    WFclasses_sino: list, singiular points orientations classes of the image

    size: integer, size of the image

    sinogram_shape: tuple, shape of the sinogram

    num_angles: integer, number of angle classes of the wavefront set

        Returns:
    ------------
    WFpoints, WFclasses: numpy array and list, wavefront set points and wavefront set classes
                                    of the image of a sinogram
    """
    # Compute the WF points in the sinogram
    WFpoints = []
    WFclasses = []
    for i in range(WFpoints_sino.shape[0]):
        WFpoints.append(point_sino2img(WFpoints_sino[i], WFclasses_sino[i]-1, size, sinogram_shape))
        WFclasses.append(np.array([(WFpoints_sino[i][1]-90)%180]))
    return np.array(WFpoints), WFclasses
