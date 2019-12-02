"""
The :mod:`ray_transform` contains a parallel beam ray transform and some
basic inverse transforms"""
# Author: Hector Loarca, Ingo Guehring
import odl
import math
import numpy as np


__all__ = ('createRayTrafo', 'filteredBackProjection')

# for examples on how to use the odl package for ray transform stuff, see:
# https://github.com/odlgroup/odl/tree/master/examples/tomo


def createRayTrafo(size, meas_p_angle, missing_wedge_angle, full=True,
                   impl='skimage', **keyword_parameters):
    """Create Ray transform

    Parameters
    -----------
    size : even integer, size of input space domain (size x size)

    meas_p_angle : real number, measurements per angle, e.g.:
        (1) meas_p_angle = 0.5 ==> two measurements every angle
        (2) meas_p_angle = 2   ==> one measurement every two angles

    missing_wedge_angle : 0 <= integer <= 180, angle of the missing wedge,
        applied on both sides

    full : boolean, if False ==> normal missing wedge tomo
        if True ==> full 180 degree measurements translated by
        missing_wedge_angle

    impl : {`None`, 'astra_cuda', 'astra_cpu', 'skimage'}, optional
            Implementation back-end for the transform. Supported back-ends:
            - ``'astra_cuda'``: ASTRA toolbox, using CUDA, 2D or 3D
            - ``'astra_cpu'``: ASTRA toolbox using CPU, only 2D
            - ``'skimage'``: scikit-image, only 2D parallel with square
              reconstruction space.

    detector_sample : optional, translation domain of ray transform.

    Returns:
    -----------

    space_image : discretized image space

    output_dim : tuple, (y_1, y_2), y_1 is the number of measurements in the
        angle domain, y_2 the number of translations of the beam

    ray_trafo : ray transform operator
    """
    assert size % 2 == 0, "use an even size"

    if ('detector_sample' not in keyword_parameters):
        detector_sample = size

    # Reconstruction space: discretized functions on the rectangle
    # min_pt, max_pt only matters for plotting, shape is important
    space_image = odl.uniform_discr(
        min_pt=[-1, -1], max_pt=[1, 1],
        shape=[size, size], dtype='float32')

    # Make a parallel beam geometry with flat detector
    n_angles = (180 - 2 * missing_wedge_angle * (not full)) * \
        (1 / meas_p_angle)
    assert n_angles % 1 == 0, "range of angles * 1/meas_p_angle =!  integer"
    n_angles = math.floor(n_angles)

    mis_wedge_pi = missing_wedge_angle * np.pi/180
    angle_partition = odl.uniform_partition(
        mis_wedge_pi, np.pi + mis_wedge_pi * full - mis_wedge_pi * (not full),
        n_angles, nodes_on_bdry=False)

    # Detector: uniformly sampled
    # We only use the image domain as detector domain for translations
    offset = .5
    detector_partition = odl.uniform_partition(-1-offset, 1+offset,
                                               detector_sample)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    # Ray transform (= forward projection)
    ray_trafo = odl.tomo.RayTransform(space_image, geometry, impl=impl)

    # Compute output dimension of ray trasform
    y_1 = geometry.partition.cell_boundary_vecs[0].shape[0] - 1
    y_2 = geometry.partition.cell_boundary_vecs[1].shape[0] - 1
    output_dim = (y_1, y_2)
    return ray_trafo


def filteredBackProjection(y, ray_trafo):
    fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo)
    return fbp(y)

