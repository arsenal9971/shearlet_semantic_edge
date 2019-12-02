# Script that generates the training/test/validation batches for the different datasets

# Import libraries
import datagen as dg
import numpy as np
import shearlab
import matplotlib.pyplot as plt
from itertools import compress
from scipy.signal import convolve2d

__all__ = ('center_origin_gen','width_height_side_gen','angle_gen','opacity_gen','ellipse_random',
            'parallelogram_random','data_random','data_random_smooth','gen_points','multi_onehot','gen_batch',
            'gen_batch_smooth','gen_batch_berkeley','gen_batch_SBD')

#### Ellipses/parallelograms dataset

# Functions that generates randomly the parameters for the data generation
def center_origin_gen(size):
    return [np.random.randint(low = 0+int(size/10), high = size-int(size/10)),
            np.random.randint(low = 0+int(size/10), high = size-int(size/10))]

def width_height_side_gen(size):
    return [np.random.randint(low = 10, high = int(size/3)),
            np.random.randint(low = 10, high = int(size/3))]

def angle_gen():
    return np.random.randint(low = 0, high = 180)

def opacity_gen():
    return np.random.uniform(0.2,1.0)

# Functions that generates the data randomly for the ellipses/parallelograms dataset
def ellipse_random(size, nClasses):
    # Random parameters for the ellipse
    center = center_origin_gen(size)
    width, height = width_height_side_gen(size)
    angle = angle_gen()
    opacity = opacity_gen()
    return dg.ellipse_construct(center, width, height, angle, opacity, size,nClasses)

def parallelogram_random(size, nClasses):
    origin = np.array(center_origin_gen(size))
    side1 = np.array(width_height_side_gen(size))
    side2 = np.array(width_height_side_gen(size))
    opacity = opacity_gen()
    return dg.parallelogram_construct(origin, side1, side2, size, opacity,nClasses)

# Function that generates data randomly for both ellipses and parallelograms
def data_random(size, nClasses, nDistr):
    # Pick the distribution type randomly
    if np.random.randint(low = 0, high = 2) == 0:
        distribution = ellipse_random(size, nClasses)
    else:
        distribution = parallelogram_random(size, nClasses)
    # Generates some sum
    for i in range(nDistr):
        # Pick the distribution type randomly
        if np.random.randn()<=0.5:
            distribution = dg.Distribution_sum(
                parallelogram_random(size, nClasses), distribution)
        else:
            distribution = dg.Distribution_sum(
                ellipse_random(size, nClasses), distribution)

    # Normalize the data
    distribution.array = distribution.array / distribution.array.max()
    # Rescaling the interval
    distribution.array = (distribution.array-
                          distribution.array.min())/(distribution.array.max()-
                                                     distribution.array.min())
    return distribution

# Function that generates a smoothed version of the ellipse-parallelogram randomly
def data_random_smooth(size, nClasses, nDistr):
    distribution = data_random(size, nClasses, nDistr)
    # With a kernel
    t = 1 - np.abs(np.linspace(-1, 1, 21))
    kernel = t.reshape(21, 1) * t.reshape(1, 21)
    kernel /= kernel.sum()
    # convolve 2d the kernel with each channel
    smooth = convolve2d(distribution.array[:,:], kernel, mode='same')
    # Change the array in distribtuion for the smooth version
    distribution.array = smooth
    return distribution

# Function that extracts generates important points
def gen_points(distribution, size_patch):
    size = distribution.array.shape[0]
    nClasses = distribution.nClasses

    # Bounds for not extracting the boundary
    upper_bound = (distribution.WFpoints.astype(int)[:,0]>=(size-
                                                            int(size_patch/2))) | (distribution.WFpoints.astype(int)[:,1]>=(size-
                                                                                                                            int(size_patch/2)))
    lower_bound = (distribution.WFpoints.astype(int)[:,0]<=(
        int(size_patch/2))) | (distribution.WFpoints.astype(int)[:,1]<=(
        int(size_patch/2)))

    bound = upper_bound | lower_bound
    bound = np.invert(bound)
    classes = []
    points = list(distribution.WFpoints.astype(int)[bound])
    classes += list(compress(distribution.WFclasses, bound))
    # Points that are not in the wavefront set
    no_points_x = list(set(([i for i in range(int(size_patch/2)+1,size-int(size_patch/2))]))-
                    set(np.array(points)[:,0]))
    no_points_y = list(set(([i for i in range(int(size_patch/2)+1,size-int(size_patch/2))]))-
                    set(np.array(points)[:,1]))
    no_points = []
    if len(no_points_x)!=0:
            no_points += [np.array([no_points_x[i],np.random.randint(low = int(size_patch/2)+1,
                                                                     high = size-int(size_patch/2))]) for i in range(min(len(no_points_x),int(len(points)*0.01)))]
    if len(no_points_y)!=0:
        no_points += [np.array([np.random.randint(low = int(size_patch/2)+1,
                                                  high = size-int(size_patch/2)),no_points_y[i]]) for i in range(min(len(no_points_y),int(len(points)*0.01)))]
    # We are going to assign a new class to the points that are not in the wavefront set
    if len(no_points)!=0:
        points += no_points
        classes += [np.array([nClasses+1])]*(len(no_points))
    return points, classes

# Function that generates one-hot multiclass vectors
def multi_onehot(classe, nClasses):
    one_hot = np.zeros(nClasses+1)
    one_hot[classe-1] = 1
    return one_hot

# Finally get the batch for the ellipses/parallelgrams
def gen_batch(size, nClasses, nDistr, shearletSystem, size_patch):
    # Create the distribution
    distribution = data_random(size, nClasses, nDistr)

    # Coefficients positive and negative
    coeffs_pos = shearlab.sheardec2D(1-distribution.array,shearletSystem)
    coeffs_neg = shearlab.sheardec2D(distribution.array,shearletSystem)

    # Get the points and classes
    points, classes = gen_points(distribution,size_patch)

    # Batch array
    batch_array = np.stack([coeffs_pos[(point[1]-int(size_patch/2)):(point[1]+int(size_patch/2)),
                      (point[0]-int(size_patch/2)):(point[0]+int(size_patch/2)),:] for point in points]
                      + [coeffs_neg[(point[1]-int(size_patch/2)):(point[1]+int(size_patch/2)),
                      (point[0]-int(size_patch/2)):(point[0]+int(size_patch/2)),:] for point in points]);

    # Batch label
    batch_label = np.stack([multi_onehot(classe, nClasses) for classe in classes]*2)

    # Free memory
    coeffs_neg = 0
    coeffs_pos = 0
    return batch_array, batch_label

# The smoothed version of gen batch
def gen_batch_smooth(size, nClasses, nDistr, shearletSystem, size_patch):
    # Create the distribution
    distribution = data_random_smooth(size, nClasses, nDistr)

    # Coefficients positive and negative
    coeffs_pos = shearlab.sheardec2D(1-distribution.array,shearletSystem)
    coeffs_neg = shearlab.sheardec2D(distribution.array,shearletSystem)

    # Get the points and classes
    points, classes = gen_points(distribution,size_patch)

    # Batch array
    batch_array = np.stack([coeffs_pos[(point[1]-int(size_patch/2)):(point[1]+int(size_patch/2)),
                      (point[0]-int(size_patch/2)):(point[0]+int(size_patch/2)),:] for point in points]
                      + [coeffs_neg[(point[1]-int(size_patch/2)):(point[1]+int(size_patch/2)),
                      (point[0]-int(size_patch/2)):(point[0]+int(size_patch/2)),:] for point in points]);

    # Batch label
    batch_label = np.stack([multi_onehot(classe, nClasses) for classe in classes]*2)

    # Free memory
    coeffs_neg = 0
    coeffs_pos = 0
    return batch_array, batch_label

### Berkeley dataset

from os import listdir
from os.path import isfile, join
from random import randint

# Function that generates the batch for berkeley dataset
def gen_batch_berkeley(path, dataset, size, nClasses, shearletSystem, size_patch):
    files = [f for f in listdir(path+'images/'+dataset+'/') if isfile(join(path+'images/'+dataset+'/', f))]

    # Random file
    i = randint(0,len(files))
    file = files[i]

    # Generate the distributions
    distribution1, distribution2, distribution3 = dg.Distribution_Berkeley(path, file, dataset, size, nClasses)

    # Coefficients positive and negative randomly
    if randint(0,1)==1:
        coeffs_pos = shearlab.sheardec2D(1-distribution1.array,shearletSystem)
        coeffs_neg = shearlab.sheardec2D(distribution2.array,shearletSystem)
    else:
        coeffs_pos = shearlab.sheardec2D(1-distribution2.array,shearletSystem)
        coeffs_neg = shearlab.sheardec2D(distribution1.array,shearletSystem)

    # Get the points and classes
    points, classes = gen_points(distribution1,size_patch)

    # Batch array
    batch_array = np.stack([coeffs_pos[(point[1]-int(size_patch/2)):(point[1]+int(size_patch/2)),
                      (point[0]-int(size_patch/2)):(point[0]+int(size_patch/2)),:] for point in points]
                      + [coeffs_neg[(point[1]-int(size_patch/2)):(point[1]+int(size_patch/2)),
                      (point[0]-int(size_patch/2)):(point[0]+int(size_patch/2)),:] for point in points]);

    # Batch label
    batch_label = np.stack([multi_onehot(classe, nClasses) for classe in classes]*2)

    return batch_array, batch_label

# Function that generates the batch for SBD dataset
def gen_batch_SBD(path, size, nClasses, shearletSystem, size_patch):
    files = [f for f in listdir(path) if isfile(join(path, f))]

    # Random file
    i = randint(0,len(files))
    file = files[i]

    # Generate the distributions
    distribution1, distribution2, distribution3 = dg.Distribution_SBD(file, size, nClasses)

    # Coefficients positive and negative randomly
    if randint(0,1)==1:
        coeffs_pos = shearlab.sheardec2D(1-distribution1.array,shearletSystem)
        coeffs_neg = shearlab.sheardec2D(distribution2.array,shearletSystem)
    else:
        coeffs_pos = shearlab.sheardec2D(1-distribution2.array,shearletSystem)
        coeffs_neg = shearlab.sheardec2D(distribution1.array,shearletSystem)

    # Get the points and classes
    points, classes = gen_points(distribution1,size_patch)

    # Batch array
    batch_array = np.stack([coeffs_pos[(point[1]-int(size_patch/2)):(point[1]+int(size_patch/2)),
                      (point[0]-int(size_patch/2)):(point[0]+int(size_patch/2)),:] for point in points]
                      + [coeffs_neg[(point[1]-int(size_patch/2)):(point[1]+int(size_patch/2)),
                      (point[0]-int(size_patch/2)):(point[0]+int(size_patch/2)),:] for point in points]);

    # Batch label
    batch_label = np.stack([multi_onehot(classe, nClasses) for classe in classes]*2)

    return batch_array, batch_label
