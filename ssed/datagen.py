# Module that generates and reads data for the WF extractor

# Import some libraries

import matplotlib.pyplot as plt
import imageio
import numpy as np
from PIL import Image
import numpy.random as rnd
from matplotlib.patches import Ellipse, Polygon
from itertools import compress

import cv2
import scipy.io
import matplotlib.image as img

__all__ = ('fig2data','ellipse_gen', 'point_gen', 'parallelogram_gen', 'angles_toclasses', 'classesto_angles',
            'rotate', 'Wavefrontset_ellipse_angles','Wavefrontset_ellipse_classes','Wavefrontset_point_angles',
            'Wavefrontset_point_classes','all_classes_rand','all_angles_rand','Wavefrontset_parallelogram_angles',
            'Wavefrontset_parallelogram_classes', 'Ellipse_class', 'ellipse_construct', 'Parallelogram_class',
            'parallelogram_construct','Point_class','point_construct','Distribution_class','Distribution_product',
            'sum_classes','Distribution_sum','Berkeley_sing_support','Berkeley_data_generation','point_to_class',
            'Distribution_Berkeley','SBD_sing_support','SBD_data_generation','Distribution_SBD')

# Function tha generates data from figure
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf


# Function that generates the data of the ellipse
def ellipse_gen(center, width, height, angle, size, opacity=1):
    # Generate the Ellipse figure
    fig = plt.figure(0,frameon=False,figsize=(1,1), dpi=size)
    ax =  plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    e = Ellipse(center, width, height, angle)
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(None)
    e.set_facecolor(np.zeros(3))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    fig.add_axes(ax)
    plt.axis('off')
    # Convert figure to data
    data = fig2data(fig)
    plt.close(fig)
    # Take just the first color entry
    data = data[:,:,1]
    # Normalize the data
    data = data/data.max()
    data = np.flip(data,0)
    return ((data-1)*opacity)+1


# Function that generates the point singularity
def point_gen(center, size, opacity=1):
    data = np.ones([size,size])
    data[center[0],center[1]]=1-opacity
    return data


# Function that generates the data of a parallelogram
def parallelogram_gen(origin, side1, side2, size, opacity = 1):
    # We rescale the points for the [1,1]^2 domain
    first_point = origin/size
    side1_scaled = side1/size
    side2_scaled = side2/size

    # Getting the points
    second_point= (first_point + side1_scaled)
    fourth_point= (first_point + side2_scaled)
    third_point = side1_scaled+side2_scaled+first_point

    x = [first_point[0],second_point[0],third_point[0],fourth_point[0]]
    y = [first_point[1],second_point[1],third_point[1],fourth_point[1]];

    fig = plt.figure(0,frameon=False,figsize=(1,1), dpi=size)
    ax =  plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    # Paralllogelogram
    p = Polygon(xy=list(zip(x,y)))
    ax.add_patch(p)
    p.set_alpha(None)
    p.set_facecolor(np.zeros(3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.add_axes(ax)
    plt.axis('off')
    # Convert figure to data
    data = fig2data(fig)
    plt.close(fig)
    # Take just the first color entry
    data = data[:,:,1]
    # Normalize the data
    data = data/data.max()
    data = np.flip(data,0)
    return ((data-1)*opacity)+1


# Wavefront set functions

# Function that generates the angles given by the number of scales and take
# Version that doesnt depend on the number of scales
def angles_toclasses(nClasses,angle):
    return int(np.floor((angle%180)/(180/nClasses)))+1

# Function that goes from classes to angles
# Version that doesnt depend on the number of scales
def classesto_angles(nClasses,classe):
    return (classe)*180/nClasses-1


# Rotation function
def rotate(origin, point, angle):
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



# Function that generates the points and angles (in degrees) of directions in the Wavefrontset of a ellipse
def Wavefrontset_ellipse_angles(center, width, height, angle):
    # Compute a and b (semiaxis)
    a = width/2
    b = height/2
    # Upper points
    x = [np.float64(x) for x in range(-int(a),int(a)+1)]
    uy = [(b/a)*np.sqrt(a**2-xi**2) for xi in x]
    # Lowere points
    ly = [-y for y in uy]
    # Concatenate the lists of points
    x = x+x
    y = uy+ly
    # Compute the corresponding angles in degress
    angles = [np.floor(np.arctan((y[i]*a**2)/(x[i]*b**2))*180/np.pi) for i in range(len(x))]
    # Rotated angles
    rot_angles = [np.array([(anglei+angle)%180]) for anglei in angles]
    # Rotated and translated points
    rot_trans_points = np.array([np.array(rotate([0,0],[x[i],y[i]],angle))+np.array(center) for i in range(len(x))])
    # Lets randomize the points and its angles to eleminate any structure given by construction
    permutations = np.random.permutation(len(rot_angles))
    rot_trans_points = rot_trans_points[permutations]
    rot_angles = [rot_angles[permutation] for permutation in permutations]
    return rot_trans_points, rot_angles



# Function that generates the points and class of angles (in degrees) of directions in the Wavefrontset of a ellipse
# Version that doesnt deppend on number of scales
def Wavefrontset_ellipse_classes(center, width, height, angle, nClasses):
    # Compute a and b (semiaxis)
    a = width/2
    b = height/2
    # Upper points
    x = [np.float64(x) for x in range(-int(a),int(a)+1)]
    uy = [(b/a)*np.sqrt(a**2-xi**2) for xi in x]
    # Lowere points
    ly = [-y for y in uy]
    # Concatenate the lists of points
    x = x+x
    y = uy+ly
    # Compute the corresponding angles in degress
    angles = [np.arctan((y[i]*a**2)/(x[i]*b**2))*180/np.pi for i in range(len(x))]
    # Rotated angles
    rot_angles_classes = [np.array([angles_toclasses(nClasses,(anglei+angle)%180)]) for anglei in angles]
    # Rotated and translated points
    rot_trans_points = np.array([np.array(rotate([0,0],[x[i],y[i]],angle))+np.array(center) for i in range(len(x))])
     # Lets randomize the points and its angles to eleminate any structure given by construction
    permutations = np.random.permutation(len(rot_angles_classes))
    rot_trans_points = rot_trans_points[permutations]
    rot_angles_classes = [rot_angles_classes[permutation] for permutation in permutations]
    return rot_trans_points, rot_angles_classes


# Function that generates the points and angles (in degrees) of directions in the Wavefrontset of a point
# Version independent of number of scales
def Wavefrontset_point_angles(center,nClasses):

    WFangles = np.array([i*180/nClasses for i in range(nClasses)])

    # Lets randomize the points and its angles to eleminate any structure given by construction
    permutations = np.random.permutation(len(WFangles))
    WFangles = WFangles[permutations]

    return np.array(center), [WFangles]


# Function that generates the points and classes of angles (in degrees) of directions in the Wavefrontset of a ellipse
def Wavefrontset_point_classes(center, nClasses):

    WFclasses = np.array([i+1 for i in range(nClasses)])

     # Lets randomize the points and its angles to eleminate any structure given by construction
    permutations = np.random.permutation(len(WFclasses))
    WFclasses = WFclasses[permutations]

    return np.array(center),  [WFclasses]



# Function that generates a random permutation of all the classes in a distribution
def all_classes_rand(nClasses):
    return np.array([k+1 for k in
                     range(nClasses)])[np.random.permutation(nClasses)]




def all_angles_rand(nClasses):
    return np.array([float(k) for k in
                     range(nClasses)])[np.random.permutation(nClasses)]



# Version with no nScales
def Wavefrontset_parallelogram_angles(origin,side1,side2,nClasses):
   # Lets parametrize the points by side

    # Side 1
    points_side1 =[origin+k*(side1)/(np.linalg.norm(side1))
                            for k in range(int(np.linalg.norm(side1)))];
    # Lets compute the angles, the endpoints directions are all
    angles_side1 = [all_angles_rand(nClasses)]
    angles_side1 = angles_side1+[np.array([(np.arctan(side1[1]/side1[0])*180/np.pi+90)%180])]*(len(points_side1)-2)
    angles_side1 = angles_side1+[all_angles_rand(nClasses)]

    # Side 2
    points_side2 =[origin+k*(side2)/(np.linalg.norm(side2))
                            for k in range(int(np.linalg.norm(side2)))];
    # Lets compute the angles, the endpoints directions are all
    angles_side2 = [all_angles_rand(nClasses)]
    angles_side2 = angles_side2+[np.array([(np.arctan(side2[1]/side2[0])*180/np.pi+90)%180])]*(len(points_side2)-2)
    angles_side2 = angles_side2+[all_angles_rand(nClasses)]

    # Side 3
    points_side3 =[origin+side1+k*(side2)/(np.linalg.norm(side2))
                            for k in range(int(np.linalg.norm(side2)))];
    # Lets compute the angles, the endpoints directions are all
    angles_side3 = [all_angles_rand(nClasses)]
    angles_side3 = angles_side3+[np.array([(np.arctan(side2[1]/side2[0])*180/np.pi+90)%180])]*(len(points_side2)-2)
    angles_side3 = angles_side3+[all_angles_rand(nClasses)]

    # Side 4
    points_side4 =[origin+side2+k*(side1)/(np.linalg.norm(side1))
                            for k in range(int(np.linalg.norm(side1)))];
    # Lets compute the angles, the endpoints directions are all
    angles_side4 = [all_angles_rand(nClasses)]
    angles_side4 = angles_side4+[np.array([(np.arctan(side1[1]/side1[0])*180/np.pi+90)%180])]*(len(points_side1)-2)
    angles_side4 = angles_side4+[all_angles_rand(nClasses)]

    # Append points and angles
    points = np.array(points_side1+points_side2+points_side3+points_side4)
    angles = angles_side1+angles_side2+angles_side3+angles_side4

    # Lets randomize the points and its angles to eleminate any structure given by construction
    permutations = np.random.permutation(len(angles))

    points = points[permutations]
    angles = [angles[permutation] for permutation in permutations]

    return points, angles



# Version independent of nScales
def Wavefrontset_parallelogram_classes(origin,side1,side2,nClasses):
        # Lets parametrize the points by side

    # Side 1
    points_side1 =[origin+k*(side1)/(np.linalg.norm(side1))
                            for k in range(int(np.linalg.norm(side1)))];
    # Lets compute the classes, the endpoints directions are all
    classes_side1 = [all_classes_rand(nClasses)]
    classes_side1 = classes_side1+[np.array([angles_toclasses(nClasses,(np.arctan(side1[1]/side1[0])*180/np.pi+90)%180)])]*(len(points_side1)-2)
    classes_side1 = classes_side1+[all_classes_rand(nClasses)]

    # Side 2
    points_side2 =[origin+k*(side2)/(np.linalg.norm(side2))
                            for k in range(int(np.linalg.norm(side2)))];
    # Lets compute the classes, the endpoints directions are all
    classes_side2 = [all_classes_rand(nClasses)]
    classes_side2 = classes_side2+[np.array([angles_toclasses(nClasses,(np.arctan(side2[1]/side2[0])*180/np.pi+90)%180)])]*(len(points_side2)-2)
    classes_side2 = classes_side2+[all_classes_rand(nClasses)]

    # Side 3
    points_side3 =[origin+side1+k*(side2)/(np.linalg.norm(side2))
                            for k in range(int(np.linalg.norm(side2)))];
    # Lets compute the classes, the endpoints directions are all
    classes_side3 = [all_classes_rand(nClasses)]
    classes_side3 = classes_side3+[np.array([angles_toclasses(nClasses,(np.arctan(side2[1]/side2[0])*180/np.pi+90)%180)])]*(len(points_side2)-2)
    classes_side3 = classes_side3+[all_classes_rand(nClasses)]

    # Side 4
    points_side4 =[origin+side2+k*(side1)/(np.linalg.norm(side1))
                            for k in range(int(np.linalg.norm(side1)))];

    # Lets compute the classes, the endpoints directions are all
    classes_side4 = [all_classes_rand(nClasses)]
    classes_side4 = classes_side4+[np.array([angles_toclasses(nClasses,(np.arctan(side1[1]/side1[0])*180/np.pi+90)%180)])]*(len(points_side1)-2)
    classes_side4 = classes_side4+[all_classes_rand(nClasses)]

    # Append points and angles
    points = np.array(points_side1+points_side2+points_side3+points_side4)
    classes = classes_side1+classes_side2+classes_side3+classes_side4

    # Lets randomize the points and its angles to eleminate any structure given by construction
    permutations = np.random.permutation(len(classes))

    points = points[permutations]
    classes = [classes[permutation] for permutation in permutations]


    return points, classes



class Ellipse_class:
    def __init__(self, center, width, height, angle, opacity, array, nClasses,
                  WFpoints, WFclasses, distrtype):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.opacity = opacity
        self.array = array
        self.nClasses = nClasses
        self.WFpoints = WFpoints
        self.WFclasses = WFclasses
        self.distrtype = distrtype

def ellipse_construct(center, width, height, angle, opacity, size, nClasses):
    array = ellipse_gen(center, width, height, angle, size, opacity)
    WFpoints, WFclasses = Wavefrontset_ellipse_classes(center, width, height, angle,nClasses)
    # Take out points outside the domain
    fil = (WFpoints[:,0]<size)*(WFpoints[:,1]<size)
    WFpoints = WFpoints[fil,:]
    WFclasses = list(compress(WFclasses,fil))
    return Ellipse_class(np.array(center), width, height, angle, opacity, array, nClasses,
                             WFpoints, WFclasses, 'ellipse')



class Parallelogram_class:
    def __init__(self, origin, side1, side2, opacity, array, nClasses,
                 WFpoints, WFclasses, distrtype):
        self.origin = origin
        self.side1 = side1
        self.side2 = side2
        self.opacity = opacity
        self.array = array
        self.nClasses = nClasses
        self.WFpoints = WFpoints
        self.WFclasses = WFclasses
        self.distrtype = distrtype



def parallelogram_construct(origin, side1, side2, size, opacity, nClasses):
    array = parallelogram_gen(origin, side1, side2, size, opacity)
    WFpoints, WFclasses = Wavefrontset_parallelogram_classes(origin,side1,side2, nClasses)
    # Take out points outside the domain
    fil = (WFpoints[:,0]<size)*(WFpoints[:,1]<size)
    WFpoints = WFpoints[fil,:]
    WFclasses = list(compress(WFclasses,fil))
    return Parallelogram_class(np.array(origin), np.array(side1), np.array(side2), opacity, array, nClasses,
                                   WFpoints, WFclasses, 'parallelogram')




class Point_class:
    def __init__(self, center, opacity, array, nClasses,
                 WFpoints, WFclasses, distrtype):
        self.center = center
        self.opacity = opacity
        self.array = array
        self.WFpoints = WFpoints
        self.WFclasses = WFclasses
        self.distrtype = distrtype



def point_construct(center, size, opacity, nClasses):
    array = point_gen(center, size, opacity)
    WFpoints, WFclasses = Wavefrontset_point_classes(center, nClasses)
    return Point_class(np.array(center), opacity, array, nClasses, WFpoints, WFclasses, 'point')



class Distribution_class:
    def __init__(self, array, nClasses, WFpoints, WFclasses):
        self.array = array
        self.nClasses = nClasses
        self.WFpoints = WFpoints
        self.WFclasses = WFclasses



def Distribution_product(distribution1, distribution2):
    # Number of classes
    nClasses = distribution1.nClasses
    # Check where the points are close and apply product theorem
    index_repetition = []
    if len(distribution1.WFpoints)<len(distribution2.WFpoints):
        for i in range(len(distribution1.WFpoints)):
            idx_rep = list(np.where(np.linalg.norm(distribution1.WFpoints[i]
                                                   -distribution2.WFpoints,axis=1)<3)[0])
            if len(idx_rep)>0:
                distribution1.WFclasses[i]=all_classes_rand(nClasses)
                for idx in idx_rep:
                    distribution2.WFclasses[idx]=all_classes_rand(nClasses)
    else:
        for i in range(len(distribution2.WFpoints)):
            idx_rep = list(np.where(np.linalg.norm(distribution2.WFpoints[i]
                                                   -distribution1.WFpoints,axis=1)<3)[0])
            if len(idx_rep)>0:
                distribution2.WFclasses[i]=all_classes_rand(nClasses)
                for idx in idx_rep:
                    distribution1.WFclasses[idx]=all_classes_rand(nClasses)
    array = 1-(1-distribution1.array)*(1-distribution2.array)
    # Lets normalize the array
    array = array/array.max()
    WFpoints = np.concatenate((distribution1.WFpoints,distribution2.WFpoints))
    WFclasses = distribution1.WFclasses+distribution2.WFclasses
    return Distribution_class(array,nClasses,WFpoints,WFclasses)


def sum_classes(class1,class2):
    return np.array(list(set(np.concatenate((class1,class2)))))


def Distribution_sum(distribution1, distribution2):
    # Number of classes
    nClasses = distribution1.nClasses
    # Check where the points are close and apply product theorem
    index_repetition = []
    if len(distribution1.WFpoints)<len(distribution2.WFpoints):
        for i in range(len(distribution1.WFpoints)):
            idx_rep = list(np.where(np.linalg.norm(distribution1.WFpoints[i]
                                                   -distribution2.WFpoints,axis=1)<3)[0])
            if len(idx_rep)>0:
                distribution1.WFclasses[i]= sum_classes(distribution1.WFclasses[i],distribution2.WFclasses[idx_rep[0]])
                for idx in idx_rep:
                    distribution2.WFclasses[idx]=sum_classes(distribution1.WFclasses[i],distribution2.WFclasses[idx])
    else:
        for i in range(len(distribution2.WFpoints)):
            idx_rep = list(np.where(np.linalg.norm(distribution2.WFpoints[i]
                                                   -distribution1.WFpoints,axis=1)<3)[0])
            if len(idx_rep)>0:
                distribution2.WFclasses[i]=sum_classes(distribution2.WFclasses[i],distribution1.WFclasses[idx_rep[0]])
                for idx in idx_rep:
                    distribution1.WFclasses[idx]=sum_classes(distribution2.WFclasses[i],distribution1.WFclasses[idx])
    array = 1-(2-distribution1.array-distribution2.array)
    # Lets normalize the array
    array = array/array.max()
    WFpoints = np.concatenate((distribution1.WFpoints,distribution2.WFpoints))
    WFclasses = distribution1.WFclasses+distribution2.WFclasses
    return Distribution_class(array,nClasses,WFpoints,WFclasses)


## Berkeley dataset

# Class for the Berkeley data set singular support
class Berkeley_sing_support:
    def __init__(self, image, boundary, corners):
        self.image = image
        self.boundary = boundary
        self.corners = corners

# Generate the data for the berkeley data set
def Berkeley_data_generation(path,file, dataset, size):
    # Reading the image
    image_path = path+'images/'+dataset+'/'
    image = img.imread(image_path+file)
    image = cv2.resize(image, dsize=(size,size), interpolation=cv2.INTER_CUBIC)

    # Reading the boudary and segmentation images
    bound_path = path+'groundTruth/'+dataset+'/'
    file_bound = file.replace('.jpg','')+'.mat'
    bound_segment = scipy.io.loadmat(bound_path+file_bound)['groundTruth'][0,4][0][0]

    bound = bound_segment[1]
    segment = bound_segment[0]

    # Resize the boundary and segmentation images
    bound = cv2.resize(bound, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
    segment = cv2.resize(segment, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

    # Extracting the points from Boundaries
    boundary = np.argwhere(bound == 1)
    boundary = np.flip(boundary,1)

    # Extranting the corners with Harris corner detector
    gray = np.float32(segment)
    dst = cv2.cornerHarris(gray,2,3,0.14)
    dst = cv2.dilate(dst,None)

    corners = np.argwhere((dst>0.01*dst.max())*1==1)
    corners = np.flip(corners,1)

    return Berkeley_sing_support(image, boundary, corners)

# Takes a point on the data and gives the corresponding class
def point_to_class(data, point_i, nClasses):
    point = data.boundary[point_i]
    # Extract the close-by points
    a = data.boundary[(data.boundary[:,0] < point[0]+8)*(
        data.boundary[:,0] > point[0]-8)*(data.boundary[:,1]
                                          < point[1]+8)*(data.boundary[:,1] > point[1]-8)]
    indexpoint = [(ai == point).all() for ai in a]

    # Compute the tangent to the curve
    dx_dt = np.gradient(a[:, 0])
    dy_dt = np.gradient(a[:, 1])
    velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    tangent = np.array([1/ds_dt] * 2).transpose() * velocity

    # Compute the normal to the curve
    normal = np.flip(tangent,1)
    normal[:,0] = -normal[:,0]

    normal_point = normal[indexpoint][0]

    #Compute the angel and then the class
    if normal_point[0] == 0:
        angle = (np.arctan(float('inf'))*180/np.pi)%180
    else:
        angle = (np.arctan(normal_point[1]/normal_point[0])*180/np.pi)%180

    return np.array([angles_toclasses(nClasses,int(angle))])

# Distribution generator for Berkeley dataset
def Distribution_Berkeley(path, file, dataset, size, nClasses):
    data = Berkeley_data_generation(path,file, dataset, size)
    array = data.image
    array1 = array[:,:,0]
    array2 = array[:,:,1]
    array3 = array[:,:,2]

    # Normalize the data
    array1 = array1 / array1.max()
    array2 = array2 / array2.max()
    array3 = array3 / array3.max()

    # Rescaling the interval
    array1 = (array1-array1.min())/(array1.max()-array1.min())
    array2 = (array2-array2.min())/(array2.max()-array2.min())
    array3 = (array3-array3.min())/(array3.max()-array3.min())

    WFpoints = data.boundary

    WFclasses = []
    # Adding corners
    for point_i in range(len(WFpoints)):
        if WFpoints[point_i] in data.corners:
            WFclasses.append(all_classes_rand(180))
        else:
            WFclasses.append(point_to_class(data, point_i, nClasses))
    return [Distribution_class(array1,nClasses,WFpoints,WFclasses),
            Distribution_class(array2,nClasses,WFpoints,WFclasses),
            Distribution_class(array3,nClasses,WFpoints,WFclasses)]

## SBD dataset

# Class for the SBD data set singular support
class SBD_sing_support:
    def __init__(self, image, boundary, corners):
        self.image = image
        self.boundary = boundary
        self.corners = corners

# Generate the data for the SBD data set
def SBD_data_generation(file, size):
    # Reading the image
    image_path = './SBD/dataset/img/'
    image = img.imread(image_path+file)
    image = cv2.resize(image, dsize=(size,size), interpolation=cv2.INTER_CUBIC)

    # Reading the boudary and segmentation images
    bound_path = './SBD/dataset/inst/'
    file_bound = file.replace('.jpg','')+'.mat'
    bound = scipy.io.loadmat(bound_path+file_bound)['GTinst']['Boundaries'][0][0][0][0].todense()
    segment = scipy.io.loadmat(bound_path+file_bound)['GTinst']['Segmentation'][0][0]

    # Resize the boundary and segmentation images
    bound = cv2.resize(bound, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
    segment = cv2.resize(segment, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

    # Extracting the points from Boundaries
    boundary = np.argwhere(bound == 1)
    boundary = np.flip(boundary,1)

    # Extranting the corners with Harris corner detector
    gray = np.float32(segment)
    dst = cv2.cornerHarris(gray,2,3,0.14)
    dst = cv2.dilate(dst,None)

    corners = np.argwhere((dst>0.01*dst.max())*1==1)
    corners = np.flip(corners,1)

    return SBD_sing_support(image, boundary, corners)

# Distributiuon generator for SBD dataset
def Distribution_SBD(file, size, nClasses):
    data = SBD_data_generation(file, size)
    array = data.image
    array1 = array[:,:,0]
    array2 = array[:,:,1]
    array3 = array[:,:,2]

    # Normalize the data
    array1 = array1 / array1.max()
    array2 = array2 / array2.max()
    array3 = array3 / array3.max()

    # Rescaling the interval
    array1 = (array1-array1.min())/(array1.max()-array1.min())
    array2 = (array2-array2.min())/(array2.max()-array2.min())
    array3 = (array3-array3.min())/(array3.max()-array3.min())

    WFpoints = data.boundary

    WFclasses = []
    # Adding corners
    for point_i in range(len(WFpoints)):
        if WFpoints[point_i] in data.corners:
            WFclasses.append(all_classes_rand(180))
        else:
            WFclasses.append(point_to_class(data, point_i, nClasses))
    return [Distribution_class(array1,nClasses,WFpoints,WFclasses),
            Distribution_class(array2,nClasses,WFpoints,WFclasses),
            Distribution_class(array3,nClasses,WFpoints,WFclasses)]
