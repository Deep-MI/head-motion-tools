"""
Functions for working with pointcloud data as numpy arrays
"""

import numpy as np
import math
try:
    import vtk # for .stl output
    from head_motion_tools.visualization import VtkTools
except Exception as e:
    #print('Vtk not found: Pointcloud writing disabled')
    NO_VTK = True


def _int_list_from_number_string(in_str):
    """
    creates a list of integers from a space seperated string
    """
    in_str = in_str.split(' ')
    out_lst = []
    for i in in_str:
        out_lst.append(int(i))

    return out_lst


def loadPcd(filename, verbose=False):
    """
    loads a .pcd file into a numpy array

    filename    relative/absolute filepath to the .pcd file
    verbose     set to True to get debug/error messages
    return      returns a numpy array with where each cell us a point, 
                which has the file format specified in the pcd header
    NOTE: only supports float32 numbers with field size 1 right now
    """

    with open(filename,'rb') as f:

        line_counter = 0

        while True:

            # prevent endless loop
            line_counter = line_counter + 1
            if line_counter > 30:
                if verbose:
                    print('could not read header ("DATA" field missing?)')
                return None


            # read one header line
            line = f.readline().decode('ascii')

            # skip comments
            if line[0] == '#':
                line_counter = line_counter-1
                continue


            field,content = line.split(' ',1)


            # Documentation from http://pointclouds.org/documentation/tutorials/pcd_file_format.php
            # the order of fields in the pcd should be the same as in the code
            try:
                # VERSION - specifies the PCD file version
                if field == 'VERSION':
                    continue

                # FIELDS - specifies the name of each dimension/field that a point can have
                elif field == 'FIELDS':
                    FIELDS = content.strip().split()

                # SIZE - specifies the size of each dimension in bytes
                elif field == 'SIZE':
                    SIZE = _int_list_from_number_string(content)

                # TYPE - specifies the type of each dimension as a char. The current accepted types are:
                #    I - represents signed types int8 (char), int16 (short), and int32 (int)
                #    U - represents unsigned types uint8 (unsigned char), uint16 (unsigned short), uint32 (unsigned int)
                #    F - represents float types
                elif field == 'TYPE':
                    TYPE = content.strip().split(' ')

                # COUNT - specifies how many elements does each dimension have. 
                #         For example, x data usually has 1 element, but a feature descriptor like the VFH has 308. 
                #         Basically this is a way to introduce n-D histogram descriptors at each point, and treating them as a single contiguous block of memory. 
                #         By default, if COUNT is not present, all dimensions count is set to 1.
                elif field == 'COUNT':
                    COUNT = _int_list_from_number_string(content)

                # WIDTH - specifies the width of the point cloud dataset in the number of points. WIDTH has two meanings:
                #    it can specify the total number of points in the cloud (equal with POINTS see below) for unorganized datasets;
                #    it can specify the width (total number of points in a row) of an organized point cloud dataset.
                elif field == 'WIDTH':
                    WIDTH = int(content)

                # HEIGHT - specifies the height of the point cloud dataset in the number of points. HEIGHT has two meanings:
                #   it can specify the height (total number of rows) of an organized point cloud dataset;
                #   it is set to 1 for unorganized datasets (thus used to check whether a dataset is organized or not).
                elif field == 'HEIGHT':
                    HEIGHT = int(content)

                # VIEWPOINT - specifies an acquisition viewpoint for the points in the dataset. This could potentially be later on used for
                #             building transforms between different coordinate systems, or for aiding with features such as surface normals, that need a consistent orientation.
                elif field == 'VIEWPOINT':
                    VIEWPOINT = _int_list_from_number_string(content)

                # POINTS - specifies the total number of points in the cloud.
                elif field == 'POINTS':
                    POINTS = int(content)

                # DATA - specifies the data type that the point cloud data is stored in. 
                #        As of version 0.7, two data types are supported: ascii and binary.
                elif field == 'DATA':
                    DATA = content.strip();
                    break;
                else:
                    if verbose:
                        print('unknown field --- recieved: ' + field + '.')
            except:
                if verbose:
                    print('couldnt decode field --- recieved: ' + content  + '.')


        #outout the header for debugging output
        if verbose:
            try:
                print('FIELDS: ' + str(FIELDS))
                print('SIZE: ' + str(SIZE))
                print('TYPE: ' + str(TYPE))
                print('COUNT: ' + str(COUNT))
                print('WIDTH: ' + str(WIDTH))
                print('HEIGHT: ' + str(HEIGHT))
                print('VIEWPOINT: ' + str(VIEWPOINT))
                print('POINTS: ' + str(POINTS))
                print('DATA: ' + str(DATA))
            except NameError:
                if verbose:
                    print('not all fields defined')


        # only one case supported right now
        # float32, field size 1
        if DATA == 'binary':
            if len(set(TYPE)) == 1 and TYPE[0] == 'F' or TYPE == ['F', 'F', 'F', 'U']:
                if len(set(SIZE)) == 1:
                        # create numpy type according to TYPE and SIZE fields
                        typelist = []
                        for i,field_name in enumerate(FIELDS):
                            # handle wrong field declaration of rb field as float
                            if field_name == 'rgb':
                                typelist.append((field_name,'<u4'))
                                continue
                            #  FIELDS=x,y,z....  TYPE=F   SIZE=4
                            typelist.append((field_name,'<f4'))
                        
                        dtype = np.dtype(typelist)
                        data = np.fromfile(f, dtype=dtype)
                else:
                    raise NotImplementedError('Dimension sizes not supported ' + str(SIZE))
            else:
                raise NotImplementedError('Type format ' + str(TYPE) + ' not supported')
        else:
            raise NotImplementedError(DATA + " not supported")

        assert(len(data)==POINTS)
        if verbose:
            print('read .pcd successfully')

        data = data.reshape((HEIGHT,WIDTH))
        r = ((data['rgb'] >> 16) & 0x0000ff)
        g = ((data['rgb'] >> 8) & 0x0000ff)
        b = ((data['rgb']) & 0x0000ff)

        #dataOut = np.empty((HEIGHT,WIDTH,6),dtype='<f4')
        dataOut = np.empty((HEIGHT,WIDTH,6),dtype=np.float32)


        dataOut[:,:,0] = data['x']
        dataOut[:,:,1] = data['y']
        dataOut[:,:,2] = data['z']

        #dataOut[:,:,:3] = np.array(data[['x','y','z']].tolist())
        dataOut[:,:,3:] = np.stack((r,g,b),axis=2)
        return dataOut


def writeHeader(pc_shape):
    """
    Gives standard pcd header which is adjusted to the pointclouds shape

    pc_shape - the shape of the numpy array representing the pointcloud
    return - string, that is the header of the pointcloud
    """

    template = """\
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {width}
HEIGHT {height}
VIEWPOINT 0 0 0 1 0 0 0
POINTS {number_of_points}
DATA binary
"""
    
    if len(pc_shape) == 2:
        height = 1
        width = pc_shape[0]
        number_of_points = width
        assert pc_shape[1] == 3
    elif len(pc_shape) == 3:
        height = pc_shape[0]
        width = pc_shape[1]
        number_of_points = height*width
        assert pc_shape[2] == 3
    else:
        raise NotImplementedError('shape of point cloud not supported - too many dimensions')


    template = template.replace("{number_of_points}",str(number_of_points))
    template = template.replace("{width}",str(width))
    template = template.replace("{height}",str(height))
    return template


def writePcd(pc, fname, verbose):
    """
    writes a pcd to file

    pc - pointcloud
    fname - filename/path
    verbose - give success message
    """
    with open(fname, 'w') as fileobj:

        header = writeHeader(pc.shape)
        fileobj.write(header)

    with open(fname, "ab") as fileobj:
        fileobj.write(pc.astype('<f4').tostring('C'))

    if verbose:
        print('wrote', fname)


def to3dArray(pc_data):
    """
    transforms the data structure into an array with the form (height,width,3) where "3" are the xyz coordinates

    pc_data - read .pcd file as input
    return - numpy array with desired shape
    """
    return pc_data[:,:,:3]

def to2dArray(pc_data,getColors=False, mask=None, structured_pc_coordinates=False):
    """
    convert structured pointcloud to unstructured pointcloud

    pc_data         structured pointcloud as 3d array
    getColors       return intesity values (bool)
    mask            mask to apply on structure before converting to 2d
    structured_pc_coordinates  whether to return an array of indices, that indicate the position on the structured grid

    return  unstructured pointcloud as 2d array
    """
    arr_dim = (pc_data.shape[0],pc_data.shape[1])
    if getColors:
        if mask is None:
            new_data = pc_data.reshape((arr_dim[0]*arr_dim[1],6))
        else:
            new_data = new_data[mask]
    else:
        new_data = pc_data[:,:,:3]
        if mask is None:
            new_data = new_data.reshape((arr_dim[0]*arr_dim[1],3))
        else:
            new_data = new_data[mask]

    # remove NaNs
    no_nan_mask = ~np.isnan(new_data[:,0])
    new_data = new_data[no_nan_mask]

    if structured_pc_coordinates:
        coordinates = np.indices(arr_dim)
        if mask is not None:
            coordinates = coordinates[:,mask]
        else:
            coordinates = coordinates.reshape((2,arr_dim[0]*arr_dim[1]))
        coordinates = coordinates[:, no_nan_mask]

    if getColors and structured_pc_coordinates:
        return new_data[:,:3],new_data[:,3], coordinates
    elif getColors:
        return new_data[:,:3],new_data[:,3]
    elif structured_pc_coordinates:
        return new_data, coordinates
    else:
        return new_data


def restructurePointcloud(pc_data, structured_pc_coordinates, pc_size=(256, 320, 3)):
    """
    restores stuctured pointcloud from coordinates

    pc_data                         unstructured pointcloud
    structured_pc_coordinates       indices mapping unstructured points to location in structured pc
    pc_size                         dimensions of output pointcloud

    """

    out_arr = np.empty(pc_size, dtype=float)
    out_arr.fill(np.nan)
    if pc_size[0] < np.max(structured_pc_coordinates[0]) or pc_size[1] < np.max(structured_pc_coordinates[1]):
        raise ValueError('output dimensions are too small for passed structure')

    out_arr[structured_pc_coordinates[0], structured_pc_coordinates[1]] = pc_data

    return out_arr



def thresholdPc(pc_data,pt_min,pt_max):
    """
    discard all points that are not within a cube spanned by the two given points
    pc_data - pointcloud as 2d or 3d array
    pt_max - point with highest x,y,z values
    pt_min - point with lowest x,y,z values
    """
    if len(pc_data.shape) == 2:
        return np.all((pt_min < pc_data[:,:3]) & (pc_data[:,:3] < pt_max),axis=1)
    elif len(pc_data.shape) == 3:
        return np.all((pt_min < pc_data[:,:,:3]) & (pc_data[:,:,:3] < pt_max),axis=2)
    else:
        raise NotImplementedError('dimension mismatch')


def getNanMap(pc_data,allNaN=False):
    """
    get a Map that indicates the NaN points in a structured point cloud

    allNaN - set to True spcifies, that all values should be NaN to count as not well defined, standard is any value
    return an array that indicates the points which are not well defined with 'True'
    """
    is_nan_arr = np.zeros(pc_data.shape,dtype='bool')

    if allNaN:
        nan_check = _all_is_nan_zero
    else:
        nan_check = lambda pt : np.isnan(pt[0]) #_is_nan

    for x in range(0,pc_data.shape[0]):
        for y in range(0,pc_data.shape[1]):
            if nan_check(pc_data[x,y]):
                is_nan_arr[x,y] = 1

    return is_nan_arr

#@jit(nopython=True)
def getDistance(x,y):
    """
    return distance between two points in the data format
    """
    #return np.linalg.norm(np.array((x[0]-y[0],x[1]-y[1],x[2]-y[2])), ord=None)
    return math.sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2) +((x[2]-y[2])**2))



def _is_nan(pt):
    """
    returns True if the point is not defined
    """
    return np.isnan(pt[0])# or np.isnan(pt[1]) or np.isnan(pt[2])


def _all_is_nan_zero(pt):
    """
    return True if the first three values are nan and the last is zero
    """
    return np.isnan(pt[0]) and np.isnan(pt[1]) and np.isnan(pt[2]) and pt[3] == 0.0


def pcdToMesh(data, outlierTreshold=10):
    """
    convert pcd data into mesh data object
    data - pcd data object from loadPCD. Expects to have the fields x,y,z
    outlierTreshold - the distance in which points should no longer be connected with a face
                        zero means that no correction will be done
    openmesh_output - output openmesh object

    return - 
        vertex_list     list of vertices with x,y,z coordinates
        facelist        list of triagnles, defined by vertex indices
        rgb_values      rgb_values of vertices

    NOTE: this will discard rgb value data, that is attached to nonexistent points
    """
    #TODO: use numpy masked arrays to do everything more elgegantly (and quick)
    
    # map pcd data point to mesh idx 
    pcd_to_mesh = dict()
    rgb_values = []
    facelist = []
    vertex_list = []

    nan_arr = np.array([np.nan,np.nan,np.nan])


    # check all points until second to last rows for possible edge connections in the next row/column
    for x in range(0,data.shape[0]-1):
        for y in range(0,data.shape[1]-1):
            if _is_nan(data[x,y]):
                continue

            # A----B
            # |    |
            # C----D

            idx_A = (x,y)
            idx_B = (x+1,y)
            idx_C = (x,y+1)
            idx_D = (x+1,y+1)

            pt_A = data[idx_A]
            pt_B = data[idx_B]
            pt_C = data[idx_C]
            pt_D = data[idx_D]


            # handle prevent points, that are far away from building a face
            if getDistance(pt_A,pt_B) > outlierTreshold:
                pt_B = nan_arr
            if getDistance(pt_A,pt_C) > outlierTreshold:
                pt_C = nan_arr
            if getDistance(pt_A,pt_D) > outlierTreshold:
                pt_D = nan_arr


            pcd_to_mesh[idx_A] = len(vertex_list)
            #vertex_list.append(np.array([pt_A['x'],pt_A['y'],pt_A['z']]))
            #rgb_values.append(pt_A['rgb'])
            vertex_list.append(np.array([pt_A[0],pt_A[1],pt_A[2]]))
            if len(pt_A) > 3:
                rgb_values.append(pt_A[3])


            dist_AD = getDistance(pt_A,pt_D)
            dist_CB = getDistance(pt_C,pt_B)

            # handle undefined points
            # possible to shorten this by one statement, with using np.nan < float -> False
            if np.isnan(dist_AD) and np.isnan(dist_CB):
                continue
            elif np.isnan(dist_AD):# and not np.isnan(dist_CB) (redundant)
                faceval = 'CB'
            elif np.isnan(dist_CB):# and not np.isnan(dist_AD) (redundant)
                faceval = 'AD'
            elif dist_AD < dist_CB:
                faceval = 'AD'
            else: # dist_AD > dist_CB
                faceval = 'CB'

            if faceval == 'CB':
                # CDB
                if not _is_nan(pt_D):
                    facelist.append([idx_C,idx_D,idx_B])
                # CBA
                if not _is_nan(pt_A):
                    facelist.append([idx_C,idx_B,idx_A])
            elif faceval == 'AD':
                # ACD
                if not _is_nan(pt_C):
                    facelist.append([idx_A,idx_C,idx_D])
                # ADB
                if not _is_nan(pt_B):
                    facelist.append([idx_A,idx_D,idx_B])
            else:
                raise NotImplementedError('this point should never be reached <.<')


    # handle the special case of the last row
    # include points, but do not add edges
    y = data.shape[1]-1
    for x in range(0,data.shape[0]):
        pt_A = data[x,y]

        if not _is_nan(pt_A):
            pcd_to_mesh[(x,y)] = len(vertex_list)
            vertex_list.append(np.array([pt_A[0],pt_A[1],pt_A[2]]))
            if len(pt_A) > 3:
                rgb_values.append(pt_A[3])

    x = data.shape[0]-1
    for y in range(0,data.shape[1]-1):

        pt_A = data[x,y]

        if not _is_nan(pt_A):
            pcd_to_mesh[(x,y)] = len(vertex_list)
            vertex_list.append(np.array([pt_A[0],pt_A[1],pt_A[2]]))
            if len(pt_A) > 3:
                rgb_values.append(pt_A[3])


    # convert indices in the list of faces to match the vertex list
    for i,face in enumerate(facelist):
        facelist[i] = (pcd_to_mesh[face[0]],pcd_to_mesh[face[1]],pcd_to_mesh[face[2]])

    # create OpenMesh object
    return vertex_list, facelist, rgb_values



def writeMesh(vertex_list,facelist,filename,verbose=False):
    """
    write mesh using openmesh
    """

    if NO_VTK:
        raise EnvironmentError('Vtk not found: Pointcloud writing disabled')

    poly = VtkTools.createPoly(vertex_list,facelist)

    polyMapper = vtk.vtkPolyDataMapper()
    polyMapper.SetInputData(poly)
     
    # Write the stl file to disk
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(filename)
    stlWriter.SetInputData(poly)
    stlWriter.Write()
     
    

    if verbose:
        print('wrote',filename)

