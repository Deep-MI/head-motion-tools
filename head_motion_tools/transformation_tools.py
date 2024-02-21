import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation

# ignore performance warnings for weighted average
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@njit()
def msDev(A, B=np.identity(4), x=np.zeros((1,3)), r=80):
    """
    Calculates the root mean square deviation of two homogenous transformations in 3d
    This distance is used in Jenkinson 1999 RMS deviation - tech report www.fmrib.ox.ac.uk/analysis/techrep .

    A       homogenous transformation matrix 
    B       homogenous transformation matrix (identity by default)
    x       sphere center (brain center, can be RAS center)
    r       sphere radius (head size)
    return  the root mean square deviation
    """
    assert(x.shape[0] == 1), 'x must be a 1x3 array'
    assert(x.shape[1] == 3), 'x must be a 1x3 array'

    A =  B @ np.linalg.inv(A) - np.identity(4)
    t = np.expand_dims(A[:3,3],0).T
    A = A[:3,:3]
    
    ret = (1/5)*(r**2)*np.trace(A.T@A) + (t+A@x.T).T @ (t+A@x.T)
    return ret.item()


@njit()
def rmsDev(A,B=np.identity(4),x=np.zeros((1,3)), r=80):
    return np.sqrt(msDev(A,B,x,r))


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape[1] == B.shape[1]

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def avg_transformation_series(transforms):
    quat_t = matToQuat({'a':transforms})['a']
    try:
        avg_r = weightedAverageQuaternions(quat_t[:,(6,3,4,5)], np.ones(len(transforms)))
    except:
        avg_r = weightedAverageQuaternionsNoJit(quat_t[:,(6,3,4,5)], np.ones(len(transforms)))

    avg_t = np.average(quat_t[:,:3], axis=0)

    out_quat = np.concatenate([avg_t, avg_r])[None,:]
    out_quat = out_quat[:,(0,1,2,4,5,6,3)]


    return quatToMat({'a': out_quat})['a']


def applyTransformation(data, transform, invert=False):
    '''
    Apply a transformation to a pointcloud, that is not in homogenous coordinates
    '''

    if data.shape[1] == 3:
        data = make_homogenous(data)
        if invert:
            stabilized_pc = np.linalg.inv(transform) @ data
        else:
            stabilized_pc = transform @ data
        stabilized_pc = revert_homogeneous(stabilized_pc)

        return stabilized_pc
    if data.shape[1] == 4:
        out_data = np.zeros_like(data)

        for i in range(data.shape[0]):
            out_data[i] = applyTransformation(data[i], transform, invert)

        return out_data
    else:
        raise ValueError('unknown pointcloud format')
    

def make_homogenous(A):
    '''
    Convert points into homogeneous coordinates, copy them to maintain the originals
    '''
    dims = A.shape[1] # dimensions for NxDim array
    
    src = np.ones((dims+1, A.shape[0]))
    src[:dims,:] = np.copy(A.T)

    return src

def revert_homogeneous(src):
    '''
    Revert points from homogeneous coordinates
    '''

    dims = src.shape[0]-1

    A = np.zeros([src.shape[1],dims])
    A = src[:dims,:].T
    
    return A


def matToEuler(t_mat, degrees=False):
    """
    Convert a transformation matrix to Euler angles.

    Parameters:
    - t_mat: The transformation matrix to convert.
    - degrees: Whether to return the angles in degrees (default: False).

    Returns:
    - euler_angles: An array containing the converted Euler angles [x, y, z, roll, pitch, yaw].

    """

    transformations = np.array(t_mat)

    x = transformations[:,0,3]
    y = transformations[:,1,3]
    z = transformations[:,2,3]

    rotations = np.array(t_mat)[:,:3,:3]
    rpy = Rotation.from_matrix(rotations).as_euler(seq='xyz', degrees=degrees)

    return np.array([x, y, z, rpy[:,0], rpy[:,1], rpy[:,2]]).T

def quatToMat(t_mat):
    """
    Convert a quaternion transformation matrix to a homogeneous transformation matrix.

    Parameters:
    t_mat (list or numpy.ndarray): The quaternion transformation matrix.

    Returns:
    numpy.ndarray: The homogeneous transformation matrix.

    """
    transformations = np.array(t_mat)

    xyz = transformations[:,:3]
    rotations = transformations[:, 3:]

    if np.isnan(rotations).all():
        homo_trans = np.empty([transformations.shape[0],4,4])
        homo_trans[:] = np.nan
        return homo_trans
    elif np.isnan(rotations).any():
        nan_mats = np.isnan(rotations[:,0])
        rot_mat = np.zeros((rotations.shape[0],3,3))
        rot_mat[~nan_mats] = Rotation.from_quat(rotations[~nan_mats]).as_matrix()
        rot_mat[nan_mats] = np.nan
    else:
        rot_mat = Rotation.from_quat(rotations).as_matrix()

    homo_trans = np.zeros([transformations.shape[0],4,4])
    homo_trans[:,:3,:3] = rot_mat
    homo_trans[:,:3,3] = xyz
    homo_trans[:,3,3] = 1.0

    return homo_trans


def matToQuat(t_mat):
    """
    Convert a transformation matrix to a quaternion representation.

    Args:
        t_mat (numpy.ndarray): The transformation matrix.

    Returns:
        numpy.ndarray: The concatenated array of translation vector and quaternion.

    """
    transformations = np.array(t_mat)

    xyz = transformations[:,:3,3]
    rotations = transformations[:,:3,:3]

    quat = Rotation.from_matrix(rotations).as_quat()
    xyz_quat = np.concatenate([xyz,quat],1)

    return xyz_quat


@njit
def weightedAverageQuaternions(Q, w, pos_q=True):
    """
    Compute the weighted average of a set of quaternions.

    Args:
        Q (numpy.ndarray): Array of quaternions to average.
        w (numpy.ndarray): Array of weights corresponding to each quaternion.
        pos_q (bool, optional): Flag indicating whether to enforce positive quaternions. Defaults to True.

    Returns:
        numpy.ndarray: Weighted average quaternion.

    Raises:
        ValueError: If all weights are zero.

    """
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4,4))
    weightSum = np.sum(w)

    assert((w >= 0).all())

    if weightSum == 0:
        raise ValueError("All weights are zero")

    for i in range(0,M):
        q = Q[i,:]
        A = w[i] * np.outer(q,q) + A

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    out = np.real(eigenVectors[:,0].ravel())

    # normalize quaternions
    if pos_q:
        if out[0] < 0:
            out = out* -1
    return out

def weightedAverageQuaternionsNoJit(Q, w, pos_q=True):
    """
    This function is a non-jit version of weightedAverageQuaternions
    """

    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4,4))
    weightSum = np.sum(w)

    assert((w >= 0).all())

    if weightSum == 0:
        raise ValueError("All weights are zero")

    for i in range(0,M):
        q = Q[i,:]
        A = w[i] * np.outer(q,q) + A

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    out = np.real(eigenVectors[:,0].ravel())

    # normalize quaternions
    if pos_q:
        if out[0] < 0:
            out = out* -1
    return out




