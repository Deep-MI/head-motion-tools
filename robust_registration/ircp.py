"""
This is a python implementation of the ICP algorithm described in [1, 2]

This is an adaption by Clemens Pollak, 2023 for face registration in during MRI scans [3].


[1] Bergström, P. and Edlund, O. 2014, 
“Robust registration of point sets using iteratively reweighted least squares”, 
Computational Optimization and Applications, vol 58, no. 3, pp. 543-561, doi: 10.1007/s10589-014-9643-2

[2] Bergström, P. and Edlund, O. (2016) 2017, 
“Robust registration of surfaces using a refined iterative closest point algorithm with a trust region approach”, 
Numerical Algorithms, doi: 10.1007/s11075-016-0170-3

[3] Pollak C, Kügler D, Breteler MM, Reuter M. 2023 May
Quantifying MR Head Motion in the Rhineland Study - A Robust Method for Population Cohorts. 
NeuroImage.  18:120176. doi: https://doi.org/10.1016/j.neuroimage.2023.120176


The original matlab code can be found at https://de.mathworks.com/matlabcentral/fileexchange/12627-iterative-closest-point-method
"""

import numpy as np
from numba import njit
from scipy.spatial import KDTree


def pre_icp_crop(reference, data, tree, k=1, max_dist=5, n_threads=6, get_idxs=False):
    """
    Prepares pointclouds for registration with preliminary cropping
    Selects points in data that have neighbours closer than max_dist to reference

    reference   pointcloud data should be registered to - the resulting area is expected to be bigger for reference, than for data
    data        checking around this pointcloud to find nearest neighbours withing max_dist
    tree   KDtree intialized with reference
    k           number of neighbours to check and keep in reference, expecially important when the resolution for the reference is higher than data
    max_dist    maximum distance before points get discarded
    """

    if reference.shape[1] != 3:
        raise ValueError('Reference Pointcloud is not 3 dimensional')
    if data.shape[1] != 3:
        raise ValueError('Input Pointcloud is not 3 dimensional')

    data = data.T
    reference = reference.T
    distances, indices = tree.query(data.T, k=k, eps=0, p=2, distance_upper_bound=max_dist, workers=n_threads) # distance_upper_bound doesnt seem to do anything..
    
    potential_matching_pts = distances < max_dist
    # select all points in data that are closer than max_dist to a point in reference
    data = data[:, potential_matching_pts[:,0]].T

    # select all points in reference that is closest point for a point in data and has < max_dist distance to it 
    reference = reference[:, np.unique(indices[potential_matching_pts])].copy() 

    if get_idxs:
        return reference.T, data, potential_matching_pts
    else:
        return reference.T, data


def fast_ircp(reference, data, fix_point=None, fix_point_data=None, fp_weight=0.2 ,maxIter=20, critFun=0, 
             getWeights=False, getError=False, est_b=1.9, n_threads=6, fixed_correspondence=False):
    
    """
    Fast Iterative Closest Point algorithm for point clouds registration

    reference   pointcloud data should be registered to - the resulting area is expected to be bigger for reference, than for data
    data        pointcloud to be registered
    fix_point   point in reference that should be fixed during registration
    fix_point_data  point in data that should be fixed during registration
    fp_weight   weight for the fixed point
    maxIter     maximum number of iterations
    critFun     criterion function to be used
                1:  Huber
                2:  Tukey
                3:  Cauchy
                4:  Welsch
    getWeights  return weights
    getError    return error
    est_b       est_b (estimation bound) parameter for robust criterion
    n_threads   number of threads to use for KDTree
    fixed_correspondence    if True, the correspondence between reference and data is fixed and the input arrays have to have the same shape, otherwise the correspondence is found using a KDTree

    Returns:
    matrix     homgenous transformation matrix mapping data to reference
    data       transformed data
    """

    # Size of reference points and data points
    if reference.shape[1] < reference.shape[0]:
        raise ValueError('Dimension mismatch reference' + str(reference.shape))
    if data.shape[1] < data.shape[0]:
        raise ValueError('Dimension mismatch data' + str(data.shape))
    if reference.shape[0] != data.shape[0]:
        raise ValueError('The dimension of the reference points and data points must be equal')

    data = data.astype(np.float64)
    reference = reference.astype(np.float64)

    if fix_point is not None:
        assert(fix_point_data is not None)
        has_fixed_point = True
        reference = np.append(reference, fix_point[:,None], axis=1) # TODO: this copies the reference - we want to avoid that
        data = np.append(data, fix_point_data[:,None], axis=1)
    else:
        has_fixed_point = False

    # Create closest point search structure
    distances = np.zeros(data.shape[1], dtype=np.float64)
    vi        = np.ones( data.shape[1], dtype=int)

    if not fixed_correspondence:
        neigh = KDTree(reference.T, leafsize=20, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
    else:
        assert(reference.shape == data.shape), 'Error: for fixed correspondence the input arrays have to have the same shape.'
        vi = np.arange(0,data.shape[1],1,dtype=int)

    # Initialize weights (Only for robust criterion)
    weights = np.ones((int(data.shape[1]),1), dtype=np.float64)

    T_rot = np.eye(int(reference.shape[0]), dtype=np.float64)
    T_trans = np.zeros((int(reference.shape[0]),1), dtype=np.float64)

    res = np.inf
    for iter in range(maxIter):
        if fixed_correspondence:
            distances = np.sqrt(np.sum((reference-data)**2, axis=0))
        else:
            distances, indices = neigh.query(data.T, k=1, eps=0, p=2, workers=n_threads)
            distances = distances.ravel()
            vi = indices.ravel()

        C, weights, data_w, ref_w = evalCritFun(critFun, distances, vi, weights, reference, data, est_b, has_fixed_point=has_fixed_point, fp_weight=fp_weight)
        # numba fails when this is merged together..
        T_trans, T_rot, data = getNewTrans(C, data_w, ref_w, data, T_rot, T_trans)

    # make homogeneous transformation
    ret = np.identity(reference.shape[0]+1, dtype=np.float64)
    ret[:reference.shape[0], :reference.shape[0]] = T_rot
    ret[:reference.shape[0], reference.shape[0]] = T_trans.squeeze()

    if getWeights and getError:
        return ret, data, weights, ret
    elif getWeights:
        return ret, data, weights
    elif getError:
        return ret, data, res
    else:
        return ret, data

@njit
def getRobustBound(distances, est_b):
    """
    creates bound for robust down-weighting function

    distances   output from KDtree with distances of corresponding points
    est_b       0<est_b<1 estimation bound for robust points
    """
    robust_bound = (1.0+est_b) * np.median(distances)
    max_distance = np.max(distances)
    if robust_bound < 1e-06 * max_distance:
        robust_bound = 0.3 * max_distance
    elif max_distance == 0:
        robust_bound = 1

    return robust_bound


@njit #(parallel=True)  # parallel is slower
def evalCritFun(critFun, distances, vi, weights, reference, data, est_b, has_fixed_point, fp_weight):

    # Estimation of bound which est_b (e.g. 80%) of data is less than
    robust_bound = getRobustBound(distances=distances ,est_b=est_b)

    if critFun == 1:
        # Huber
        robust_bound = 2.0138 * robust_bound
        for i in range(0, data.shape[1]):
            if distances[i] < robust_bound:
                weights[i] = 1
            else:
                weights[i] = robust_bound / distances[i]
    elif critFun == 2:
        # Tukey's bi-weight
        robust_bound = 7.0589 * robust_bound
        for i in range(0, data.shape[1]):
            if distances[i] < robust_bound:
                weights[i] = (1 - (distances[i] / robust_bound) ** 2) ** 2
            else:
                weights[i] = 0
    elif critFun == 3:
        # Cauchy
        robust_bound = 4.304 * robust_bound
        weights = 1.0 / (1 + (distances / robust_bound) ** 2)
        weights = np.expand_dims(weights,axis=1)
    elif critFun == 4:
        # Welsch
        robust_bound = 4.7536 * robust_bound

        weights = np.exp(- (distances / robust_bound) ** 2)
        weights = np.expand_dims(weights,axis=1)
    else:
        raise NotImplementedError('wrong citFun parameter')

    if has_fixed_point:
        weights[-1] +=  np.sum(weights)*fp_weight

    sum_of_weights = np.sum(weights)
    normalized_weights =  weights / sum_of_weights
    
    data_w = data @ normalized_weights
    ref_w  = reference[:, vi] @ normalized_weights

    C = (data * weights[:,0]) @ reference[:, vi].T - (sum_of_weights * data_w) @ ref_w.T
    return C, weights, data_w, ref_w#, average_selected_dist


@njit # (parallel=True)
def getNewTrans(C, data_w, ref_w, data, T_rot, T_trans):
    U, _, V = np.linalg.svd(C)
    Ri = V.T @ U.T
    if np.linalg.det(Ri) < 0:
        V[:, -1] = -V[:, -1]
        Ri = V @ U.T

    Ti = ref_w - (Ri @ data_w)
    data = Ri @ data

    for i in range(data.shape[0]): # iterate through dimensions
        data[i, :] = data[i, :] + Ti[i]

    T_rot = Ri @ T_rot
    T_trans = Ri @ T_trans + Ti

    return T_trans, T_rot, data



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    bunny_ref = np.loadtxt('bunny_source.txt')

    bunny_noise = bunny_ref + np.random.normal(0,0.0005,bunny_ref.shape) + 0.05
    bunny_noise[50:60] += np.random.normal(0,0.01,bunny_ref[50:60].shape)

    print(bunny_ref.shape)

    transform, out_bunny, weights = fast_ircp(bunny_ref.T, bunny_noise.T, est_b=1.2, max_iterations=20, error_target=None, critFun='huber', getWeights=True, getError=False,  n_threads=6, fixed_correspondence=True)
    out_bunny = out_bunny.T


    print('weights range from', np.min(weights), 'to', np.max(weights))
    pt_size = 20

    plt.figure()
    plt.scatter(bunny_ref[:,0], bunny_ref[:,1], s=pt_size, label='fixed pc', c='blue')
    plt.scatter(bunny_noise[:,0], bunny_noise[:,1], s=pt_size, label='movable', c='green')
    #plt.scatter(bunny_target[:,0], bunny_target[:,1], s=pt_size, label='movable')
    output_scat = plt.scatter(out_bunny[:,0], out_bunny[:,1], s=pt_size, label='aligned bunny', c=weights, cmap='copper')
    plt.legend()
    cbar = plt.colorbar(output_scat)
    cbar.set_label('aligned bunnys weights')
    plt.show()