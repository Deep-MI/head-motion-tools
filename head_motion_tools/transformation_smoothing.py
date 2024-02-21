import math

import numpy as np
from numba import njit

from head_motion_tools import transformation_tools


def weightedSmoothing(transform_series, timestamps, weight_factor=0.1, window_size=9, 
                      interpolation_timestamps=None, output='matrix', average_mode='linear', window_in_s=False):
    """
    Wrapper function for weighted smoothing of transformation timeseries with interpolation capability

    transform_series       numpy array with shape (n,4,4) for homogeneous transformation matrices
    timestamps             numpy array with shape (n,) for timestamps
    weight_factor          weight decay per time unit (linear mapping function/relu), higher decay -> less smooth output
    window size            number of values included in smoothing window
    interpolation_timestamps if this is None then subj_dict_timestamps are taken to be center ponts, otherwise center points are taken from this array
                        must be sorted!
    output                 if this is "matrix" transformation matrices are returned instead of quaternions
    average_mode           ['linear'/'constant'] linear: linear decay of weights over time,     constant: weights either 1 or 0, no decay
    window_in_s            if true the window size is calculated according to the time between samples, otherwise number of samples
    """


    if interpolation_timestamps is None:
        INTERPOLATE = False
    else:
        INTERPOLATE = True
        assert(np.all(np.diff(interpolation_timestamps) >= 0))
        assert(interpolation_timestamps.ndim == 1)


    transform_series = transformation_tools.matToQuat(transform_series)

    

    
    timestamps = np.array(timestamps)

    if timestamps.shape[0] != transform_series.shape[0]:
        raise ValueError('timestamps and transform_series must have the same length')

    # filter out nans from failed registrations
    if np.isnan(transform_series).any():
        nan_mask = np.ones(len(transform_series), dtype=bool)
        nan_idxs = np.unique(np.argwhere(np.isnan(transform_series))[:,0])
        nan_mask[nan_idxs] = False
        transform_series = transform_series[nan_mask]
        timestamps = timestamps[nan_mask]

    if INTERPOLATE:
        smoothed_translation = np.zeros((interpolation_timestamps.shape[0],3))
        smoothed_quaternions = np.zeros((interpolation_timestamps.shape[0],4))
        #if not window_size is None:
        half_window_size = window_size//2
        previous_index = 0
        for i, t in enumerate(interpolation_timestamps):

            if window_in_s:
                idx1 = find_nearest_sorted(timestamps, t- window_size/2)
                #closest_index = find_nearest_sorted(timestamps[idx1:], t- window_size/2) + idx1 # increase speed by searching masked array
                idx2 = find_nearest_sorted(timestamps[idx1:], t+ window_size/2) + idx1
                closest_index = idx1

                if idx2 - idx1 < 2:
                    smoothed_quaternions[i] = np.nan
                    smoothed_translation[i] = np.nan
                    continue

            else:
                # TODO: implement index finding based on zero points of weight function (window_size == None)
                closest_index = find_nearest_sorted(timestamps[previous_index:], t)
                closest_index =  closest_index + previous_index

                if closest_index < half_window_size or closest_index > len(timestamps) - half_window_size - 1:
                    smoothed_quaternions[i] = np.nan
                    smoothed_translation[i] = np.nan
                    continue
                idx1, idx2 = closest_index-half_window_size, closest_index+half_window_size

            


            weights = calc_w_mapping(weight_factor, timestamps[idx1:idx2], p=t, mode=average_mode)


            # zero weights can be caused by a large gao in timestamps e.g. when the motiontracker was paused / restarted
            if np.sum(weights) == 0:
                smoothed_quaternions[i] = np.nan
                smoothed_translation[i] = np.nan
                continue

            try:
                smoothed_quaternions[i] = transformation_tools.weightedAverageQuaternions(transform_series[idx1:idx2,(6,3,4,5)], weights)
            except (SystemError, ValueError) as e:
                #print('WARNING: python fallback', e)
                smoothed_quaternions[i] = transformation_tools.weightedAverageQuaternionsNoJit(transform_series[idx1:idx2,(6,3,4,5)], weights)
            smoothed_translation[i] = np.average(transform_series[idx1:idx2,:3], axis=0, weights=weights)
            previous_index = closest_index

        #smoothed_quaternions = np.array([weightedAverageQuaternions(xi[:,(6,3,4,5)], calc_w_mapping(weight_factor, find_nearest_sorted(t), t)) for xi, t in zip(transform_series[index],inperolate_timestamps)])
    else: # sliding window smoothing
        # this is equal to mean sliding window smoothing when weight_factor=0 
        index = createSldidingWindowIndices(transform_series, window_size=window_size)
        w_mappings = [calc_w_mapping(weight_factor, t, mode=average_mode) for t in timestamps[index]]
        try:
            smoothed_quaternions = np.array([transformation_tools.weightedAverageQuaternions(xi[:,(6,3,4,5)], wm) for xi, wm in zip(transform_series[index],w_mappings)])
        except (SystemError, ValueError):
            #print('WARNING: python fallback')
            smoothed_quaternions = np.array([transformation_tools.weightedAverageQuaternionsNoJit(xi[:,(6,3,4,5)], wm) for xi, wm in zip(transform_series[index],w_mappings)])
        smoothed_translation = np.array([np.average(xi[:,:3],axis=0, weights=wm) for xi, wm in zip(transform_series[index],w_mappings)])

    smoothed_series = np.concatenate([smoothed_translation, smoothed_quaternions], axis=1)
    
    if output == 'matrix': # put q to the back for scipy
        smoothed_series = smoothed_series[:,(0,1,2,4,5,6,3)]

    #subj_dict_ircp[subject_name] = subj_dict_ircp[subject_name][SMOOTHING_DIST//2:-(SMOOTHING_DIST//2)]

    if output == 'matrix':
        smoothed_series = transformation_tools.quatToMat(smoothed_series)

    return smoothed_series



"""
find the index of the value closest to the input value in an array
"""
def find_nearest_sorted(array,value): # only for sorted arrays!
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


"""
generates a mapping based on the distance of timepoints to distance point p and a weight factor

weight_factor       linear decay factor, if input is in seconds then a weight factor of 0.1 means that weighting decays 10% per second
in_arr              input array with timestamps
p                   timepoint associated with output - if None this is the middle of in_arr otherwise a desired timepoint for interpolation
mode                linear: linear weight decay, constant: step function; no weight decay until 1-weight_factor*distance < 0, then weights =0
"""
@njit
def calc_w_mapping(weight_factor, in_arr, p=None, mode='linear'):
    out_array = np.zeros_like(in_arr, dtype=np.float64)
    if p is None:
        window_size = in_arr.shape[0]
        p = in_arr[window_size//2]

    in_arr = np.abs(p - in_arr)

    if mode=='linear':
        #wmp = np.vectorize(lambda distance: 0 if 1-weight_factor*distance < 0 else 1-weight_factor*distance)
        for i in range(in_arr.shape[0]):
            w = 1-in_arr[i]*weight_factor
            out_array[i] = 0 if w < 0 else w

    elif mode =='constant':
        #wmp = np.vectorize(lambda distance: 0 if 1-weight_factor*distance < 0 else 1)
        for i in range(in_arr.shape[0]):
            w = 1-in_arr[i]*weight_factor
            out_array[i] = 0 if w < 0 else 1
    else:
        raise ValueError('unknown mode identifier')

    #return wmp(np.abs(p - in_arr))
    return out_array


def createSldidingWindowIndices(arr, window_size, step_size=1):
    """
    Create sliding window indices for a given array.

    Parameters:
    arr (numpy.ndarray): The input array.
    window_size (int): The size of the sliding window.
    step_size (int, optional): The step size between consecutive windows. Default is 1.

    Returns:
    numpy.ndarray: An array of sliding window indices.
    """
    return np.arange(window_size)[None, :] + step_size * np.arange(arr.shape[0]-window_size+step_size)[:, None]


def simpleSmoothing(transformation_series, window_size=9, output='matrix'):
    """
    Wrapper for average quaternions to do a sliding window smoothing

    subj_dict_ircp      dictionary with homogenous matrices
    window_size         sliding window size
    output              whether a matric or quaternions should be returned
    """

    subj_dict_ircp = transformation_tools.matToQuat(transformation_series)

    smoothed_data = {}

    

    index = createSldidingWindowIndices(transformation_series, window_size=window_size)

    smoothed_quaternions = np.array([transformation_tools.averageQuaternions(xi[:,(6,3,4,5)]) for xi in transformation_series[index]])
    smoothed_translation = np.array([np.mean(xi[:,:3],axis=0) for xi in transformation_series[index]])
    smoothed_series = np.concatenate([smoothed_translation, smoothed_quaternions], axis=1)

    
    if output == 'matrix': # put q to the back for scipy
        smoothed_series = smoothed_series[:,(0,1,2,4,5,6,3)]

    #subj_dict_ircp[subject_name] = subj_dict_ircp[subject_name][SMOOTHING_DIST//2:-(SMOOTHING_DIST//2)]

    if output == 'matrix':
        smoothed_data = transformation_tools.quatToMat(smoothed_data)

    return smoothed_data
