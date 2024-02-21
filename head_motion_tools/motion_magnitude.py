import numpy as np


from head_motion_tools import transformation_tools

def quantifyMotion(transformation_series, head_center=np.zeros((1,3)), seq='T1'):
    """
    Calculates the speed of movement from transformations.

    Parameters:
    - transformation_series: numpy array of homogenous transformations
    - head_center: numpy array representing the center of the head (default: 3x3 array of zeros)
    - seq: MRI sequence to analyze (default: 'T1')
    - correct_timestamps: boolean indicating whether to correct timestamps (default: True)

    Returns:
    - Speed of movement as calculated by the quantifier function.
    """
    return quantifier(transformation_series, mode='speed', head_center=head_center, from_starting_position=False, seq=seq)


def quantifyDeviation(transformation_series, head_center=np.zeros((1,3)), zeroIn=True, seq='T1', mode='RMSD'):
    """
    wrapper for "quantifier"
    calculates the distance to starting point
    
    Parameters:
    transformation_series: numpy array of homogenous transformations
    head_center: numpy array representing the center of the head (default: 3x3 array of zeros)
    zeroIn: boolean indicating whether to calculate the distance from the starting point (default: True)
    seq: MRI sequence to analyze (default: 'T1')
    mode: string indicating the mode to use (default: 'RMSD')
    correct_timestamps: boolean indicating whether to correct timestamps (default: True)
    """
    if not (mode == 'RMSD' or mode == 'centroid'):
        raise ValueError('Wrong mode identifier')
    return quantifier(transformation_series, mode=mode, head_center=head_center, from_starting_position=zeroIn, seq=seq)




def quantifier(transforms, mode, head_center=np.zeros((1,3)), from_starting_position=True, seq='FULL'):
    """
    calculates different quantifiers for the input transformations
    see wrapper functions quantifyDeviation, quantifyMotion

    transform_dict      dictionary of with lists of transformations as numpy arrays
    mode                (RMSD|centroid|speed) different measures
    sr                  sphere radius for root mean square distance
    from_starting_position when True the distances will be calculated with respect to the first transformation in the set
                            otherwise just the size of the given transforms - only applies to mode RMSD
    smoothing_dist      how many values to include in the smoothing
    seq                 MRI sequence to analyze
    mode                (RMSD|centroid|speed) different measures

    return  output          containing the quantifier as specified in 'mode'
            avgs            average smoothed output
            mad             median absolute deviation of smoothed RMSD_diff_dict
    """
    if not (mode == 'RMSD' or mode == 'centroid' or mode == 'speed'):
        raise ValueError('Wrong mode identifier')

    if from_starting_position is not False:
        if type(from_starting_position).__module__ == np.__name__:
            startingTrans = from_starting_position
            from_starting_position = True

    if head_center is None:
        print('head center unknown')
        RMS_r = 82.5
        RMS_x = np.array([[ -7.89449484, -48.07730192, 239.47502511]])
    else:
        RMS_r = 82.5
        RMS_x = head_center
        
    
    if mode == 'RMSD':
        if from_starting_position:
            startingTrans = transforms[0]
            if np.isnan(startingTrans).any():
                trans_arr = np.array(transforms)
                trans_arr = trans_arr[~np.isnan(np.array(transforms)[:,0,0])]
                startingTrans = trans_arr[0]


        RMSDs = []
        for i in range(0, len(transforms)):
            if np.isnan(transforms[i]).any(): # enforce numpy nan handling for numba
                RMSDs.append(np.nan)
            else:
                if from_starting_position:
                    RMSDs.append(transformation_tools.rmsDev(startingTrans, transforms[i], r=RMS_r, x=RMS_x))
                else:
                    RMSDs.append(transformation_tools.rmsDev(transforms[i], r=RMS_r, x=RMS_x))

        arr = np.array(RMSDs)

    elif mode == 'speed':
        RMSD_diffs = []
        prev = transforms[0]
        for i in range(1, len(transforms)):
            if np.isnan(transforms[i]).any() or np.isnan(prev).any(): # enforce numpy nan handling for numba
                RMSD_diffs.append(np.nan)
            else:
                RMSD_diffs.append(transformation_tools.rmsDev(prev, transforms[i], r=RMS_r, x=RMS_x))
            prev = transforms[i]

        arr = np.array(RMSD_diffs)

    return arr.tolist()
