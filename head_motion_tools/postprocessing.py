from argparse import Namespace
import os
from glob import glob

import numpy as np
import pandas as pd
import nibabel as nib

from head_motion_tools import metadata_io, transformation_tools, transformation_smoothing, motion_magnitude, mri_tools, sequential_registration

def load_transformation_series(input_dir, output_dir, get_euler_form=True, load_tracsuite_registration=False, mapping=None):
    """
    Loads a series of transformations and writes it to the output directory

    Args:
        input_dir (str): The input directory path.
        output_dir (str): The output directory path.
        get_euler_form (bool, optional): Whether to get the Euler form. Defaults to True.
        load_tracsuite_registration (bool, optional): Whether to load Trac Suite registration. Defaults to False.
        mapping (np.ndarray, optional): A mapping matrix to convert the transformation series into another space. Preferably this space has the head center at the origin. Defaults to None.
    """
    if not load_tracsuite_registration:
        transformations = np.load(os.path.join(output_dir, 'matrices', 'registration_matrices.npy'))
    else:
        try:
            pc_filenames = metadata_io.load_pointcloud_paths(output_dir) # try to load from cache
        except:
            pc_filenames, _ = metadata_io.get_point_cloud_paths(input_dir)
        transformations = metadata_io.load_tracsuite_transformation_series(input_dir, pc_filenames)

    if mapping is not None:
        transformations = mapping @ transformations @ np.linalg.inv(mapping)


    if get_euler_form:
        xzyrpy = transformation_tools.matToEuler(transformations)

        euler_form_df = pd.DataFrame(xzyrpy)
        euler_form_df.columns = ['x','y','z','roll','pitch','yaw']
    else:
        xzyrpy = None
        euler_form_df = None

    tim_file_path = glob(os.path.join(input_dir, '*TIM.tst'))[0]
    date = os.path.split(tim_file_path)[-1].split('_')[0]

    return transformations, date, xzyrpy, euler_form_df



def prepare_motion_data(input_folder, output_folder, parameter_dict, get_euler_form=True, load_tracsuite_transformations=False, t1w_image=None):
    """
    Reads motion data for analysis or display.

    Args:
        input_folder (str): The input folder path, where point cloud data is stored.
        output_folder (str): The output folder path, where the transformation data is already stored.
        parameter_dict (dict): A dictionary containing the parameters.
        get_euler_form (bool, optional): Whether to get the Euler form. Defaults to True.
        load_tracsuite_transformations (bool, optional): Whether to load Trac Suite transformations. Defaults to False.
        

    Values in parameter dict:
        MODE                    MOTION / DEVIATION      MOTION: calculates the RMSD to previous transformation
                                                        DEVIATION: caulculates to reference
        INTERPOLATION_MODE      INTERPOLATION_MODE      TRANSFROMATION: interpolates on transformation level, including smoothing w/ SMOOTHING_DIST
                                                        None: no iterpolation   TODO: make this a boolean
        SMOOTH   SMOOTH   True: smooths the ground truth transformation
                                                        False: no smoothing
        SMOOTHING_DIST          SMOOTHING_DIST          window size for smoothing

                                


    Returns:
        tuple: A tuple containing the motion data, date dataframe, and transform dataframe.
    """
    p = Namespace(**parameter_dict)


    if p.INTERPOLATION_MODE != 'Transformation' and not p.INTERPOLATION_MODE is None:
        raise ValueError('wrong INTERPOLATION_MODE identifier')
    
    if p.MAP_TO_RAS:

        mt_ras_savepath = os.path.join(output_folder, 'matrices','mt_to_ras.npy')
        if os.path.isfile(mt_ras_savepath):
            pc_to_ras_mri_mapping = np.load(mt_ras_savepath)
        else:
            nibabel_image = nib.load(t1w_image)
            reference_pointcloud, _ = sequential_registration.load_reference(input_folder, ref_type='EYE', output_space='MT')
            # robust scaling and converts to LIA
            nibabel_image_conformed = mri_tools.conform(nibabel_image)
            # get transformation from point cloud to MRI RAS space
            pc_to_ras_mri_mapping = mri_tools.register_pc_to_mri(nibabel_image_conformed, reference_pointcloud, debug=False)
            np.save(mt_ras_savepath, pc_to_ras_mri_mapping)
    else:
        pc_to_ras_mri_mapping = None


    
    transformation_series, date, euler_form, euler_form_df = load_transformation_series(input_folder, output_folder, get_euler_form, load_tracsuite_transformations, mapping=pc_to_ras_mri_mapping)


    if p.INTERPOLATION_MODE == 'Transformation': # smoothing + interpolation
        print('doing transformation interpolation', 'with smooth parameters' if p.SMOOTH else 'with hard parameters (no smoothing)')
        timestamps = metadata_io.getTimestampsForSequence(input_folder, output_folder, cut=0, zeroIn=True)

        if p.CROP is not None:
            timestamps = timestamps[p.CROP[0]:p.CROP[1]]

        interpolation_timestamps = np.arange(0, timestamps[-1], 0.125)

        transformation_series = transformation_smoothing.weightedSmoothing(transform_series=transformation_series,
                                                   timestamps=timestamps,
                                                   weight_factor=0.1 if p.SMOOTH else 0.4,
                                                   window_size=p.SMOOTHING_DIST if p.SMOOTH else 3,
                                                   interpolation_timestamps=interpolation_timestamps,
                                                   output='matrix',
                                                   average_mode='linear',
                                                   window_in_s=False)
        


    elif p.SMOOTH: # smoothing only
        print('doing sliding window transformation smoothing')
        smoothed_data = transformation_smoothing.simpleSmoothing(transformation_series, window_size=p.SMOOTHING_DIST, verbose=False, output='matrix')
        diff = len(smoothed_data) - len(transformation_series)
        assert(np.all(np.array(diff) == np.array(diff)[0], axis = 0))
        transformation_series = smoothed_data

    # save subj_dict_ircp time_dict_full


    if p.MODE == 'MOTION':
        scalar_motion = motion_magnitude.quantifyMotion(transformation_series,seq='FULL')
    elif p.MODE == 'DEVIATION':
        scalar_motion = motion_magnitude.quantifyDeviation(transformation_series,seq='FULL', zeroIn=True)
    else:
        raise ValueError('wrong MODE identifier')
        
    sec_to_timedelta = lambda x : pd.to_timedelta(x, unit='sec')


    ######### INTERPOLATE (if not done previously)

    if p.INTERPOLATION_MODE == 'Transformation': # interpolation was already performed in the previous step
        print('create dataframe')
        datetime_idx = sec_to_timedelta(interpolation_timestamps)
        if p.MODE == 'MOTION':
            datetime_idx = datetime_idx[:-1] # cut first value because of reference registration
        elif p.MODE == 'DEVIATION':
            pass
        else:
            print('Wrong MODE identifier')
            raise NotImplementedError('Wrong MODE identifier')
        
        motion_data = pd.Series(scalar_motion)
        motion_data.index = datetime_idx
    else:
        motion_data = pd.Series(scalar_motion)
            
    return motion_data, date, euler_form_df



def datetime_to_s_since_midnight(in_time):
    midnight = in_time.replace(hour=0, minute=0, second=0, microsecond=0)
    return (in_time - midnight).seconds


"""
splits motion data daraframe into different the different sequences

motion data     data frame with subjects as index and times since first point-cloud as columns
acq_times       data frame with times of acquisition for each subjects/sequence combination
TRIM            amount of seconds of signal to discard at the beginning and end of each sequence
"""
def split_sequences(motion_data, acq_times, sequence_lengths, start_zero=True, crop=10, AVERAGE=True, use_sync=False, pilot_mode=False):

    sequences = sequence_lengths.keys()

    if pilot_mode:
        sequences = [] # in pilot mode sequences might be different for every subject, so we cant predefine them

    if start_zero:
        split_sequences = pd.DataFrame(index=sequences,
                                       columns=motion_data.index[:int(max(sequence_lengths.values())/0.125)], data=None)
    else:
        split_sequences = pd.DataFrame(index=pd.MultiIndex.from_product(
                                                [sequences, ['timestamps', 'data']],
                                                names=['sequence', 'subjectid', 'data']),
                                        columns=np.arange(stop=int(max(sequence_lengths.values())/0.1)),
                                        data=None)

    if not pilot_mode:
        # setup output dataframe
        motion_averages = pd.DataFrame(index=motion_data.columns)
        for seq in sequences:
            motion_averages[seq] = None

        seq_start_times = pd.DataFrame(index=motion_data.columns)
        for seq in sequences:
            seq_start_times[seq] = None
        
        acq_times['nifti_filename'] = acq_times['nifti_filename'].str.replace('.nii.gz', '')
        #acq_times['nifti_filename'] = acq_times['nifti_filename'].values.astype(str)
        acq_times = acq_times.set_index('nifti_filename')
    else:
        motion_averages = {}
        seq_start_times = {}
        acq_times = acq_times.set_index('nifti_filename')


    # these are the resampled timestamps for the motion, since the first point cloud was acquired
    pc_times = motion_data.index.total_seconds() #.to_numpy().astype('float')      

    for seq_name in sequences:
        seq_key = seq_name

        if seq_name not in acq_times.index:
            if seq_name == 'T2' and 'T2_caipi' in acq_times.T.columns: # handle edge case of two different T2 protocols
                seq_key = 'T2_caipi'
            else:
                continue
        if start_zero:
            seq_start = 0
        else:
            seq_start = (acq_times.T[seq_key]['acq_time_seconds']) #*1e9

        
        start_idx = transformation_smoothing.find_nearest_sorted(pc_times,seq_start + crop) #* 1e9)            # add 10 sec to start time
        end_idx =   transformation_smoothing.find_nearest_sorted(pc_times,seq_start + (sequence_lengths[seq_name] - crop))# * 1e9)
        end_idx   = start_idx + ((sequence_lengths[seq_name] - crop*2) / 0.125)  # remove 10 sec from end time to remove early breaks and smoothing effects
        tmp = motion_data[int(start_idx):int(end_idx)]
        if start_zero:
            seq_idx = np.where(split_sequences.index.values.astype(str) == seq_name)[0]
            assert(len(seq_idx) == 1)
            split_sequences.iloc[seq_idx.squeeze(), :len(tmp)] = tmp.values.squeeze()
        else:
            split_sequences.loc[(seq_name,'data'),:len(tmp)-1] = tmp.values
            split_sequences.loc[(seq_name,'timestamps'),:len(tmp)-1] = motion_data.index[int(start_idx):int(end_idx)]

        if AVERAGE:
            motion_averages[seq_name] = tmp.mean()

        final_start_time = (acq_times.T[seq_key]['acq_time_seconds'] - (0 if pilot_mode else acq_times.T['B0_Phase']['acq_time_seconds']))
        seq_start_times[seq_name] = final_start_time

    split_sequences = split_sequences.dropna(axis=1, how='all')

    if AVERAGE:
        return split_sequences, seq_start_times, motion_averages
    else:
        return split_sequences, seq_start_times

