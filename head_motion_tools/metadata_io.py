import json
import os
from glob import glob
import time
import re
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm


from head_motion_tools import file_helpers, metadata_io



def get_reference_path(pointcloud_dir: str, ref_type: str):
    """
    Returns the path to the reference pointcloud or image

    subject         image ID
    ref_type        PCL, REF, EYE or T1 for different types of references that can be loaded
    """

    subject_id = os.path.basename(pointcloud_dir)

    if ref_type == 'EYE' or ref_type == 'PCL' or ref_type == 'REF':
        #LOG_DIR = getLogDir(subject)

        files = file_helpers.find_files_by_wildcard(pointcloud_dir, '*.pcd')

        # multiple references found -> determine appropriate reference, by matching MOT.tsm file
        if len(files) > 2:
            MOTION_FILES = file_helpers.find_files_by_wildcard(pointcloud_dir, '*_MOT.tsm')
            assert(len(MOTION_FILES) > 1), 'many pointclouds found, but one or less MOT.tsm files found -- directory might be corrupted'
            time_dict = get_timestamps_for_sequence(cut=0,seq='FULL',zeroIn=False)
            seq_start = time_dict[subject_id][0]
            RUN_ID = None

            seconds_in_subsession = []
            # check if a MOT.tsm file describes all known pointclouds
            for motion_file in MOTION_FILES:
                acq_times_raw, _ = read_opt_time(motion_file)
                acq_times = convert_acquisition_time_to_seconds(acq_times_raw)
                if seq_start <= acq_times[-1] and seq_start >= acq_times[0]:
                    RUN_ID = re.match(r'.+_(\d+)_MOT.tsm',motion_file).group(1) # detmine ID of subsession from MOT.tsm file
                    break
                seconds_in_subsession.append(acq_times[-1] - acq_times[0])
            # if no MOT.tsm file describes all known pointclouds, use the MOT.tsm file that covers the most time
            if RUN_ID is None:
                motion_file = MOTION_FILES[np.argmax(seconds_in_subsession)]
                RUN_ID = re.match(r'.+_(\d+)_MOT.tsm',motion_file).group(1) # detmine ID of subsession from MOT.tsm file
                print('Couldnt find any motiontracker run that contains the full sequence, using the longest one instead')

            new_files = []
            for pcd_file in files:
                if RUN_ID in pcd_file:
                    new_files.append(pcd_file)
            
            files = new_files
            assert(len(files) == 2)

        # select either raw or 
        for filepath in files:
            if ('PCL' if ref_type == 'PCL' else 'REF') in filepath:
                ref_pc_path = filepath

        try:
            assert(os.path.isfile(ref_pc_path))
        except:
            raise FileNotFoundError(subject_id,'ERROR: could not find reference')
    else:
        raise ValueError('wrong ref_type')

    return ref_pc_path


def get_timestamps_for_sequence(output_dir, zeroIn=True, cut=None, only_corrected=False):
    """
    returns the timestamps for each registration/pointcloud in a dict of subjects, as list
    seq     sequence of interest
    zeroIn  define if the timestamps should start at 0 
             (if false the original timestamps will be returned, which should be according to scanner start time)
    cut     if not None cop away x number of values at the front and end of the list of timstamps. 
             If cut is a tuple (x,y) crop x in front and y in the back.
    

    return  dictionary with subject IDs as keys, containing lists of timestamps
    """

    
    with open(os.path.join(output_dir, 'timestamps.json'), 'rb') as f:
        timestamps = json.load(f)

    time_dict = {}
        
    for subject_id in timestamps.keys():

        if zeroIn:
            try:
                time_dict[subject_id] = (np.array(timestamps[subject_id]) - timestamps[subject_id][0]).tolist()
            except Exception as e:
                print('skipping timeseries for', subject_id, e)
                continue
        else:
            time_dict[subject_id] = timestamps[subject_id]
        
        if cut:
            if isinstance(cut, tuple):
                if cut[0] == 0 and cut[1] == 0:
                    pass
                elif cut[0] == 0:
                    time_dict[subject_id] = time_dict[subject_id][:-cut[1]]
                elif cut[1] == 0:
                    time_dict[subject_id] = time_dict[subject_id][cut[0]:]
                else:
                    time_dict[subject_id] = time_dict[subject_id][cut[0]:-cut[1]]
            else:
                time_dict[subject_id] = time_dict[subject_id][cut:-cut]


    return time_dict

def convert_acquisition_time_to_seconds(acq_times):
    '''
    convert acquisition times to seconds after midnight

    acq_times: a single string or a list of strings representing time since midnight in the format HHMMSS.ffff…. (f=fraction of a second)
    return: corresponding numpy float32 array of times in seconds after midnight
    '''
    fs = [np.fromstring('0.' + el.split('.')[1], dtype=np.float64,sep=' ') for el in acq_times]
    tm = [time.strptime(el.split('.')[0], '%H%M%S') for el in acq_times]
    t = [np.float32(el1.tm_hour)*3600.0 +
        np.float32(el1.tm_min)*60.0 +
        np.float32(el1.tm_sec) +
        el2 for el1, el2 in zip(tm, fs)]

    return np.squeeze(t)

def read_opt_time(trac_file):
    '''
    reads REMOTE time, called system time and corresponding pcd file number from MOT.tsm logfile

    return - times in seconds as list, point cloud file IDs as pandas series
    '''
    with open(trac_file, 'r', encoding = "ISO-8859-1") as f:
        pos = 0
        while "Point Cloud Number" not in f.readline():
            pos = f.tell()
        f.seek(pos)    
        trac_info = pd.read_csv(f, sep=r"\s\s+", dtype='str', engine='python')
        acq_times = [str(t).replace(":", "") for t in trac_info['System Time']]
        point_clouds = trac_info['Point Cloud Number']

    return acq_times, point_clouds.to_numpy()


def get_point_cloud_paths(pointcoud_data_dir):
    """
    Retrieves the paths of point cloud files and their corresponding timestamps.

    Args:
        pointcoud_data_dir (str): The directory containing the point cloud data.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array contains the filenames of the point cloud files,
               and the second array contains the timestamps of the point cloud files.
    """

    subject = os.path.basename(pointcoud_data_dir)
    pointcloud_list_dir = os.path.join(pointcoud_data_dir, 'PointClouds')

    if not pointcloud_list_dir.endswith('/'):
        pointcloud_list_dir = pointcloud_list_dir + '/'

    MOTION_FILE = file_helpers.find_files_by_wildcard(pointcoud_data_dir, '*_MOT.tsm')

    if len(MOTION_FILE) == 0:
        raise FileNotFoundError('no MOT.tsm file found in', pointcoud_data_dir)
    elif len(MOTION_FILE) == 1:
        MOTION_FILE = MOTION_FILE[0]
    else:
        try:
            csv_data = get_mri_start_times()
        except:
            raise FileNotFoundError(f'couldnt load csv with sequence info to choose between {MOTION_FILE}')

        if subject in csv_data.keys():
            find_motionfile_output = find_sequence_in_motion_files(MOTION_FILE, csv_data[subject])

            if find_motionfile_output is None:
                raise FileNotFoundError('couldnt identify correct logfile')
                # TODO: we could have another fallback here
            else:
                MOTION_FILE = find_motionfile_output
        else:
            raise FileNotFoundError('couldnt resolve motion file - no sequence data found')


    FILE_PREFIX = MOTION_FILE[:-len('MOT.tsm')].split('/')[-1]
    acq_times_raw, point_cloud_ids_logs = metadata_io.read_opt_time(MOTION_FILE)

    assert(len(acq_times_raw) == len(point_cloud_ids_logs))
    acq_times = metadata_io.convert_acquisition_time_to_seconds(acq_times_raw)


    def id_to_filename(ID):
        return FILE_PREFIX + 'PCL_' + str(ID).rjust(6, '0') + '.pcd'
    filename_to_id = lambda x: int(x.split('_')[-1][:-4])

    



    existing_pc_files = os.listdir(pointcloud_list_dir)
    existing_pc_files = [x for x in existing_pc_files if x.endswith('.pcd') and not x.startswith('.')]
    existing_pc_files.sort(key=filename_to_id)
    point_cloud_files_logs = [id_to_filename(file_id) for file_id in point_cloud_ids_logs]

    timestamps = np.full(len(existing_pc_files), -1, dtype=np.float64)
    filenames  = np.zeros(len(existing_pc_files), dtype=f'<U200')
    current_idx = 0
    for i,exfname in enumerate(existing_pc_files):
        try:
            current_idx = point_cloud_files_logs.index(exfname, current_idx)
        except ValueError:
            continue

        timestamps[i] = acq_times[current_idx]
        filenames[i] =  os.path.join(pointcloud_list_dir, point_cloud_files_logs[current_idx])

        assert(len(pointcloud_list_dir + point_cloud_files_logs[current_idx]) == len(filenames[i])), 'memory insufficient for path length'


    not_found_mask = timestamps == -1
    not_found = np.sum(not_found_mask)
    if not_found > 0:
        print(not_found, 'pcs not in logfile')
        timestamps = timestamps[~not_found_mask]
        filenames = filenames[~not_found_mask]

    print('found', len(filenames), 'of ', len(point_cloud_files_logs), 'saved pointclouds ---', str(round(len(filenames)/len(point_cloud_files_logs)*100, 2))+'%' )


    return filenames, timestamps


def load_pointcloud_paths(working_dir):
    """
    Load pointcloud paths from a text file

    working_dir     directory where the file is located
    """
    with open(os.path.join(working_dir, 'pointcloud_paths.json'), 'r') as f:
        data = json.load(f)
    return data

def load_pointcloud_timestamps(working_dir):
    """
    Load pointcloud timestamps from a text file

    working_dir     directory where the file is located
    """
    with open(os.path.join(working_dir, 'timestamps.json'), 'r') as f:
        data = json.load(f)
    return data


def get_mri_start_times(acq_times_csv_path, input_data_dir, timeColumn='acq_time_seconds', subjectColumn='subjectid', seqDescrColumn='nifti_filename', verbose=True):
    """
    Reads Csv file and provides a dict with subject Ids and corresponding MRI sequences and their starting Points
    """
    if isinstance(acq_times_csv_path, list):
        subject_dict = {}
        for file in acq_times_csv_path:
            subject_dict_temp = metadata_io.read_acq_time(file,timeColumn=timeColumn, subjectColumn=subjectColumn, seqDescrColumn=seqDescrColumn)
            subject_dict = dict(subject_dict, **subject_dict_temp)
    else:
        subject_dict = metadata_io.read_acq_time(acq_times_csv_path,timeColumn=timeColumn, subjectColumn=subjectColumn, seqDescrColumn=seqDescrColumn)

    deleted_ids = 0
    full_subject_id_list = list(subject_dict.keys())

    for subject_id in full_subject_id_list:

            if not os.path.isdir(os.path.join(input_data_dir, subject_id)) or not os.path.isdir(os.path.join(input_data_dir, subject_id)):
                del subject_dict[subject_id]
                deleted_ids = deleted_ids + 1

    if verbose:
        if deleted_ids != 0:
            print('discarded', str(deleted_ids) + '/' + str(len(full_subject_id_list)), 'subject infos (files not found)')
        else:
            print('all subjects from csv found')

    return subject_dict


def read_acq_time(csv_file, timeColumn='acq_time_seconds', subjectColumn='subjectid',seqDescrColumn = 'nifti_filename'):

    # NOTE: in the provided files was a comma instead of a semincolon in one spot
    timestamp_csv = pd.read_csv(csv_file,delimiter=';')
    timestamp_csv = timestamp_csv.sort_values(timeColumn)

    subjects = timestamp_csv.groupby(subjectColumn)

    # dict with mapping of subject IDs to sequences
    subject_scans_dict = {}

    for subject_id,scan_data in subjects:

        # drop data we dont need
        scan_data = scan_data[[seqDescrColumn,timeColumn]]

        scan_data = scan_data.groupby(timeColumn)

        scan_data = scan_data.apply(lambda x: x.values.tolist())

        
        # create dict with mapping of time points to descriptions
        sequence_list = []
        for row in scan_data:
            sequence_name = ''
            for tupl in row:
                sequence_name = sequence_name + ',' + tupl[0]
                tupl[0]
            # remove leading comma
            sequence_list.append([row[0][1], sequence_name[1:]])

        subject_scans_dict[subject_id] = sequence_list

    return subject_scans_dict


def find_sequence_in_motion_files(motion_files, sequence_list, sequence_to_find = 'T1', sequence_length = 394.24):
    """
     select relevant mot.tsm (timestamp) from a list of files
     this happens when the scanner is restarted at some point
     the file is returned that specifies the run, which recorded the prefered sequence

     NOTE: use get reference if you want to acquire the correct refence matching known pointclouds

     motion_files       mot.tsm files
     sequence_list      list of tuples with sequence description and start time
     sequence_to_find   sequence of interest
     sequence_length    length of the sequence of interest
     return             either the mot.tsm file from MOTION file that belongs to 
                        the sequence_to_find (T1) or None when no file fits the requirements
    """
    for seq_start_time, description in sequence_list:
        if sequence_to_find in description:
            for file in motion_files:
                acq_times_raw, _ = metadata_io.read_opt_time(file)
                acq_times = metadata_io.convert_acquisition_time_to_seconds(acq_times_raw)

                # motion scan was before or after the MRI scan we are searching for
                if seq_start_time > acq_times[-1] or seq_start_time < acq_times[0]:
                    continue
                # motion scan ends in the middle of the preffered sequence               
                elif seq_start_time + sequence_length > acq_times[-1]:
                    print(file,'- motion scan stopped in the middle of MRI scan')
                # the preferred sequence is recorded by the file in question
                else:
                    assert(seq_start_time < acq_times[-1] and seq_start_time > acq_times[0])
                    print('located',sequence_to_find,'sequence in', file.split('/')[-1])
                    return file
            print('Motion scanner was not running during the',sequence_to_find,'sequence')
            return None
    print('No',sequence_to_find,'scanned')
    return None


def load_tracsuite_transformation_series(input_directory, pc_filenames, getConfidence=False):
    """
    Load the transformation matrices from the TracSuite registration

    input_directory     directory where the TracSuite files are located
    pc_filenames        list of pointcloud filenames
    getConfidence       if True, also returns the confidence values
    return              list of transformation matrices (4x4) - if getConfidence is True, also returns the confidence values
    """


    # list of pcd file numbers in our dataset
    existing_pcd_numbers = []
    for filename in pc_filenames:
        existing_pcd_numbers.append(int(re.findall('\d{6}',filename)[-1]))


    transform_file = file_helpers.find_files_by_wildcard(input_directory,'*POA.tsp')

    if len(transform_file) > 1:

        # identify POA file by matching with TSP info from pickle
        for fname in transform_file:
            TS_transforms, TS_index_vector, TS_confidence, _ = read_trac_suite_poa_file(fname,False)

            if existing_pcd_numbers[1] in TS_index_vector:
                break

    elif len(transform_file) == 0:
        raise FileNotFoundError(f'no POA file found in {input_directory}')
    else:
        assert(len(transform_file) == 1)
        transform_file = transform_file[0]
        TS_transforms, TS_index_vector, TS_confidence, _ = read_trac_suite_poa_file(transform_file,False)

    # creating numpy array, that masks index and transformation data to match our own
    trac_mask = np.zeros(TS_index_vector.shape,dtype=bool)
    for i,pcd_number in enumerate(existing_pcd_numbers):
        trac_idx = np.argwhere(pcd_number == TS_index_vector)
        
        assert(len(trac_idx) == 1)
        assert(len(trac_idx[0]) == 1)
        assert(pcd_number == TS_index_vector[trac_idx[0,0]])

        trac_mask[trac_idx[0,0]] = True


    assert((TS_index_vector[trac_mask] == np.array(existing_pcd_numbers)).all())

    # return transformations in the same format as our own registered data
    transformations = np.linalg.inv(np.moveaxis(TS_transforms[:,:,trac_mask],2,0))
    if getConfidence:
        confidence = TS_confidence[trac_mask]

    if getConfidence:
        return transformations, confidence
    else:
        return transformations



def read_trac_suite_poa_file(fileName, refOnly):
    """
    Reads in the TracSuite poa file, which contains the transformation matrices of the registration
    ## Reader for version 1 files, which includes quality parameter

    fileName    path to the file
    refOnly     if True, only the reference frame is read
    """

    fid = open(fileName,'r')

    ## Find the reference frame number
    tline = fid.readline()
    while not tline.startswith('Reference'):
        tline = fid.readline()

    C = tline.split()
    TS_index_0=float(C[-1])

    if refOnly:
        A_vector = []
        TS_index_vector = []
    else:
        ## Read the header info
        while not tline.startswith('Frame Number'):
            tline = fid.readline()
        
        # Done reading header
        
        ## Read the scan number, A matrix (transformation)
        rows_opt = fid.readlines()
        Nt = len(rows_opt)
        TS_index_vector = np.zeros(Nt,dtype=int)
        A_vector = np.zeros((4, 4, Nt))
        confidence_vec = np.empty((Nt))
        
        for key, row in enumerate(rows_opt):
            parts = row.split()
            TS_index_vector[key] = int(parts[0])
            A_vector[0,0,key] = parts[1]
            A_vector[0,1,key] = parts[2]
            A_vector[0,2,key] = parts[3]
            A_vector[0,3,key] = parts[4]
            A_vector[1,0,key] = parts[5]
            A_vector[1,1,key] = parts[6]
            A_vector[1,2,key] = parts[7]
            A_vector[1,3,key] = parts[8]
            A_vector[2,0,key] = parts[9]
            A_vector[2,1,key] = parts[10]
            A_vector[2,2,key] = parts[11]
            A_vector[2,3,key] = parts[12]
            A_vector[3,0,key] = parts[13]
            A_vector[3,1,key] = parts[14]
            A_vector[3,2,key] = parts[15]
            A_vector[3,3,key] = parts[16]
            confidence_vec[key] = parts[17]

    return A_vector, TS_index_vector, confidence_vec, TS_index_0



def getTimestampsForSequence(input_directory, output_directory, zeroIn=True, cut=None):
    """
    returns the timestamps for each registration/pointcloud in a list

    input_directory     directory where the TracSuite files are located
    output_directory    directory where the output files should be saved
    seq                 sequence of interest - FULL for the whole session
    zeroIn              define if the timestamps should start at 0 
                        (if false the original timestamps will be returned, which should be according to scanner start time)
    cut                 if not None cop away x number of values at the front and end of the list of timstamps. 
                        If cut is a tuple (x,y) crop x in front and y in the back.

    return  dictionary with subject IDs as keys, containing lists of timestamps
    """

    json_file = os.path.join(output_directory, 'raw_timestamps.json')

    try:
        with open(json_file, 'rb') as f:
            timestamps = json.load(f)
    except (FileNotFoundError, EOFError, ValueError) as e:
        print('couldnt find json, regenerating')
        try:
            timestamps = metadata_io.load_pointcloud_timestamps(input_directory)
        except:
            pc_paths, timestamps = metadata_io.get_point_cloud_paths(input_directory)
            with open(json_file, 'w') as f:
                json.dump(timestamps.tolist(), f)
            with open(os.path.join(output_directory, 'pointcloud_paths.json'), 'w') as f:
                json.dump(pc_paths.tolist(), f)
                

        try:
            with open(json_file,'r') as f:
                timestamps = json.load(f)
        except (FileNotFoundError, EOFError) as e:
            raise IOError('failed to read or generate', json_file, e)

        
        if zeroIn:
            timestamps = (np.array(timestamps) - timestamps[0]).tolist()
        
        if cut:
            if isinstance(cut, tuple):
                if cut[0] == 0 and cut[1] == 0:
                    pass
                elif cut[0] == 0:
                    timestamps = timestamps[:-cut[1]]
                elif cut[1] == 0:
                    timestamps = timestamps[cut[0]:]
                else:
                    timestamps = timestamps[cut[0]:-cut[1]]
            else:
                timestamps = timestamps[cut:-cut]


    return timestamps



def peekDate(input_dir, timestamp=False):
    """
    gets Date of subject recording
    subject     subjectID
    timestamp   whether an addtional timestamp in seconds after midnight should be returned
    """
    assert(os.path.isdir(input_dir))

    tim_file_path = glob(os.path.join(input_dir, '*TIM.tst'))[0]

    recording_date = tim_file_path.split('/')[-1].split('_')[0]

    if timestamp:
        with open(tim_file_path) as fp:
            for i, line in enumerate(fp):
                if i == 10:
                    assert(line.startswith('Point Cloud Number'))
                if i == 11:
                    time = line[-13:].strip()
                    return datetime.strptime(recording_date +' '+time, '%Y-%m-%d %H:%M:%S.%f')
    else:
        return datetime.strptime(recording_date, '%Y-%m-%d')

def peekEndDate(input_dir, timestamp=False):
    """
    same as peekDate, but for the end of the recording
    """
    assert(os.path.isdir(input_dir))
    tim_file_path = glob(os.path.join(input_dir, '*TIM.tst'))[0]
    recording_date = tim_file_path.split('/')[-1].split('_')[0]


    if timestamp:
        with open(tim_file_path, 'rb') as f:
            try:  # catch OSError in case of a one line file 
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
            time = last_line[-15:].strip()
        return datetime.strptime(recording_date +' '+time, '%Y-%m-%d %H:%M:%S.%f')
    else:
        return datetime.strptime(recording_date, '%Y-%m-%d')


