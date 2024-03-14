import os
import argparse
import json
import time

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from head_motion_tools import sequential_registration, metadata_io, postprocessing




def make_motion_plot(input_file, output_file, mode='deviation'):
    fig, ax = plt.subplots(figsize=(10, 5))

    motion_data = pd.read_csv(input_file, index_col=0)
    motion_data.index = pd.to_timedelta(motion_data.index)
    if mode == 'motion':
        motion_data = (motion_data * 8).rolling('10s').mean()

    motion_data.plot(ax=ax, legend=None)
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    #fig.axes[0].xaxis.set_major_formatter(formatter)
    plt.xlabel('Time after motiontracker start')
    plt.ylabel('Head deviation from starting point (mm)' if mode == 'deviation' else 'Head speed (mm/s)')

    plt.title(motion_data.columns[0])
    
    plt.tight_layout()
    plt.savefig(output_file)


def make_motion_plot_split(motion_data, input_folder, output_file, acquisition_start_times, acquisition_lengths, mode='deviation'):
    offset = .5
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if mode == 'motion':
        motion_data = (motion_data * 8).rolling('10s').mean()
    ax = motion_data.plot(ax=ax, legend=None)
    acquisition_start_times = acquisition_start_times.set_index('nifti_filename')
    max_motion = np.nanmax(motion_data.values)


    for seq_name in acquisition_start_times.T.sort_values(by='acq_time_seconds',axis='columns'):
        recording_start = metadata_io.peekDate(input_folder,timestamp=True)
        seq_start = (acquisition_start_times.T[seq_name]['acq_time_seconds'] - postprocessing.datetime_to_s_since_midnight(recording_start)) * 1e9
        seq_name = seq_name[:-7] if seq_name.endswith('nii.gz') else seq_name  # remove '.nii.gz' ending
        
        if seq_name not in acquisition_lengths.keys():
            continue
            
        seq_time = (acquisition_lengths[seq_name] - 10) * 1e9 # time ns -> s
        
        plt.vlines(seq_start ,0,max_motion + 0.1, color='red')
        plt.text(seq_start,max_motion + 0.03,seq_name, fontsize=6)
        rect = matplotlib.patches.Rectangle((seq_start,0),seq_time,max_motion + 0.1,alpha=0.3)
        ax.add_patch(rect)
        offset += .05
        #print(seq_name, seq_start)

    plt.xlabel('Time after motiontracker start')
    if 'deviation' in output_file:
        plt.ylabel('Head deviation from starting point (mm)')
    elif 'motion' in output_file:
        plt.ylabel('Head speed (mm/s)')

    plt.title(motion_data.columns[0])

    plt.tight_layout()
    
    plt.savefig(output_file)


def main(input_folder, output_folder, deviation=False, t1w_image=None, acquisition_times_csv=None, sequence_length_json=None, tracsuite_registration=False, param_dict_registration=None, param_dict_postprocessing=None):

    if param_dict_registration is None:
            param_dict_registration = {
                'REGISTRATION_METHOD' : 'IRCP',
                'REFERENCE': 'REF',
                'MAX_ITER': 30,
                'CROP' : None,
                'OUTPUT_FOLDER' : output_folder,
                'OUTDIR': os.path.join(output_folder, 'matrices'),
                'HEAD_PINPOINT': False,
                'SAVE_WEIGHTS' : False,
                'CARRY_MASK': False,
                'PRE_ALIGN' : True,
                'FP_WEIGHT': 0.02,
                'EST_B': -0.5,
                'UNDERSAMPLING': 3,
                'N_THREADS': 4,
                'CRITERION': 1,
                'REF_TO_PC': False,
            }

    if param_dict_postprocessing is None:
        param_dict_postprocessing = {
            'TRANSFORM_FOLDER_IRCP': output_folder,
            'MODE': 'DEVIATION' if deviation else 'MOTION',
            'SMOOTH': True, 
            'SMOOTHING_DIST': 13,
            'INTERPOLATION_MODE': 'Transformation',
            'MAP_TO_RAS': t1w_image is not None,
            'CROP': param_dict_registration['CROP'],
        }

    subject_name = os.path.basename(input_folder)

    registration_output_file = os.path.join(output_folder, 'matrices', 'registration_matrices.npy')
    if not tracsuite_registration and not os.path.isfile(registration_output_file):
        pc_list, timestamp_list = metadata_io.get_point_cloud_paths(input_folder)

        with open(os.path.join(output_folder, 'pointcloud_paths.json'), 'w') as f:
            f.write(json.dumps(pc_list.tolist()))
        with open(os.path.join(output_folder, 'raw_timestamps.json'), 'w') as f:
            f.write(json.dumps(timestamp_list.tolist()))

        # main registration loop
        if not os.path.isfile(registration_output_file):
            sequential_registration.register_series(input_folder, pc_list, t1_path=t1w_image, param_dict=param_dict_registration, debug=True)
        else:
            print('registration matrices already exist, skipping registration')


    postfix = param_dict_postprocessing["MODE"].lower() + ("_tracsuite" if tracsuite_registration else "")
    motion_file = f'motion_data_{postfix}.csv'

    if not os.path.isfile(os.path.join(output_folder, motion_file)):
        # de-noising, dimensionality reduction
        motion_data, dates, euler_form_df = postprocessing.prepare_motion_data(input_folder, output_folder, param_dict_postprocessing, get_euler_form=True, load_tracsuite_transformations=tracsuite_registration, t1w_image=t1w_image)
        motion_data = pd.DataFrame(motion_data, columns=[subject_name])

        # write motion data to csv
        motion_data.to_csv(os.path.join(output_folder, motion_file))

        with open(os.path.join(output_folder, 'dates.json'), 'w') as f:
            f.write(json.dumps(dates))
        euler_form_df.to_csv(os.path.join(output_folder, 'euler_transform.csv'))
    else:
        motion_data = pd.read_csv(os.path.join(output_folder, motion_file), index_col=0)
        motion_data.index = pd.to_timedelta(motion_data.index)
        print('motion data already exists, skipping postprocessing')

    plot_name = f'motion_plot_{postfix}.png'
    plot_path = os.path.join(output_folder, plot_name)

    if not os.path.isfile(plot_path):
        make_motion_plot(os.path.join(output_folder, motion_file), os.path.join(output_folder, plot_name), mode=param_dict_postprocessing["MODE"].lower())
        print(f'saved motion plot to {plot_path}')
    else:
        print('motion plot already exists, skipping plotting')


    split_sequences_filename = os.path.join(output_folder, f'split_sequences_{postfix}.csv')
    seq_start_times_filename = os.path.join(output_folder, f'seq_start_times_{postfix}.csv')
    motion_averages_filename = os.path.join(output_folder, f'motion_averages_{postfix}.csv')

    if acquisition_times_csv is None: # skip splitting by acquisitions
        return
    else:
        acquisition_start_times = pd.read_csv(acquisition_times_csv, delimiter=';')

        if not subject_name in acquisition_start_times['subjectid'].values:
            print(f'subject {subject_name} not found in acquisition times csv, skipping splitting by acquisitions')
            return

        acquisition_start_times = acquisition_start_times.set_index('subjectid').loc[subject_name]
        with open(sequence_length_json, 'r') as f:
            acquisition_lengths = json.load(f)
        if not (os.path.isfile(split_sequences_filename) and os.path.isfile(seq_start_times_filename) and os.path.isfile(motion_averages_filename)):
            

            split_sequences, seq_start_times, motion_averages = postprocessing.split_sequences(motion_data, acquisition_start_times, acquisition_lengths, crop=10)
            
            split_sequences.to_csv(split_sequences_filename)
            print(f'wrote per sequence motion data to {split_sequences_filename}')
            seq_start_times.to_csv(seq_start_times_filename)
            print(f'wrote sequence start times to {seq_start_times_filename}')
            motion_averages.to_csv(motion_averages_filename)
            print(f'wrote motion averages to {motion_averages_filename}')
        else:
            print('loading split sequences from file')
            split_sequences = pd.read_csv(split_sequences_filename, index_col=0)
            seq_start_times = pd.read_csv(seq_start_times_filename, index_col=0)
            motion_averages = pd.read_csv(motion_averages_filename, index_col=0)
        


    motion_split_file = os.path.join(output_folder, f'motion_plot_split_{postfix}.png')
    if not os.path.isfile(motion_split_file):
        make_motion_plot_split(motion_data, input_folder, motion_split_file, acquisition_start_times, acquisition_lengths, mode=param_dict_postprocessing["MODE"].lower())
        print(f'saved split motion plot to {motion_split_file}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run registration on a dataset')

    parser.add_argument('-i','--input_folder', type=str, help='Folder with TracSuite outputs', required=True)
    parser.add_argument('-o', '--output_folder', type=str, help='Folder to save outputs to', required=True)
    parser.add_argument('--t1', type=str, help='T1w image to help registration', default=None)
    parser.add_argument('--acquisition_times_csv', type=str, help='Csv file with start times of MRI sequences', default=None)
    parser.add_argument('--sequence_length_json', type=str, help='Json file with lengths of MRI sequences', default=None)
    parser.add_argument('--tracsuite_registration', help='Use TracSuite registrations', action='store_true', default=False)
    parser.add_argument('--deviation', help='Use deviation from starting position instead of motion', action='store_true', default=False)

    args = parser.parse_args()

    if not os.path.isdir(args.output_folder):
        try:
            os.mkdir(args.output_folder)
        except:
            print(f'could not create output folder {args.output_folder}')
            exit(1)

    assert(os.path.isdir(args.input_folder)), f'input folder {args.input_folder} does not exist'
    assert(os.path.isdir(args.output_folder)), f'output folder {args.output_folder} does not exist, please create it first'
    if args.acquisition_times_csv is not None:
        assert(os.path.isfile(args.acquisition_times_csv)), f'acquisition times csv {args.acquisition_times_csv} does not exist'

    if args.sequence_length_json is None and args.acquisition_times_csv is not None:
        print('Provide sequence lengths csv to split by acquisitions, or remove acquisition times csv to skip splitting by acquisitions')
        exit(1)
    if args.sequence_length_json is not None and args.acquisition_times_csv is None:
        print('Provide acquisition times csv to split by acquisitions, or remove sequence lengths csv to skip splitting by acquisitions')
        exit(1)
        

    main(input_folder=args.input_folder, 
         output_folder=args.output_folder, 
         t1w_image=args.t1, 
         acquisition_times_csv=args.acquisition_times_csv, 
         sequence_length_json=args.sequence_length_json, 
         tracsuite_registration=args.tracsuite_registration,
         deviation=args.deviation)