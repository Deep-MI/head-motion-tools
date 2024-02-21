"""
Display the through registration with a reference stabilized point clouds for visual quality control.
"""

import os
import numpy as np
import json
import argparse

from head_motion_tools import transformation_tools, postprocessing, metadata_io, point_cloud_io, file_helpers
from head_motion_tools.visualization import VtkVisualizer, VtkTools



def main(input_folder, output_folder, crop=None, mapping='Depth', save_3d=False, save_mp4=False, timestep=24.995*4):
    """
    Main function to process and visualize point cloud sequences.

    Args:
        input_folder (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        crop (tuple): Tuple containing the start and end index of the point cloud sequence to process (default: None).
        mapping (str): Color mapping to use for the 3d visualization (default: 'Depth'). Options: 'Depth', 'Gray', 'Weights'.
        save_3d (bool): Whether to save the 3d visualization as a movie (default: False).
        save_mp4 (bool): Whether to save the matplotlib output as an mp4 (default: False).

    Returns:
        None
    """

    # if SHOW_REGISTRATION:
    if os.path.isfile(os.path.join(output_folder, 'pointcloud_paths.json')):
        with open(os.path.join(output_folder, 'pointcloud_paths.json'), 'r') as f:
            pc_filenames = json.load(f)
    else:
        pc_filenames = metadata_io.get_point_cloud_paths(input_folder)

    

    with open(os.path.join(output_folder, 'matrices', 'registration_matrices.npy'), 'rb') as f:
        own_transforms = np.load(f)

    if crop is not None:
        pc_filenames = pc_filenames[crop[0]:crop[1]]
        own_transforms = own_transforms[crop[0]:crop[1]]


    if mapping == 'Weights':
        with open(os.path.join(output_folder, 'matrices', 'registration_matrices_weights.npy'), 'rb') as f:
            weights = np.load(f)
        with open(os.path.join(output_folder, 'matrices', 'registration_matrices_output_data.npy'), 'rb') as f:
            gen_pcs = np.load(f)

        if crop is not None:
            weights = weights[crop[0]:crop[1]]
            gen_pcs = gen_pcs[crop[0]:crop[1]]
    

    ref_filename =  file_helpers.find_files_by_wildcard(input_folder, '*REF*.pcd')[-1]
    ref_pc = point_cloud_io.loadPcd(ref_filename)
    ref_pc, _ = point_cloud_io.to2dArray(ref_pc, getColors=True)

    pc_seq = VtkVisualizer.PointCloudSequence(125, no_pcs=1, colors_min=0, colors_max=60)

    for i, f in enumerate(pc_filenames):
        if (i+1) % 5 == 0:
            print('read',i+1,'of',len(pc_filenames),'files', end='\r')

        orig_pc = point_cloud_io.loadPcd(f)
        orig_pc, colors = point_cloud_io.to2dArray(orig_pc, getColors=True)

        if mapping == 'Gray':
            pc_seq.addPointCloud(orig_pc, colors, 'red')
        elif mapping == 'Weights':
            pc_seq.addOverlaidPointClouds([ref_pc, gen_pcs[i]], [weights[i], colors], ['weights','gray'])
        elif mapping == 'Depth':
            pc_seq.addPointCloud(orig_pc)#, colors_min=pt_min, colors_max=pt_max)
        else:
            raise EnvironmentError('unknown color mapping setting for 3d view')

    print('read',i+1,'files          ')

    pc_seq.show()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize the registration of a point cloud sequence.')
    parser.add_argument('-i', '--input_folder', type=str, help='Path to the input folder.', required=True)
    parser.add_argument('-o', '--output_folder', type=str, help='Path to write intermediate outputs folder. Also used to read registrations', required=True)
    parser.add_argument('--crop', type=int, nargs=2, help='Crop the point cloud sequence to the specified range (start, end).', default=None)
    parser.add_argument('--mapping', type=str, help='Color mapping to use for the 3d visualization (default: Depth). Options: Depth, Gray, Weights', default='Depth')
    parser.add_argument('--save_3d', help='Save the 3d visualization as a movie.', action='store_true', default=False)
    parser.add_argument('--save_mp4', help='Save the matplotlib output as an mp4.', action='store_true', default=False)
    parser.add_argument('--timestep', type=float, help='Time between frames in movie (default: 24.995*4).', default=24.995*4)
    args = parser.parse_args()

    


    main(args.input_folder, args.output_folder, crop=args.crop, mapping='Depth', save_3d=False, save_mp4=False, timestep=24.995*4)

