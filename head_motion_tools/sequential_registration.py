"""
This script generates transforms of a reference pointcloud 
to a set of other pointclouds specified in a pickle file.
"""
VERSION = '2.0.3'
VISUALIZE = False # debugging mode determines imports (this is useful e.g. in docker, where we dont debug and therefore dont need vtk)

import os

# python native imports
import pickle
import time
from types import SimpleNamespace

# pip package imports
import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree

# custom library imports
from head_motion_tools import ircp, transformation_tools, metadata_io, preprocessing, mri_tools
from head_motion_tools import point_cloud_io as pc_io


if VISUALIZE:    
    from head_motion_tools.visualization import VtkVisualizer, VtkTools



def load_reference(input_dir, ref_type, output_space, structured_pc_coordinates=False):
    """
    Loads a reference 

    input_dir       path to the directory containing the reference
    type            PCL, REF, EYE or T1 for different types of references that can be loaded
    process         whether references should be postprocessed
    output_space    space to output reference in   RAS | MT | None/native

    return ref_pc, colors       reference pointcloud and attached intensity values
    """

    if (output_space == 'RAS' and ref_type != 'T1') or (output_space == 'MT' and ref_type == 'T1'):
        raise NotImplementedError('Scanner space transformations not implemented')
        # with open(os.path.join(T1_TO_MT_DIR,'mat',subject + '.npy'), 'rb') as f:
        #     ras_to_mt_transform = np.load(f)

    if ref_type == 'T1' and structured_pc_coordinates:
        raise NotImplementedError('cant provide structure for T1 image yet')

    ref_pc_path = metadata_io.get_reference_path(input_dir, ref_type)


    if ref_type == 'EYE' or ref_type == 'PCL' or ref_type == 'REF':
        # get reference pointcloud
        if ref_type == 'PCL' or ref_type == 'REF':
            if structured_pc_coordinates:
                load_pc, colors, s_pc_coords = pc_io.to2dArray(pc_io.loadPcd(ref_pc_path), getColors=True, structured_pc_coordinates=True)
            else:
                load_pc, colors = pc_io.to2dArray(pc_io.loadPcd(ref_pc_path), getColors=True)
        elif ref_type == 'EYE':
            ref_pc = pc_io.loadPcd(ref_pc_path)
            pcl_pc = pc_io.loadPcd(ref_pc_path.replace('REF','PCL'))
            ref_pc = preprocessing.copyEye(ref_pc, pcl_pc, inplace=True)
            load_pc, colors = pc_io.to2dArray(ref_pc, getColors=True)
            if structured_pc_coordinates:
                raise NotImplementedError('cant return structured coordinates after eye copying yet')
        else:
            raise ValueError('wrong ref_type')

        # map to output space
        if output_space == 'RAS':
            raise NotImplementedError('Scanner space transformations not implemented')
            ref_pc = transformation_tools.applyTransformation(load_pc, ras_to_mt_transform, invert=True)
        elif output_space == 'MT' or output_space == 'None' or output_space is None or output_space == 'native':
            ref_pc = load_pc
        else:
            raise NotImplementedError('wrong output space identifier')

        if structured_pc_coordinates:
            return ref_pc, colors, s_pc_coords
        else:
            return ref_pc, colors

    # load T1 reference
    elif ref_type == 'T1':
        raise NotImplementedError('T1 reference loading not implemented')
        nifti_data = nib.load(ref_pc_path)
        nifti_data = conform.conform(nifti_data)#.get_fdata()
        nifti_data_raw = nifti_data.get_fdata()
        mri_pc, mri_colors = extractFaceFromConformedRAS(nifti_data_raw, coronal_axis=2, offset=5, reverse_direction=False, skin_threshold=30)

        # map to output space
        if output_space == 'RAS':
            ref_pc = TransformationTools.applyTransformation(mri_pc, nifti_data.affine, invert=False)
        elif output_space == 'MT':
            ref_pc = TransformationTools.applyTransformation(mri_pc, ras_to_mt_transform @ nifti_data.affine, invert=False)
        elif output_space == 'None' or output_space is None or output_space == 'native':
            ref_pc = mri_pc
        else:
            raise NotImplementedError('wrong output space identifier')

        return ref_pc, mri_colors
        #return extractFaceFromConformed(nifti_data)
    else:
        raise NotImplementedError('wrong reference file')


def register_series(input_directory, pc_list, param_dict, t1_path=None, debug=False):
    """
    This function registers the pointcloud sequence corresponding to the subject_name
    The output is saved as a pickle file in the specified folder (param_dict).

    input_directory  path to the directory containing the pointclouds
    pc_list          list of pointclouds
    param_dict      dictionary with registration parameters
    t1_path          path to the T1 image (optional, can improve registration)
    debug           whether to display debug information

    return          'done' if successful


    options for parameter dict:
    'REFERENCE': 'PCL', 'REF', 'EYE', 'T1'   # type of reference
    'MAX_ITER': 30  # maximum number of iterations for ICP
    'CROP' : None  # cropping of the sequence
    'OUTPUT_FOLDER' : 'motiontracker_support'  # output folder
    'OUTDIR': os.path.join(output_folder, 'matrices')  # output directory
    'SAVE_WEIGHTS' : True  # whether to save the weights of the registration
    'CARRY_MASK': False  # whether to carry the mask from the previous registration
    'PRE_ALIGN' : True  # whether to pre-align the pointclouds with the previous transformation
    'FP_WEIGHT': 0.02  # weight for the fixpoint in ICP
    'EST_B': -0.5  # robust bound for ICP
    'UNDERSAMPLING': 3  # factor for undersampling the pointclouds
    'N_THREADS': 4  # number of threads for parallel processing
    'CRITERION': 1  # 1 for point to point, 2 for point to plane
    'REF_TO_PC': False  # whether the reference should be registered to the pointclouds
    """
    global VISUALIZE
    start_t = time.time()

    p = SimpleNamespace(**param_dict)
    p.OUTFILE = 'registration_matrices.npy'

    ref_pc, _ = load_reference(input_directory, ref_type=p.REFERENCE, output_space='MT')

    if p.CROP is not None:

        if isinstance(p.CROP, dict):
            p.CROP = p.CROP[input_directory]

        pc_list = pc_list[p.CROP[0]:p.CROP[1]]

        

        pc = pc_io.to2dArray(pc_io.loadPcd(pc_list[0]))
        neigh = KDTree(ref_pc, leafsize=20, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
        crop_ref, pc, _ = ircp.pre_icp_crop(reference=ref_pc, data=pc, tree=neigh, k=3, max_dist=5, n_threads=p.N_THREADS, get_idxs=True)
        prev_trans, out_pc, weights = ircp.fast_ircp(reference=crop_ref.T, data=pc.T, critFun=p.CRITERION, maxIter=p.MAX_ITER, getWeights=True,
                                                    est_b=p.EST_B, n_threads=p.N_THREADS, fix_point=None, fp_weight=p.FP_WEIGHT)

        if VISUALIZE:
            print('before alignment because of sequence cropping')
            VtkVisualizer.displayOverlaidPointCloud(
                    VtkTools.VtkPointCloud(pc, colors='red'), 
                    VtkTools.VtkPointCloud(ref_pc), title='before alignment because of sequence cropping')
            print('alignment because of sequence cropping')
            VtkVisualizer.displayOverlaidPointCloud(
                    VtkTools.VtkPointCloud(out_pc.T, colors=weights / np.max(weights), colors_min=0, colors_max=1, color_map='default'), 
                    VtkTools.VtkPointCloud(ref_pc), title='alignment because of sequence cropping')
                                        
    else:
        prev_trans = np.eye(4)

    if p.FP_WEIGHT > 0 and t1_path is not None:
        USING_FIXPOINT = True

        fixpoint_savepath = os.path.join(p.OUTDIR, 'matrices', 'fixpoint_reference.txt')
        mt_ras_savepath = os.path.join(p.OUTDIR, 'matrices', 'mt_to_ras.npy')
        if os.path.isfile(fixpoint_savepath) and os.path.isfile(mt_ras_savepath):
            fixpoint_reference = np.loadtxt(fixpoint_savepath)
        else:
            fixpoint_reference, pc_to_ras_mri_mapping = mri_tools.get_stabilizing_point(t1_path, load_reference(input_directory, ref_type='EYE', output_space='MT')[0], debug=debug)
            np.savetxt(fixpoint_savepath, fixpoint_reference)
            np.save(mt_ras_savepath, pc_to_ras_mri_mapping)
        fix_point_data = fixpoint_reference.copy()
    else:
        USING_FIXPOINT = False
        fixpoint_reference = None
        fix_point_data = None


    transform_list = []
    if p.SAVE_WEIGHTS:
        weight_list = []
        output_data_list = []

    print('generating', p.OUTFILE, 'from', len(pc_list), 'pointclouds')

    if debug: 
        # compile numba accelerated fucntions first
        test_pc = pc_io.loadPcd(pc_list[0])
        test_pc = test_pc[::p.UNDERSAMPLING,::p.UNDERSAMPLING]
        test_pc = pc_io.to2dArray(test_pc)
        ircp.fast_ircp(reference=test_pc.T, data=test_pc.T, 
            maxIter=1, critFun=2, getWeights=False, est_b=0.5)
        # display registration speed with progress bar
        iterator = tqdm(pc_list)
    else:
        iterator = pc_list # no progress bar

    
    
    fallback_mat = np.empty((4,4))
    fallback_mat[:] = np.nan
    empty_mat = np.empty(shape=(0, 0))


    if p.CARRY_MASK:
        eye_cloud = preprocessing.get_eye_point_cloud(input_directory)
    
    if p.REF_TO_PC:
        ref_pc = ref_pc[::p.UNDERSAMPLING]

    neigh = KDTree(ref_pc, leafsize=20, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)

    for pc_path in iterator:  # registration loop

        # load pointcloud
        pc = pc_io.loadPcd(pc_path)
        pc, pc_structure = pc_io.to2dArray(pc,getColors=False, mask=None, structured_pc_coordinates=True)


        if not p.REF_TO_PC:
            pc = pc[::p.UNDERSAMPLING]
            pc_structure = pc_structure[::p.UNDERSAMPLING]

        if p.PRE_ALIGN:
            pc = transformation_tools.applyTransformation(pc, prev_trans, invert=False)

            if p.CARRY_MASK:
                pc = preprocessing.applyMaskFromRegistration(eye_cloud, pc)#, colors=orig_colors if VISUALIZE else None)


        if debug and VISUALIZE:
            print('displaying: to register - preprocessed, before pre_crop')
            VtkVisualizer.displayPointCloud(VtkTools.VtkPointCloud(ref_pc, colors='red'), title='to register - preprocessed, before pre_crop')#,colors=mri_colors))
            VtkVisualizer.displayPointCloud(VtkTools.VtkPointCloud(pc, colors='green'), title='to register - preprocessed, before pre_crop')#,colors=mri_colors))

            VtkVisualizer.displayOverlaidPointCloud(
                VtkTools.VtkPointCloud(pc, colors='green'), 
                VtkTools.VtkPointCloud(ref_pc, colors='red'), title='to register - preprocessed, before pre_crop')#,colors=mri_colors))

        # search for KNN for all points in ref
        crop_ref, pc, pre_crop_idxs = ircp.pre_icp_crop(reference=ref_pc, data=pc, tree=neigh, k=3, max_dist=5, n_threads=p.N_THREADS, get_idxs=True)
        pre_crop_idxs = pre_crop_idxs[:,0]


        if VISUALIZE:
            print('displaying: to register - preprocessed, after pre_crop')
            VtkVisualizer.displayOverlaidPointCloud(
                VtkTools.VtkPointCloud(pc, colors='green'), 
                VtkTools.VtkPointCloud(crop_ref,colors='red'), title='to register - preprocessed, after pre_crop')

        if crop_ref.shape[0] < 30 or pc.shape[0] < 30: # error handling, if too few points are found
            transform_list.append(fallback_mat)
            if p.SAVE_WEIGHTS:
                weight_list.append(empty_mat)
                output_data_list.append(empty_mat)
            continue

        if p.REF_TO_PC:
            registration_data = crop_ref.T
            registration_reference = pc.T
        else:
            registration_data = pc.T
            registration_reference = crop_ref.T
        transform, out_pc, weights = ircp.fast_ircp(reference=registration_reference, data=registration_data, critFun=p.CRITERION, maxIter=p.MAX_ITER, getWeights=True,
                                                est_b=p.EST_B, n_threads=p.N_THREADS, fix_point=fixpoint_reference, fix_point_data=fix_point_data, fp_weight=p.FP_WEIGHT)
        out_pc = out_pc.T

        if p.REF_TO_PC:
            transform = np.linalg.inv(transform)

        if np.isnan(transform).any(): # error handling
            print('registration did not converge')
            transform_list.append(fallback_mat)
            if p.SAVE_WEIGHTS:
                weight_list.append(None)
                output_data_list.append(None)
            continue

        if p.SAVE_WEIGHTS:
            weight_list.append(weights)
            output_data_list.append(out_pc)
        else:
            out_pc = out_pc.T

        if p.PRE_ALIGN:
            transform = transform @ prev_trans
        transform_list.append(transform)

        # save previous tranformation for quicker and more accurate icp
        prev_trans = transform
        if debug and VISUALIZE:
            print('displaying: registration reference and registered pc with weights')
            VtkVisualizer.displayOverlaidPointCloud(
                VtkTools.VtkPointCloud(registration_reference.T),#, colors=mri_colors),
                VtkTools.VtkPointCloud(out_pc, colors=weights / np.max(weights) if not USING_FIXPOINT else weights[:-1] / np.max(weights[:-1]), colors_min=0, colors_max=1, color_map='default'),
                title='registration reference and registered pc with weights')

        if USING_FIXPOINT:
            fix_point_data = out_pc[:,-1]

        

    #### end loop #########

    end_t = time.time()

    if not os.path.isdir(p.OUTDIR):
        os.makedirs(p.OUTDIR)

    with open(os.path.join(p.OUTDIR, p.OUTFILE), 'wb') as f:
        #pickle.dump(transform_list, f)
        np.save(f, transform_list)
        print('saved', p.OUTFILE, ' --- runtime ',end_t - start_t,'seconds')

    if p.SAVE_WEIGHTS:
        os.makedirs(os.path.join(p.OUTDIR, 'weights'), exist_ok=True)

        for i, w in tqdm(enumerate(weight_list), total=len(weight_list)):
            with open(os.path.join(p.OUTDIR, 'weights', f'weights_{i:05}.npy'), 'wb') as f:
                np.save(f, w)
                
        os.makedirs(os.path.join(p.OUTDIR, 'output_data'), exist_ok=True)
                
        for i, p in tqdm(enumerate(output_data_list), total=len(output_data_list)):
            with open(os.path.join(p.OUTDIR, 'output_data', f'output_data_{i:05}.npy'), 'wb') as f:
                np.save(f, p)
        
        print('saved weights')


    return 'done'

