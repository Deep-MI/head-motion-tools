import numpy as np
import os

import nibabel as nib

from head_motion_tools import metadata_io
from head_motion_tools import point_cloud_io as pc_io


def get_eye_mask(pc_ref, getIdx=False):
    """
    Get manual segmentation of the eye from reference pointcloud
    pc_ref          (N,6) array (pointcloud)
    getIdx          if True, returns the indices of the mask
    return          (N,6) boolean array (mask)
    """
    mask = np.zeros((pc_ref.shape[0],pc_ref.shape[1]), dtype=np.bool)
    mask[np.logical_and(np.logical_and(pc_ref[:,:,3] > 100, pc_ref[:,:,4] == 0), pc_ref[:,:,5] == 0)] = True
    if getIdx:
        return mask, np.nonzero(mask)
    else:
        return mask


def get_eye_point_cloud(subject_name):
    """
    Crops out eye from reference
    This can be used to later mask the eye in non-reference pointclouds.

    subject_name       subject ID
    return             (N,3) float array (pointcloud) 
    """

    ref_pcl, _, ref_pcl_2d_structure = pc_io.loadReference(subject_name, ref_type='PCL', output_space='MT', structured_pc_coordinates=True)
    ref_pcl_3d = pc_io.restructurePointcloud(ref_pcl, ref_pcl_2d_structure)
    raw_ref_ref = pc_io.loadPcd(metadata_io.get_reference_path(subject_name, ref_type='REF'))

    eye_mask = get_eye_mask(raw_ref_ref)
    eye_cloud = ref_pcl_3d[eye_mask]
    return eye_cloud


def get_boundaries(pc,rounding=True):
    """
    determines the minimum and maximum coordinates of points in a pointcloud with fixed rounding
    also returns the center point
    pc - pointcloud
    return - point with floored minimum values, point with ceiled maximum values, center point (exactly between the other two)
    """
    if len(pc.shape) == 2:
        if rounding:
            pt_min = (round(np.nanmin(pc[:,0])-25,-1),round(np.nanmin(pc[:,1])-25,-1),round(np.nanmin(pc[:,2])-25,-1))
            pt_max = (round(np.nanmax(pc[:,0])+25,-1),round(np.nanmax(pc[:,1])+25,-1),round(np.nanmax(pc[:,2])+25,-1))
        else:
            pt_min = (np.nanmin(pc[:,0]),np.nanmin(pc[:,1]),np.nanmin(pc[:,2]))
            pt_max = (np.nanmax(pc[:,0]),np.nanmax(pc[:,1]),np.nanmax(pc[:,2]))
    elif len(pc.shape) == 3:
        if rounding:
            pt_min = (round(np.nanmin(pc[:,:,0])-25,-1),round(np.nanmin(pc[:,:,1])-25,-1),round(np.nanmin(pc[:,:,2])-25,-1))
            pt_max = (round(np.nanmax(pc[:,:,0])+25,-1),round(np.nanmax(pc[:,:,1])+25,-1),round(np.nanmax(pc[:,:,2])+25,-1))
        else:
            pt_min = (np.nanmin(pc[:,:,0]),np.nanmin(pc[:,:,1]),np.nanmin(pc[:,:,2]))
            pt_max = (np.nanmax(pc[:,:,0]),np.nanmax(pc[:,:,1]),np.nanmax(pc[:,:,2]))
    else:
        raise NotImplementedError('dismension mismatch')

    center = ((pt_min[0] + pt_max[0]) /2,(pt_min[1] + pt_max[1]) /2,(pt_min[2] + pt_max[2]) /2)
    return pt_min,pt_max,center


# def get_back_of_head_point(subject, output_space='MT'):
#     """
#     Robust method to get a point on the skull furthest to the posterior direction
#     subject         subject ID
#     return          np array with 3 values (x,y,z)
#     """
#     ref_pc_path = metadata_io.get_reference_path(subject, 'T1')
#     nifti_data = nib.load(ref_pc_path)
#     nifti_data = mri_tools.conform(nifti_data)
#     nifti_data_raw = nifti_data.get_fdata()
#     #nifti_data = DataTools.loadReference(subject, ref_type='T1', output_space='RAS') # cant use this because we doesnt yield point-cloud

#     # careful, this says ras, but is operating in LIA !!!!!!!!!
#     t1_ref_pc_back, _ = mri_tools.mri_to_pc(nifti_data_raw, skin_threshold=25, reverse_direction=True)

#     sorting_index = np.argsort(t1_ref_pc_back[:,2])
#     back_of_head_slice = t1_ref_pc_back[sorting_index][10,2]  # select the depth that is tenth largest (for robustness, can be the same as the largest depth)
#     back_of_head_indices = np.where(t1_ref_pc_back[:,2] == back_of_head_slice)[0]
#     back_of_head_indices = np.append(back_of_head_indices, np.where(t1_ref_pc_back[:,2] == back_of_head_slice+1)[0])
#     back_of_head_indices = np.append(back_of_head_indices, np.where(t1_ref_pc_back[:,2] == back_of_head_slice+2)[0])

#     back_of_head = t1_ref_pc_back[back_of_head_indices]

#     back_of_head_center = np.median(back_of_head,axis=0)

#     back_of_head_center_ras = transformation_tools.applyTransformation(back_of_head_center[None], nifti_data.affine).squeeze()

#     if output_space == 'RAS':
#         return back_of_head_center_ras
#     elif output_space == 'MT':
#         raise NotImplementedError('Handling of MT space not implemented')
#         t1ar_to_mt_transform = np.load(os.path.join(T1_TO_MT_DIR, 'mat' ,subject + '.npy'))
#         return transformation_tools.applyTransformation(back_of_head_center_ras[None], t1ar_to_mt_transform).squeeze()
