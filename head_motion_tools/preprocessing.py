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


def getEyeMask(pc_ref, getIdx=False):
    """
    Extracts the eye mask from the reference pointcloud
    pc_ref          (N,6) array (pointcloud)
    getIdx          if True, returns the indices of the mask
    return          (N,6) boolean array (mask), (optional) (N,6) array (indices)
    """

    mask = np.zeros((pc_ref.shape[0],pc_ref.shape[1]), dtype=bool)
    mask[np.logical_and(np.logical_and(pc_ref[:,:,3] > 100, pc_ref[:,:,4] == 0), pc_ref[:,:,5] == 0)] = True
    if getIdx:
        return mask, np.nonzero(mask)
    else:
        return mask


def copyEye(pc_ref,pc_pcl, inplace=False):
    """
    copy eye from pcl to ref file, if the eye is cut out as a closed off hole
    pc_ref  segmented pointcloud with cut out eye
    pc_pcl  complete pointcloud (artifacts + eye)

    return  segmented pointcloud with eye
    """

    eye_mask = getEyeMask(pc_ref)

    if inplace:
        pc_ref[eye_mask] = pc_pcl[eye_mask]
        return pc_ref
    else:
        new_ref = pc_ref.copy()
        new_ref[eye_mask] = pc_pcl[eye_mask]
        return new_ref


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
