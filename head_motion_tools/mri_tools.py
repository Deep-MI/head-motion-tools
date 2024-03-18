import numpy as np
import nibabel as nib

from head_motion_tools import transformation_tools, ircp

from head_motion_tools.visualization import VtkVisualizer, VtkTools


def get_back_of_head_point(conformed_nibabel_image: nib.Nifti1Image):
    """
    Robust method to get a point on the skull furthest to the posterior direction
    subject         subject ID
    return          np array with 3 values (x,y,z)
    """
    
    nifti_data_raw = conformed_nibabel_image.get_fdata()

    # convert LIA mri to point cloud
    t1_ref_pc_back, _ = mri_to_pc(nifti_data_raw, intensity_threshold=25, back_to_front=True)

    sorting_index = np.argsort(t1_ref_pc_back[:,2])
    back_of_head_slice = t1_ref_pc_back[sorting_index][10,2]  # select the depth that is tenth largest (for robustness, can be the same as the largest depth)
    back_of_head_indices = np.where(t1_ref_pc_back[:,2] == back_of_head_slice)[0]
    back_of_head_indices = np.append(back_of_head_indices, np.where(t1_ref_pc_back[:,2] == back_of_head_slice+1)[0])
    back_of_head_indices = np.append(back_of_head_indices, np.where(t1_ref_pc_back[:,2] == back_of_head_slice+2)[0])
    back_of_head = t1_ref_pc_back[back_of_head_indices]
    back_of_head_center = np.median(back_of_head,axis=0)

    # convert point to ras
    back_of_head_center_ras = transformation_tools.applyTransformation(back_of_head_center[None], conformed_nibabel_image.affine).squeeze()

    return back_of_head_center_ras



def get_stabilizing_point(mri_path, reference_point_cloud, debug=False):

    nibabel_image = nib.load(mri_path)

    # robust scaling and converts to LIA
    nibabel_image_conformed = conform(nibabel_image)
    
    # get transformation from point cloud to MRI RAS space
    pc_to_ras_mri_mapping = register_pc_to_mri(nibabel_image_conformed, reference_point_cloud, debug=debug)

    # get stabilizing point in RAS
    back_of_head_pt_ras = get_back_of_head_point(nibabel_image_conformed)

    # convert to motiontracker space
    stabilizing_point = transformation_tools.applyTransformation(back_of_head_pt_ras[None], np.linalg.inv(pc_to_ras_mri_mapping)).squeeze()


    if debug: # display registration & stabilizing point for QC
        mri_pc, _ = mri_to_pc(nibabel_image_conformed.get_fdata(), back_to_front=False)
        mri_pc = np.append(mri_pc, mri_to_pc(nibabel_image_conformed.get_fdata(), back_to_front=True)[0], axis=0)
        mri_pc_ras = transformation_tools.applyTransformation(mri_pc, nibabel_image_conformed.affine)
        mri_pc_mt = transformation_tools.applyTransformation(mri_pc_ras, np.linalg.inv(pc_to_ras_mri_mapping))

        VtkVisualizer.displayOverlaidPointClouds([
            VtkTools.VtkPointCloud(reference_point_cloud, colors='red').vtkActor,
            VtkTools.VtkArrow(startPoint=stabilizing_point+np.array([0,0,10]), endPoint=stabilizing_point).vtkActor,
            VtkTools.VtkPointCloud(mri_pc_mt).vtkActor
        ])

    return stabilizing_point, pc_to_ras_mri_mapping




def mri_to_pc(nifti_data, coronal_axis=2, intensity_threshold=25, offset = 5, back_to_front=False):
    """
    grabs values in coronal direction above a certain threshold

    :param np.ndarray nifti_data: 3D image data in 
    :param int coronal_axis: axis to look at (0=x, 1=y, 2=z)
    :param int intensity_threshold: threshold for intensity
    :param int offset: offset in coronal direction
    :param bool reverse_direction: if true, looks in reverse direction
    :return: location of values and colors
    """

    if back_to_front:
        depth = (nifti_data>intensity_threshold).argmax(axis=coronal_axis) # find points closest to front with intesity lower than treshold
        indices = np.where(depth) # find indices of nonzeros
        depth_pts = depth[indices] # convert to 2d
    else:
        depth = (np.flip(nifti_data, axis=coronal_axis)>intensity_threshold).argmax(axis=coronal_axis)
        offset *= -1
        indices = np.where(depth)
        depth_pts = nifti_data.shape[coronal_axis]-1 -depth[indices]

    # undo flip 320 - x and 
    if coronal_axis == 0:
        mri_pc = np.array([depth_pts, indices[0], indices[1]]).T
    elif coronal_axis == 1:
        mri_pc = np.array([indices[0], depth_pts, indices[1]]).T
    elif coronal_axis == 2:
        mri_pc = np.array([indices[0], indices[1], depth_pts]).T
    else:
        raise ValueError('only 3d data is supported')

    colors_idxs = mri_pc.copy()
    colors_idxs[:,coronal_axis] += offset

    colors=[]
    for i in range(len(mri_pc)):
        colors.append(nifti_data[(colors_idxs[i,0], colors_idxs[i,1], colors_idxs[i,2])])


    return mri_pc, np.array(colors)


def register_pc_to_mri(nifti_data_raw, reference_pc, debug=True):
    """
    Registers reference point cloud to MRI.

    :param: str mri_path: path to the MRI image
    :param: str pc_path: path to the point cloud
    :param: bool debug: if true, prints additional information
    :return: the homogenous transformation matrix (4x4) that maps the point cloud to the MRI space
    """

    # convert LIA mri to point cloud
    mri_front, mri_colors = mri_to_pc(nifti_data_raw.get_fdata(), intensity_threshold=25, back_to_front=False)

    # convert to RAS
    mri_front = transformation_tools.applyTransformation(mri_front, nifti_data_raw.affine)


    # crop top and bottom of the mri point cloud
    center_area_idx = np.where((mri_front[:,2] < 50) & (mri_front[:,2] > -100) & (mri_front[:,1] > 20))
    mri_front = mri_front[center_area_idx]
    mri_colors = mri_colors[center_area_idx]


    # apply pre-registration transformation (may be individual for every scan setup)
    pre_registration_transformation = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) # 90 degree back rotation
    pre_registration_transformation = np.array([[ -1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) @ pre_registration_transformation  # 180 degree lr rotation
    #pre_registration_transformation = np.array([[ 1, 0, 0, 30], [0, 1, 0, 260], [0, 0, 1, 0], [0, 0, 0, 1]]) @ pre_registration_transformation  # 60 mm to the frond
    pre_registration_transformation = np.array([[ 1, 0, 0, 40], [0, 1, 0, 300], [0, 0, 1, 0], [0, 0, 0, 1]]) @ pre_registration_transformation  # 40 mm to the right and 300 mm to the front
    #pre_registration_transformation = np.array([[ 1, 0, 0, 100], [0, 1, 0, 100], [0, 0, 1, 90], [0, 0, 0, 1]]) @ pre_registration_transformation  # 40 mm to the right and 300 mm to the front
    pre_registration_transformation = np.array([[0.9990482,  0.0436194, 0, 0], [-0.0435928,  0.9984396, -0.0348995, 0],[-0.0015223, 0.0348663, 0.9993908,0], [0, 0, 0, 1]]) @ pre_registration_transformation # empericially determined rotation

    # pre_registration_transformation = np.array([[ -0.5456259,  0.7728130, -0.3241180, 0],
    #                                             [0.7728130,  0.6135935,  0.1620590, 0],
    #                                         [0.3241180, -0.1620590, -0.9320324, 0],
    #                                         [0, 0, 0, 1]])

    #reference_pc = transformation_tools.applyTransformation(reference_pc, pre_registration_transformation)


    # subsample the point cloud
    reference_pc_subsampled = reference_pc[::8]
    mri_front_subsampled = mri_front[::4]
    
    print('starting registration to structural image')
    # apply registration in a grid search

    transformations = []
    errors = []

    if debug:
        registered_pc_list = []
        weights_list = []
        pre_transformations = []

    for i in range(-100, 100, 10):
        added_transformation = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, i], [0, 0, 0, 1]])

        current_pre_registration_transformation = added_transformation @ pre_registration_transformation

        reference_pc_pre_aligned = transformation_tools.applyTransformation(reference_pc_subsampled, current_pre_registration_transformation)

        if debug:
            pre_reg_to_mri, registered_pc, weights, error = ircp.fast_ircp(
                mri_front_subsampled.astype(np.float64).T,
                reference_pc_pre_aligned.astype(np.float64).T, 
                maxIter=30, critFun=2, est_b=0.7, 
                getError=True, getWeights=True
            )
            registered_pc_list.append(registered_pc)
            weights_list.append(weights)
            pre_transformations.append(current_pre_registration_transformation)
        else:
            pre_reg_to_mri, registered_pc, error = ircp.fast_ircp(
                mri_front_subsampled.astype(np.float64).T,
                reference_pc_pre_aligned.astype(np.float64).T, 
                maxIter=200, critFun=2, est_b=0.7, 
                getError=True, getWeights=False
            )
        transformations.append(pre_reg_to_mri @ current_pre_registration_transformation)
        errors.append(error)

    print('finished registration to structural image')

    #print('errors:', errors)
    chosen = np.argmin(errors)
    #print(errors[chosen])

    
    if debug:
        reference_pc_pre_aligned = transformation_tools.applyTransformation(reference_pc, pre_transformations[chosen])

        VtkVisualizer.displayOverlaidPointCloud(
            VtkTools.VtkPointCloud(mri_front, colors=mri_colors, color_map='gray'),
            VtkTools.VtkPointCloud(reference_pc_pre_aligned, colors='green')
        )

        VtkVisualizer.displayOverlaidPointCloud(
            VtkTools.VtkPointCloud(mri_front, colors=mri_colors, color_map='gray'),
            VtkTools.VtkPointCloud(registered_pc_list[chosen].T, colors=weights_list[chosen], color_map='color')
        )

    #mt_to_ras = pre_reg_to_mri @ pre_registration_transformation

    return transformations[chosen]

def map_image(img, out_affine, out_shape, ras2ras=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
              order=1):
    """
    Function to map image to new voxel space (RAS orientation)

    :param nibabel.MGHImage img: the src 3D image with data and affine set
    :param np.ndarray out_affine: trg image affine
    :param np.ndarray out_shape: the trg shape information
    :param np.ndarray ras2ras: ras2ras an additional maping that should be applied (default=id to just reslice)
    :param int order: order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: mapped Image data array
    """
    from scipy.ndimage import affine_transform
    from numpy.linalg import inv

    # compute vox2vox from src to trg
    vox2vox = inv(out_affine) @ ras2ras @ img.affine

    # here we apply the inverse vox2vox (to pull back the src info to the target image)
    new_data = affine_transform(img.get_fdata(), inv(vox2vox), output_shape=out_shape, order=order)
    return new_data

def getscale(data, dst_min, dst_max, f_low=0.0, f_high=0.999, verbose=False):
    """
    Function to get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.
    Equivalent to how mri_convert conforms images.

    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param f_low: robust cropping at low end (0.0 no cropping)
    :param f_high: robust cropping at higher end (0.999 crop one thousandths of high intensity voxels)
    :return: returns (adjusted) src_min and scale factor
    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    if src_min < 0.0:
        raise ValueError('ERROR: Min value in input is below 0.0!')

    if verbose:
        print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cummulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # print("bin min: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(cs[idx])+"\n")
    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print('ERROR: rescale upper bound not found')

    src_max = idx * bin_size + src_min
    # print("bin max: "+format(idx)+"  nth: "+format(nth)+"  passed: "+format(voxnum-cs[idx])+"\n")

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    if verbose:
        print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))

    return src_min, scale


def scalecrop(data, dst_min, dst_max, src_min, scale, verbose=False):
    """
    Function to crop the intensity ranges to specific min and max values

    :param np.ndarray data: Image data (intensity values)
    :param float dst_min: future minimal intensity value
    :param float dst_max: future maximal intensity value
    :param float src_min: minimal value to consider from source (crops below)
    :param float scale: scale value by which source will be shifted
    :return: scaled Image data array
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    if verbose:
        print("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))

    return data_new


def conform(img, order=1):
    """
    Python version of mri_convert -c, which turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1 mm isotropic voxel sizes.

    Difference to mri_convert -c is that we first interpolate (float image), and then rescale to uchar. mri_convert is
    doing it the other way. However, we compute the scale factor from the input to be more similar again

    :param nibabel.MGHImage img: loaded source image
    :param int order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:nibabel.MGHImage new_img: conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader

    cwidth = 256
    csize = 1
    h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format

    h1.set_data_shape([cwidth, cwidth, cwidth, 1])
    h1.set_zooms([csize, csize, csize])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = cwidth
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # get scale for conversion on original input before mapping to be more similar to mri_convert
    src_min, scale = getscale(img.get_fdata(), 0, 255)

    mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)
    # print("max: "+format(np.max(mapped_data)))

    if not img.get_data_dtype() == np.dtype(np.uint8):

        if np.max(mapped_data) > 255:
            mapped_data = scalecrop(mapped_data, 0, 255, src_min, scale)

    new_data = np.uint8(np.rint(mapped_data))
    new_img = nib.MGHImage(new_data, h1.get_affine(), h1)

    # make sure we store uchar
    new_img.set_data_dtype(np.uint8)

    return new_img
