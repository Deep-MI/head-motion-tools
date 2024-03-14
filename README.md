# Quantifying MR head motion using markerless tracking

This is a repository for rigid & robust point cloud registration and analysis of head motion, as described in our paper, "Quantifying MR head motion in the Rhineland Study - A Robust Method for Population Cohorts", https://doi.org/10.1016/j.neuroimage.2023.120176 .

Apart from a library of functions for viewing, registering and analyzing sequential point-cloud and motion data, this repository contains two python3.10 scripts for i. Generating head motion estimates, motion traces, and motion estimates during MRI scans from raw point cloud data, ii. Viewing raw point cloud data. 

## Registering motion tracking data (sequential point clouds)

The script run_registration.py automatically registers a series of point clouds captured by a motion-tracking camera and computes the head positions, motion metrics and plots.

Example usage:
```bash
python3 run_registration.py -i /path/to/TracSuite/output/session_ID -o /output/folder/session_ID
```

Running this command will automatically generate multiple output files in "/output/folder/session_ID".

Parameters:
```
-i              Input directory: The path to a directory with created by TracSuite all metadata is required to be present. If some point-clouds are missing they will be skipped.
-o              An empty output directory. If the directory contains data from a previous run it will be used to speed up the current processing.
--t1                          (optional) Path to a T1-weighted MRI image to stabilize the registration. This may increase the registration and postprocessing accuracy. However, the registration may need to be modified for new scan setups & T1-weighted acquisition protocols.
--acquisition_times_csv       (optional) A comma seperated file with the starting times of the acquisition present in the session. This can be extracted from DICOM headers.
--sequence_length_json        (optional) A json file containing the length of acquisitions in the present session.
--tracsuite_registration      (optional) skip sequential registration and load registration from TracSuite instead.
--deviation                   (optional) Use the deviation from the starting position instead of the speed of the head.
```


## Viewing data

The second script can be used to view a series of point-cloud files captured by an in-MRI motion tracking camera.

Example usage:
```bash
python3 show_series.py -i /path/to/TracSuite/output/session_ID -o /output/folder/session_ID --crop 0 500
```

This script will open a window displaying the captured data. If the "--crop" parameter is specified, only a range of point clouds will be opened (for example, the first 500).


## Citation


> Pollak, C., Kügler, D., Breteler, M.M. and Reuter, M., 2023. Quantifying MR head motion in the Rhineland Study–A robust method for population cohorts. NeuroImage, 275, p.120176.
