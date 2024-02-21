from glob import glob
import os


def find_file_by_wildcard(dir, wildcard):
    """
    Helper that returns a file that match a wildcard in "dir", raises an Error in other
    """
    files = glob(os.path.join(dir, wildcard))

    if not len(files) == 1:
        raise FileNotFoundError('found ' + str(len(files)) + ' files matching ' + str(os.path.join(dir, wildcard)))
    return files[0]


def find_files_by_wildcard(dir, wildcard):
    """
    Helper that returns a list of all files that match a wildcard in "dir"
    """
    return glob(os.path.join(dir,wildcard))

