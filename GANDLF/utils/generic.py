import os, datetime, pkg_resources, sys
import numpy as np


def checkPatchDivisibility(patch_size, number=16):
    """
    This function checks the divisibility of a numpy array or integer for architectural integrity

    Args:
        patch_size (numpy.array): The patch size for checking.
        number (int, optional): The number to check divisibility for. Defaults to 16.

    Returns:
        bool: If all elements of array are divisible or not, after taking 2D patches into account.
    """
    if isinstance(patch_size, int):
        patch_size_to_check = np.array(patch_size)
    else:
        patch_size_to_check = patch_size
    # for 2D, don't check divisibility of last dimension
    if patch_size_to_check[-1] == 1:
        patch_size_to_check = patch_size_to_check[:-1]
    # for 2D, don't check divisibility of first dimension
    elif patch_size_to_check[0] == 1:
        patch_size_to_check = patch_size_to_check[1:]
    if np.count_nonzero(np.remainder(patch_size_to_check, number)) > 0:
        return False

    # adding check to address https://github.com/CBICA/GaNDLF/issues/53
    # there is quite possibly a better way to do this
    unique = np.unique(patch_size_to_check)
    if (unique.shape[0] == 1) and (unique[0] <= number):
        return False
    return True


def fix_paths(cwd):
    """
    This function takes the current working directory of the script (which is required for VIPS) and sets up all the paths correctly

    Args:
        cwd (str): The current working directory.
    """
    if os.name == "nt":  # proceed for windows
        vipshome = os.path.join(cwd, "vips/vips-dev-8.10/bin")
        os.environ["PATH"] = vipshome + ";" + os.environ["PATH"]


def get_date_time():
    """
    Get a well-parsed date string

    Returns:
        str: The date in format YYYY/MM/DD::HH:MM:SS
    """
    now = datetime.datetime.now().strftime("%Y/%m/%d::%H:%M:%S")
    return now


def get_filename_extension_sanitized(filename):
    """
    This function returns the extension of the filename with leading and trailing characters removed.
    Args:
        filename (str): The filename to be processed.
    Returns:
        str: The filename with extension removed.
    """
    _, ext = os.path.splitext(filename)
    # if .gz or .nii file is detected, always return .nii.gz
    if (ext == ".gz") or (ext == ".nii"):
        ext = ".nii.gz"
    return ext


def parse_version(version_string):
    """
    Parses version string, discards last identifier (NR/alpha/beta) and returns an integer for comparison.

    Args:
        version_string (str): The string to be parsed.

    Returns:
        int: The version number.
    """
    version_string_split = version_string.split(".")
    if len(version_string_split) > 3:
        del version_string_split[-1]
    return int("".join(version_string_split))


def version_check(version_from_config):
    """
    This function checks if the version of the config file is compatible with the version of the code.

    Args:
        version_from_config (str): The version of the config file.

    Returns:
        bool: If the version of the config file is compatible with the version of the code.
    """
    gandlf_version = pkg_resources.require("GANDLF")[0].version
    gandlf_version_int = parse_version(gandlf_version)
    min_ver = parse_version(version_from_config["minimum"])
    max_ver = parse_version(version_from_config["maximum"])
    if (min_ver > gandlf_version_int) or (max_ver < gandlf_version_int):
        sys.exit(
            "Incompatible version of GaNDLF detected (" + gandlf_version + ")"
        )
        
    return True