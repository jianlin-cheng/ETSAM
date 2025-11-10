# Taken from https://github.com/teamtomo/membrain-seg/blob/main/src/membrain_seg/tomo_preprocessing
from dataclasses import dataclass
import warnings
import numpy as np
import mrcfile
from typing import Any, Optional

@dataclass
class Tomogram:
    """
    A class used to represent a Tomogram.

    Attributes
    ----------
    data : np.ndarray
        The 3D array data representing the tomogram.
    header : Any
        The header information from the tomogram file.
    voxel_size : Any, optional
        The voxel size of the tomogram.
    """

    data: np.ndarray
    header: Any
    voxel_size: Optional[Any] = None


def load_tomogram(
    filename: str,
    normalize_data: bool = False,
) -> Tomogram:
    """
    Loads tomogram and transposes s.t. we have data in the form x,y,z.

    If specified, also normalizes the tomogram.

    Parameters
    ----------
    filename : str
        File name of the tomogram to load.
    normalize_data : bool, optional
        If True, normalize data.

    Returns
    -------
    tomogram : Tomogram
        A Tomogram dataclass containing the loaded data, header
        and voxel size.

    """
    warnings.filterwarnings(
        "ignore",
        message="Map ID string not found - \
not an MRC file, or file is corrupt",
    )
    warnings.filterwarnings(
        "ignore",
        message="Unrecognised machine stamp: \
0x00 0x00 0x00 0x00",
    )
    with mrcfile.open(filename, permissive=True) as tomogram:
        data = tomogram.data.copy()
        data = np.transpose(data, (2, 1, 0))
        header = tomogram.header
        voxel_size = tomogram.voxel_size
    if normalize_data:
        data -= np.mean(data)
        data /= np.std(data)
    return Tomogram(data=data, header=header, voxel_size=voxel_size)
