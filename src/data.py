import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Dict

def read_nii(filepath: str) -> np.ndarray:
    """
    Reads a .nii file and returns a rotated pixel array.
    """
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array

def get_image_and_segmentation_files(volume_dirs: List[Path], segmentations_dir: Path) -> Dict[str, List[str]]:
    """
    Returns sorted lists of image and segmentation file paths.
    """
    image_files = []
    for volume_dir in volume_dirs:
        image_files.extend(sorted(volume_dir.glob("volume-*.nii")))
    segmentation_files = sorted(segmentations_dir.glob("segmentation-*.nii"))
    return {
        "images": [str(f) for f in image_files],
        "segmentations": [str(f) for f in segmentation_files]
    }
