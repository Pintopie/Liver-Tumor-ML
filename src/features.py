import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm

from .data import read_nii

def calculate_features(segmentation_array: np.ndarray) -> dict:
    """
    Calculate features from a segmentation array.
    """
    features = {}
    features['tumor_volume'] = np.sum(segmentation_array)
    tumor_pixels = segmentation_array[segmentation_array == 1]
    if tumor_pixels.size > 0:
        features['mean_intensity'] = np.mean(tumor_pixels)
        features['variance_intensity'] = np.var(tumor_pixels)
        features['skewness_intensity'] = scipy.stats.skew(tumor_pixels)
        features['kurtosis_intensity'] = scipy.stats.kurtosis(tumor_pixels)
    else:
        features['mean_intensity'] = 0
        features['variance_intensity'] = 0
        features['skewness_intensity'] = 0
        features['kurtosis_intensity'] = 0
    return features

def extract_features(segmentation_files: list) -> pd.DataFrame:
    """
    Extract features for all segmentation files.
    """
    data = []
    for seg_file in tqdm(segmentation_files, desc="Extracting features"):
        segmentation_array = read_nii(seg_file)
        features = calculate_features(segmentation_array)
        features['file'] = seg_file
        data.append(features)
    df = pd.DataFrame(data)
    return df
