import nibabel as nib
import matplotlib.pyplot as plt

def visualize_slice(nii_file_path: str, slice_idx: int = None) -> None:
    """
    Visualize a slice from a NIfTI file.
    """
    nii_image = nib.load(nii_file_path)
    nii_data = nii_image.get_fdata()
    if slice_idx is None:
        slice_idx = nii_data.shape[2] // 2
    plt.imshow(nii_data[..., slice_idx], cmap='gray')
    plt.title(f"Slice {slice_idx}")
    plt.axis('off')
    plt.show()
