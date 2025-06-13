import argparse
import logging
from pathlib import Path

from src.data import get_image_and_segmentation_files
from src.features import extract_features
from src.visualization import visualize_slice
from src.model import train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    parser = argparse.ArgumentParser(description="Liver Tumor ML Pipeline")
    parser.add_argument('--data_dir', type=str, required=True, help='Root data directory')
    parser.add_argument('--output_features', type=str, default='features.csv', help='Output CSV for features')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--visualize', type=str, help='Visualize a NIfTI file')
    parser.add_argument('--slice', type=int, default=None, help='Slice index for visualization')
    args = parser.parse_args()

    segmentations_dir = Path(args.data_dir) / "segmentations"
    volume_dirs = [Path(args.data_dir) / f"volume_pt{i}" for i in range(1, 6)]

    files = get_image_and_segmentation_files(volume_dirs, segmentations_dir)
    df = extract_features(files['segmentations'])
    df.to_csv(args.output_features, index=False)
    logging.info(f"Features saved to {args.output_features}")

    if args.visualize:
        visualize_slice(args.visualize, args.slice)

    if args.train:
        # Dummy target for demonstration; replace with actual labels
        import numpy as np
        y = np.random.randint(0, 2, size=len(df))
        train_model(df.drop(columns=['file']), y)

if __name__ == "__main__":
    main()
