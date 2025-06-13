# Liver Tumor ML Project

A professional machine learning pipeline for liver tumor segmentation and analysis using NIfTI medical images.

## Features

- Feature extraction from segmentation masks
- Model training (Random Forest)
- Visualization of NIfTI slices
- Containerized with Docker

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run feature extraction

```bash
python Model.py --data_dir /path/to/liver-tumor-dataset
```

### 3. Train a model

```bash
python Model.py --data_dir /path/to/liver-tumor-dataset --train
```

### 4. Visualize a NIfTI file

```bash
python Model.py --data_dir /path/to/liver-tumor-dataset --visualize /path/to/file.nii --slice 50
```

### 5. Docker

```bash
docker build -t liver-tumor-ml .
docker run --rm -v /local/data:/data liver-tumor-ml --data_dir /data
```

## Project Structure

- `Model.py` - Main pipeline script
- `requirements.txt` - Python dependencies
- `Dockerfile` - Containerization
- `README.md` - Documentation

## License

MIT License
