# UrbanSound8K Classifier and Spectrogram Visualization

This folder contains scripts for:
1. **Visualizing** spectrograms of the UrbanSound8K dataset using different window functions.
2. **Training** and **Testing** models (both scikit-learn SVM/MLP classifiers and a custom CNN) to classify audio samples in the UrbanSound8K dataset.

## Table of Contents
- [Folder Structure](#folder-structure)
- [Dataset](#dataset)
- [Usage](#usage)
  - [1. Spectrogram Visualization (`visualize.py`)](#1-spectrogram-visualization-visualizepy)
  - [2. Training and Testing with scikit-learn (`train_sklearn.py` and `test_sklearn.py`)](#2-training-and-testing-with-scikit-learn-train_sklearnpy-and-test_sklearnpy)
  - [3. Training and Testing the Custom CNN (`train.py` and `test.py`)](#3-training-and-testing-the-custom-cnn-trainpy-and-testpy)
- [Model Files](#model-files)
- [Notes](#notes)

---

## Folder Structure

```
spectrogram_visualization/
│
├── dataloader.py              # Utility script containing window functions and data loading
├── getmodel.py                # Helper script (if any) to instantiate models
├── main.py                    # (Optional) May contain orchestrating code for overall workflow
├── model.py                   # Custom CNN model implementation
├── model_sklearn.py           # scikit-learn model definitions (SVM, MLPClassifier)
├── test_sklearn.py            # Testing script for scikit-learn models
├── train_sklearn.py           # Training script for scikit-learn models
├── test.py                    # Testing script for the custom CNN model
├── train.py                   # Training script for the custom CNN model
├── visualize.py               # Spectrogram visualization script
└── README.md                  # This README
└── report_2a.pdf               # Task 2a report
```

---


## Dataset

The scripts assume you have the **UrbanSound8K** dataset downloaded and placed in a specific directory structure, typically something like:

```
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   ├── ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

Update any script paths accordingly to point to your dataset location.

---

## Usage

### 1. Spectrogram Visualization (`visualize.py`)

- **Purpose:** Generates and saves spectrogram images (using different window functions) for random samples from the UrbanSound8K dataset.
- **Key Code Portions:**
  - `save_spectrogram(y, sr, window_func, filename, title_suffix)`
  - `visualize_samples()`
- **How to Run:**
  ```bash
  python visualize.py
  ```
  (Adjust any hardcoded paths in `visualize.py` to point to your UrbanSound8K dataset folder and desired output directories.)

By default, this script:
- Loads 5 audio examples (one per fold, up to 5).
- Saves 3 spectrograms for each example (Hann, Hamming, Rectangular windows).
- Stores them in a folder named `spectrogram_visualizations`.

### 2. Training and Testing with scikit-learn (`train_sklearn.py` and `test_sklearn.py`)

- **Purpose (train_sklearn.py):** Trains either an SVM or an MLP on extracted audio features (using a specified window function).
  - **Arguments:**
    - `--window`: Choose which window function to use. Options: `['hann', 'hamming', 'rectangular']`.
    - `--model`: Choose which scikit-learn model to train. Options: `['svm', 'mlp']`.
  - **Example Usage:**
    ```bash
    python train_sklearn.py --window hann --model svm
    ```
    This trains an SVM using the Hann window function. Adjust dataset paths inside the script if necessary.

- **Purpose (test_sklearn.py):** Tests the trained scikit-learn model on a hold-out set or a specified fold.
  - **Arguments:**
    - `--window`: The same window function used during training.
    - `--model`: The same model type used during training (`svm` or `mlp`).
  - **Example Usage:**
    ```bash
    python test_sklearn.py --window hann --model svm
    ```
  Adjust or confirm that the paths to the saved model and data are correct.

### 3. Training and Testing the Custom CNN (`train.py` and `test.py`)

- **Purpose (train.py):** Trains a custom CNN (defined in `model.py`) on UrbanSound8K.
  - **Important Arguments:**  
    - `--metadata`: Path to the UrbanSound8K `metadata` CSV (default `"path/to/metadata.csv"`).  
    - `--window`: Window function for feature extraction (`hann`, `hamming`, or `rectangular`).  
    - `--epochs`: Number of training epochs (default `100`).  
    - `--batch_size`: Training batch size (default `32`).  
    - `--lr`: Learning rate (default `3e-4`).  
    - `--weight_decay`: Weight decay for the optimizer (default `1e-5`).  
    - `--lr_factor`: Factor by which to reduce LR if no improvement (default `0.1`).  
    - `--lr_patience`: Epochs without improvement before LR reduction (default `10`).  
    - `--model_out`: Path to save the trained PyTorch model checkpoint (default `"audio_cnn_hann.pth"`).

  - **Example Usage:**
    ```bash
    python train.py \
        --metadata "/data/b22cs089/speechPA1/dataset/UrbanSound8K/metadata/UrbanSound8K.csv" \
        --window hann \
        --epochs 50 \
        --batch_size 32 \
        --lr 0.001 \
        --weight_decay 1e-5 \
        --lr_factor 0.1 \
        --lr_patience 5 \
        --model_out "audio_cnn_hann.pth"
    ```

- **Purpose (test.py):** Tests the custom CNN on the specified test fold (commonly fold 10 for UrbanSound8K).
  - **Important Arguments:**  
    - `--metadata`: Path to `UrbanSound8K.csv`.  
    - `--window`: Window function used for feature extraction (same as used in training).  
    - `--batch_size`: Batch size for testing (default `128`).  
    - `--model_in`: Path to the trained model checkpoint (default `"/data/b22cs089/speechPA1/classifier/audio_cnn_hann.pth"`).

  - **Example Usage:**
    ```bash
    python test.py \
        --metadata "/data/b22cs089/speechPA1/dataset/UrbanSound8K/metadata/UrbanSound8K.csv" \
        --window hann \
        --batch_size 128 \
        --model_in "/data/b22cs089/speechPA1/classifier/audio_cnn_hann.pth"
    ```

---

## Notes

1. **Path Adjustments:** 
   - Make sure to update all paths in the scripts (e.g., dataset paths, model checkpoints) to match your local directory structure.
2. **Window Functions:** 
   - Implemented in `dataloader.py` as `hann_window`, `hamming_window`, `rectangular_window`.
3. **Custom CNN Architecture:** 
   - Defined in `model.py`; you may modify or extend it as needed.
4. **Feature Extraction:** 
   - By default, these scripts use STFT features with various window functions. You may adjust the parameters like `N_FFT`, `HOP_LENGTH`, etc.
5. **Performance Tuning:** 
   - The default hyperparameters might need tuning based on your hardware and experimental goals.

---
