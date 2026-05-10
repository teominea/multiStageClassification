# Multi-Stage Classification for Human Action Recognition
---

## Project Description

This project implements a two-stage classification pipeline for skeleton-based human action recognition using the NTU RGB+D 60 dataset. The pipeline consists of a Spatial-Temporal Graph Convolutional Network (ST-GCN) as the primary classifier, followed by a Multi-Layer Perceptron (MLP) that refines the final prediction by combining the ST-GCN output with statistical features extracted from the raw skeleton data.

The goal of this thesis is to evaluate whether a two-stage architecture produces measurable improvements in classification accuracy compared to each model trained independently.

---

## Repository Structure

```
.
|-- eda/
|   |-- apply_eda.py                    # EDA script
|-- eda_output/
|   |-- 1_class_distribution.png
|   |-- 2_metadata_distribution.png
|   |-- 3_sequence_length.png
|   |-- 4_skeleton_quality.png
|   |-- 5_joint_position_stats.png
|   |-- 5b_mean_skeleton_pose.png
|   |-- 6_motion_intensity.png
|   |-- 7_sequence_length_by_class.png
|-- preprocess_data/
|   |-- preprocess.py                   # Step 1 - Parse raw skeleton files
|   |-- preprocess_step2.py             # Step 2 - Forward-fill missing frames
|   |-- preprocess_step3.py             # Step 3 - Coordinate normalization
|   |-- preprocess_step3b.py            # Step 3b - Outlier removal
|   |-- preprocess_step4.py             # Step 4 - Sequence length standardization
|   |-- preprocess_step5.py             # Step 5 - Train/test split and save
|-- network_pipeline/
|   |-- ntu60_stgcn_pipeline.ipynb      # Full training pipeline notebook
```

---

## Preprocessing Pipeline

### Step 1 - Parse Raw Skeleton Files (`preprocess.py`)

Reads all raw `.skeleton` files and converts them into numpy arrays of shape (F, 25, 3). The 302 officially corrupted files published by the dataset authors are excluded before parsing. For two-person interaction classes, both bodies are read and their joint positions are averaged per frame, using a greedy matching algorithm to ensure consistent body ordering across consecutive frames.

Output: `step1_parsed.pkl`

### Step 2 - Forward-Fill Missing Frames (`preprocess_step2.py`)

Some frames contain no detected body due to Kinect sensor dropout. These frames are stored as all-zero arrays in Step 1. This step replaces each missing frame with a copy of the last valid frame before it. For sequences where the very first frame is missing, the first valid frame is propagated backwards.

Output: `step2_filled.pkl`

### Step 3 - Coordinate Normalization (`preprocess_step3.py`)

The raw skeleton coordinates are in camera space, meaning they reflect the physical distance between the subject and the sensor rather than the body movement itself. Two normalization operations are applied per frame:

1. **Translation** - the base of spine joint is moved to the origin (0, 0, 0), making all other joint positions relative to the body center.
2. **Scaling** - all joint coordinates are divided by the torso length (distance from spine base to neck), making the representation more consistent across subjects of different sizes and distances from the sensor.

Output: `step3_normalized.pkl`

### Step 3b - Outlier Removal (`preprocess_step3b.py`)

After normalization, samples where any joint value exceeds 20 times the torso length are flagged as physically impossible. These are caused by residual Kinect body-swap tracking errors in two-person interaction classes. The flagged samples are removed from the dataset.

Output: `step3_cleaned.pkl`

### Step 4 - Sequence Length Standardization (`preprocess_step4.py`)

Sequences in the dataset range from 32 to 300 frames. Since all models require fixed-length input, sequences are standardized to 100 frames. Sequences shorter than 100 frames are zero-padded at the end. Sequences longer than 100 frames are truncated from the end.

Output: `step4_padded.pkl`

### Step 5 - Train/Test Split and Save (`preprocess_step5.py`)

Applies the official cross-subject train/test split and saves the final arrays in numpy format. The axis order is reordered from (100, 25, 3) to (3, 100, 25) to match the input convention expected by the ST-GCN model. Class labels are converted from the original range 1-60 to 0-59 for compatibility with PyTorch.

Output:
- `train_data.npy` - shape (40069, 3, 100, 25)
- `train_labels.npy` - shape (40069,)
- `test_data.npy` - shape (16468, 3, 100, 25)
- `test_labels.npy` - shape (16468,)

---

## Model Architecture

### Stage 1 - Spatial-Temporal Graph Convolutional Network (ST-GCN)

The first stage is based on the ST-GCN architecture proposed by Yan et al. The skeleton is modeled as a graph where the 25 joints are nodes and the physical bones connecting them are edges. This structural representation allows the model to learn spatial dependencies between connected body joints explicitly, rather than treating joint coordinates as a flat vector.

The model consists of 9 ST-GCN blocks stacked sequentially, each combining two operations:

- **Spatial graph convolution** - for each joint at each frame, aggregates feature information from all physically connected neighboring joints using the normalized adjacency matrix.
- **Temporal convolution** - applies a 1D convolution with a 9-frame kernel along the time axis for each joint independently, capturing local motion patterns across consecutive frames.

Each block also includes a residual connection, which adds the block input directly to its output to prevent gradient vanishing across the 9 stacked blocks. (I followed the architecture as presented in this article: https://thachngoctran.medium.com/spatial-temporal-graph-convolutional-networks-st-gcn-explained-bf926c811330)

The channel progression across the 9 blocks is as follows:

```
Input          : (batch, 3,   100, 25)
Blocks 1 to 3  : (batch, 64,  100, 25)
Blocks 4 to 6  : (batch, 128, 100, 25)
Blocks 7 to 9  : (batch, 256, 100, 25)
```

After the final block, global average pooling collapses the time and joint dimensions into a single 256-dimensional vector, which is passed to a linear classifier producing 60 class scores.

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts (T0=20) |
| Loss | CrossEntropyLoss (label smoothing=0.1) |
| Batch size | 32 |
| Epochs | 60 |
| Random seed | 42 |

**Data augmentation applied during training:**
- Random rotation around the vertical axis (+/- 15 degrees)
- Random translation (+/-0.1 in each axis)

### Stage 2 - Multi-Layer Perceptron (MLP)

The second stage receives two inputs simultaneously and learns to combine them into a refined final prediction:

**Input 1 - ST-GCN logits**

The raw class scores produced by the frozen Stage 1 model, representing its confidence for each of the 60 action classes.

**Input 2 - Skeleton statistical features**

Statistical summaries computed directly from the raw skeleton sequence, capturing global properties that the ST-GCN local 9-frame temporal window does not model:

| Feature | Size | Description |
|---|---|---|
| Mean joint position | 75 | Average position of each joint across all frames |
| Std joint position | 75 | Movement range of each joint |
| Max joint position | 75 | Peak position of each joint |
| Min joint position | 75 | Lowest position of each joint |
| Mean joint velocity | 75 | Average frame-to-frame displacement |
| Std joint velocity | 75 | Variance of frame-to-frame displacement |
| Mean bone vector | 60 | Average orientation of each of the 20 physical bones |
| Std bone vector | 60 | Variance of bone orientations |

The concatenated input vector of 630 values is passed through a 5-layer MLP with residual connections:

```
Input    : 630
Layer 1  : 630 to 512  (Linear + BatchNorm + ReLU + Dropout)
Layer 2  : 512 to 512  (Linear + BatchNorm + ReLU + Dropout + residual)
Layer 3  : 512 to 256  (Linear + BatchNorm + ReLU + Dropout)
Layer 4  : 256 to 256  (Linear + BatchNorm + ReLU + Dropout + residual)
Output   : 256 to 60
```

**Training configuration:**

| Parameter | Value |
|---|---|
| ST-GCN weights | Frozen (not updated) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts (T0=30) |
| Loss | CrossEntropyLoss (label smoothing=0.1) |
| Epochs | 60 |


## Results 

All experiments were conducted on the NTU RGB+D 60 dataset with a fixed random seed of 42 to ensure reproducibility.

| Model | Test Accuracy |
|---|---|
| ST-GCN Standalone | 71.49% |
| ST-GCN + MLP Pipeline | 78.23% |
| ST-GCN + LSTM Pipeline | 76.96% |
