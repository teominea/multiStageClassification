import os
import pickle
import numpy as np
from collections import Counter


INPUT_FILE  = "./preprocessed/step4_padded.pkl"
OUTPUT_DIR  = "./preprocessed"

# Official cross-subject training subjects (from NTU RGB+D paper)
TRAIN_SUBJECTS = {1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38}


def main():
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    # Split by subject
    train_samples = [s for s in data if s["subject"] in TRAIN_SUBJECTS]
    test_samples  = [s for s in data if s["subject"] not in TRAIN_SUBJECTS]

    print(f"\nCross-subject split:")
    print(f"  Train samples : {len(train_samples)}")
    print(f"  Test samples  : {len(test_samples)}")
    print(f"  Train ratio   : {100*len(train_samples)/len(data):.1f}%")

    # Build arrays for PyTorch

    # joints shape per sample: (100, 25, 3)
    # We reorder axes to (3, 100, 25) → (C, T, V) for PyTorch
    # C = 3 coordinates (x, y, z)
    # T = 100 time steps (frames)
    # V = 25 vertices (joints)

    def build_arrays(samples, split_name):
        N = len(samples)

        # Define arrays
        data_arr  = np.zeros((N, 3, 100, 25), dtype=np.float32)
        label_arr = np.zeros((N,),            dtype=np.int64)

        for i, sample in enumerate(samples):
            # joints: (100, 25, 3) -> transpose to (3, 100, 25)
            joints = sample["joints"]                    # (100, 25, 3)
            joints = joints.transpose(2, 0, 1)           # (3, 100, 25)
            data_arr[i]  = joints
            label_arr[i] = sample["action"] - 1          # 1-60 -> 0-59

        return data_arr, label_arr

    train_data,  train_labels = build_arrays(train_samples, "train")
    test_data,   test_labels  = build_arrays(test_samples,  "test")

    # Verify shapes
    print(f"\nArray shapes:")
    print(f"  train_data   : {train_data.shape}   : (N, C, T, V)")
    print(f"  train_labels : {train_labels.shape}")
    print(f"  test_data    : {test_data.shape}")
    print(f"  test_labels  : {test_labels.shape}")

    # Verify labels are in range 0-59
    print(f"\nLabel verification:")
    print(f"  Train — min label: {train_labels.min()}  max label: {train_labels.max()}")
    print(f"  Test  — min label: {test_labels.min()}   max label: {test_labels.max()}")
    print(f"  Unique train labels: {len(np.unique(train_labels))}  ← should be 60")
    print(f"  Unique test labels : {len(np.unique(test_labels))}   ← should be 60")

    # Class balance check
    train_counts = Counter(train_labels.tolist())
    test_counts  = Counter(test_labels.tolist())
    train_vals   = list(train_counts.values())
    test_vals    = list(test_counts.values())

    print(f"\nClass balance:")
    print(f"  Train — min: {min(train_vals)}  max: {max(train_vals)}  "
          f"mean: {np.mean(train_vals):.1f}")
    print(f"  Test  — min: {min(test_vals)}   max: {max(test_vals)}   "
          f"mean: {np.mean(test_vals):.1f}")
    
    # Save
    files = {
        "train_data.npy":   train_data,
        "train_labels.npy": train_labels,
        "test_data.npy":    test_data,
        "test_labels.npy":  test_labels,
    }

    for fname, arr in files.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        np.save(fpath, arr)


if __name__ == "__main__":
    main()