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
    def build_arrays(samples, split_name):
        N = len(samples)

        # Define arrays
        data_arr  = np.zeros((N, 3, 100, 25), dtype=np.float32)
        label_arr = np.zeros((N,),            dtype=np.int64)

        for i, sample in enumerate(samples):
            # joints: (100, 25, 3) - transpose to (3, 100, 25)
            joints = sample["joints"]
            joints = joints.transpose(2, 0, 1)
            data_arr[i]  = joints
            # Convert action label from 1-60 to 0-59 for zero-based indexing
            label_arr[i] = sample["action"] - 1

        return data_arr, label_arr

    train_data,  train_labels = build_arrays(train_samples, "train")
    test_data,   test_labels  = build_arrays(test_samples,  "test")
    
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