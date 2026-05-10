import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter


INPUT_FILE  = "./preprocessed/step3_cleaned.pkl"
OUTPUT_FILE = "./preprocessed/step4_padded.pkl"

TARGET_LEN  = 100

def pad_or_truncate(joints, target_len):
    # Resize a sequence of joints to the target length by padding with zeros or truncating
    F = joints.shape[0]

    if F == target_len:
        return joints.copy(), "exact"

    elif F < target_len:
        # Pad with zeros at the end
        pad_width = target_len - F
        padding   = np.zeros((pad_width, 25, 3), dtype=np.float32)
        resized   = np.concatenate([joints, padding], axis=0)
        return resized, "padded"

    else:
        # Keep the first target_len frames
        resized = joints[:target_len].copy()
        return resized, "truncated"



def main():
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    results    = []
    status_counts = Counter()
    original_lengths = []

    for sample in tqdm(data, desc="Padding/Truncating"):
        joints = sample["joints"]
        original_lengths.append(joints.shape[0])

        resized, status = pad_or_truncate(joints, TARGET_LEN)
        status_counts[status] += 1

        # Verify shape is correct
        assert resized.shape == (TARGET_LEN, 25, 3), \
            f"Unexpected shape {resized.shape} for {sample['filename']}"

        results.append({
            "filename": sample["filename"],
            "action":   sample["action"],
            "subject":  sample["subject"],
            "joints":   resized,
        })

    # Summary

    print(f"\nResize operations:")
    print(f"  Exact      (= 100 frames) : {status_counts['exact']}")
    print(f"  Padded     (< 100 frames) : {status_counts['padded']}")
    print(f"  Truncated  (> 100 frames) : {status_counts['truncated']}")

    # Verify all shapes are correct
    shapes = set(r["joints"].shape for r in results)
    print(f"\nShape verification:")
    print(f"  Unique shapes in output : {shapes}")

    # Save
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(results, f, protocol=4)


if __name__ == "__main__":
    main()