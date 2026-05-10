import os
import pickle
import numpy as np
from tqdm import tqdm

# Configuration files

INPUT_FILE  = "./preprocessed/step1_parsed.pkl"
OUTPUT_FILE = "./preprocessed/step2_filled.pkl"

# This function will be used to fill missing frames by copying the last valid frame forward
def forward_fill(joints, missing_frames):

    # Base case
    if not missing_frames:
        return joints, False

    filled = joints.copy()
    F = filled.shape[0]

    # Build a boolean mask: True = this frame is missing
    missing_set = set(missing_frames)

    # Fill forward from first valid frame
    first_valid = None
    for i in range(F):
        if i not in missing_set:
            first_valid = i
            break

    if first_valid is None:
        # No valid frames at all
        return None, False

    if first_valid > 0:
        # Fill frames 0 to first_valid-1 with the first valid frame
        for i in range(first_valid):
            filled[i] = filled[first_valid]

    # Forward-fill remaining missing frames 
    for i in range(1, F):
        if i in missing_set:
            filled[i] = filled[i - 1]

    return filled, True


def main():
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    results = []
    fixed_count = 0
    skipped = 0

    for sample in tqdm(data, desc="Forward-filling"):
        joints = sample["joints"]
        missing_frames = sample["missing_frames"]

        filled, was_fixed = forward_fill(joints, missing_frames)

        if filled is None:
            # Completely empty file - skip it
            print(f"All frames missing: {sample['filename']}")
            skipped += 1
            continue

        if was_fixed:
            fixed_count += 1

        # Store the fixed version, drop the missing_frames list
        results.append({
            "filename": sample["filename"],
            "action":   sample["action"],
            "subject":  sample["subject"],
            "joints":   filled,
        })

    # Summary
    print(f"\nResults:")
    print(f"  Input samples    : {len(data)}")
    print(f"  Samples fixed    : {fixed_count}")
    print(f"  Samples skipped  : {skipped}  (all frames missing)")
    print(f"  Output samples   : {len(results)}")

    # Save
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(results, f, protocol=4)

if __name__ == "__main__":
    main()