import os
import pickle
import numpy as np
from collections import Counter

INPUT_FILE   = "./preprocessed/step3_normalized.pkl"
OUTPUT_FILE  = "./preprocessed/step3_cleaned.pkl"

# Threshold: joints should never be more than 20x the torso length
MAX_JOINT_VALUE = 20.0


def main():
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    kept = []
    removed = []

    for sample in data:
        max_val = np.abs(sample["joints"]).max()
        if max_val > MAX_JOINT_VALUE:
            removed.append(sample)
        else:
            kept.append(sample)

    # Show which classes were affected
    removed_classes = Counter(s["action"] for s in removed)
    print(f"\nRemoved samples by class:")
    for action_id, count in sorted(removed_classes.items()):
        print(f"  A{action_id:03d} : {count} samples removed")

    # Show remaining class distribution to confirm balance is preserved
    kept_classes = Counter(s["action"] for s in kept)
    counts = list(kept_classes.values())
    print(f"\nRemaining class distribution:")
    print(f"  Min samples per class : {min(counts)}")
    print(f"  Max samples per class : {max(counts)}")
    print(f"  Mean : {np.mean(counts):.1f}")
    print(f"  Classes affected : {len(removed_classes)}")

    # Save
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(kept, f, protocol=4)

if __name__ == "__main__":
    main()