import os
import pickle
import numpy as np
from tqdm import tqdm

INPUT_FILE  = "./preprocessed/step2_filled.pkl"
OUTPUT_FILE = "./preprocessed/step3_normalized.pkl"

# Joint indices
SPINE_BASE = 0
NECK = 2

# Minimum torso length to avoid division by near-zero
MIN_TORSO_LENGTH = 0.1  # meters

# This function normalizes a sequence of joints by translating and scaling
def normalize_sequence(joints):

    F = joints.shape[0]
    normalized = joints.copy()

    torso_lengths = []

    for f in range(F):

        # Get the spine base position for this frame
        spine_pos = normalized[f, SPINE_BASE, :].copy()

        # Subtract spine position from all joints
        normalized[f] -= spine_pos

        # Compute torso length - distance from spine base to neck (euclidean)
        neck_pos     = normalized[f, NECK, :]
        torso_length = np.linalg.norm(neck_pos)

        torso_lengths.append(torso_length)

        # Check for near-zero torso length to avoid extreme scaling
        if torso_length < MIN_TORSO_LENGTH:
            torso_length = MIN_TORSO_LENGTH

        # Divide all joints by torso length
        normalized[f] /= torso_length

    mean_torso = float(np.mean(torso_lengths))
    return normalized, mean_torso


def main():
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    results = []
    torso_lengths = []
    scale_warnings = 0

    for sample in tqdm(data, desc="Normalizing"):
        joints = sample["joints"]

        normalized, mean_torso = normalize_sequence(joints)

        if mean_torso < MIN_TORSO_LENGTH:
            scale_warnings += 1

        torso_lengths.append(mean_torso)

        results.append({
            "filename": sample["filename"],
            "action":   sample["action"],
            "subject":  sample["subject"],
            "joints":   normalized,
        })


    # Save
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(results, f, protocol=4)

if __name__ == "__main__":
    main()