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

        # Translation
        # Get the spine base position for this frame
        spine_pos = normalized[f, SPINE_BASE, :].copy()

        # Subtract spine position from ALL joints
        # After this, joint 0 (spine base) = [0, 0, 0]
        normalized[f] -= spine_pos  # broadcasts across all 25 joints

        # Scaling
        # Compute torso length = distance from spine base to neck
        # After translation, spine base is at origin, so:
        # torso_length = ||neck_position||
        neck_pos     = normalized[f, NECK, :]
        torso_length = np.linalg.norm(neck_pos)  # Euclidean distance

        torso_lengths.append(torso_length)

        # Check for near-zero torso length to avoid extreme scaling
        if torso_length < MIN_TORSO_LENGTH:
            torso_length = MIN_TORSO_LENGTH

        # Divide ALL joints by torso length
        # After this, the torso length = 1.0 for every frame
        normalized[f] /= torso_length

    mean_torso = float(np.mean(torso_lengths))
    return normalized, mean_torso


def main():
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    results          = []
    torso_lengths    = []
    scale_warnings   = 0

    for sample in tqdm(data, desc="Normalizing"):
        joints = sample["joints"]   # (F, 25, 3)

        normalized, mean_torso = normalize_sequence(joints)

        if mean_torso < MIN_TORSO_LENGTH:
            scale_warnings += 1

        torso_lengths.append(mean_torso)

        results.append({
            "filename": sample["filename"],
            "action":   sample["action"],
            "subject":  sample["subject"],
            "joints":   normalized,   # (F, 25, 3), normalized
        })

    # Summary statistics
    torso_arr = np.array(torso_lengths)
    print(f"\nTorso length statistics (before scaling):")
    print(f"  Mean   : {torso_arr.mean():.4f} m")
    print(f"  Std    : {torso_arr.std():.4f} m")
    print(f"  Min    : {torso_arr.min():.4f} m")
    print(f"  Max    : {torso_arr.max():.4f} m")
    print(f"  Warnings (near-zero torso): {scale_warnings}")

    # Verify normalization worked
    sample_orig = data[0]
    sample_norm = results[0]

    print(f"\nVerification - {sample_orig['filename']}:")
    print(f"\n  Frame 0, Joint 0 (base of spine):")
    print(f"    Before : {sample_orig['joints'][0, SPINE_BASE, :]}")
    print(f"    After  : {sample_norm['joints'][0, SPINE_BASE, :]}")

    print(f"\n  Frame 0, Joint 2 (neck) - torso length check:")
    neck_after = sample_norm['joints'][0, NECK, :]
    torso_after = np.linalg.norm(neck_after)
    print(f"    Neck position after : {neck_after}")
    print(f"    ||neck||            : {torso_after:.4f}")

    print(f"\n  Frame 0, Joint 3 (head):")
    print(f"    Before : {sample_orig['joints'][0, 3, :]}")
    print(f"    After  : {sample_norm['joints'][0, 3, :]}")

    # Verify spine is at origin across multiple frames
    spine_positions = sample_norm['joints'][:, SPINE_BASE, :]
    max_spine_deviation = np.abs(spine_positions).max()
    print(f"\n  Max spine deviation from origin (all frames): {max_spine_deviation:.6f}")

    # Save
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(results, f, protocol=4)

if __name__ == "__main__":
    main()