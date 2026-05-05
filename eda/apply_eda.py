"""
NTU RGB+D 60 — Exploratory Data Analysis (EDA)
================================================
Parses a sample of .skeleton files and produces a full EDA report:

  1. Dataset overview (file counts, class distribution)
  2. Sequence length analysis (frames per sample)
  3. Subject & camera view distribution
  4. Skeleton quality analysis (missing/zero bodies)
  5. Joint position statistics (mean, std per joint)
  6. Motion intensity analysis (mean displacement per class)
  7. Cross-subject split preview

All plots are saved to ./eda_output/

Usage:
    pip install numpy matplotlib seaborn pandas tqdm
    python eda_ntu60.py

Configuration:
    SKELETON_DIR  — folder containing all .skeleton files
    SAMPLE_SIZE   — how many files to parse for deep analysis
                    (parsing all 56k is slow; 3000 gives solid stats)
"""

import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

# 

SKELETON_DIR = "./dataset/nturgb+d_skeletons"
OUTPUT_DIR   = "./eda_output"
SAMPLE_SIZE  = 10000
RANDOM_SEED  = 42

# Official list of missing/corrupted files from NTU RGB+D 60 dataset

MISSING_FILES = set("""
S001C002P005R002A008
S001C002P006R001A008
S001C003P002R001A055
S001C003P002R002A012
S001C003P005R002A004
S001C003P005R002A005
S001C003P005R002A006
S001C003P006R002A008
S002C002P011R002A030
S002C003P008R001A020
S002C003P010R002A010
S002C003P011R002A007
S002C003P011R002A011
S002C003P014R002A007
S003C001P019R001A055
S003C002P002R002A055
S003C002P018R002A055
S003C003P002R001A055
S003C003P016R001A055
S003C003P018R002A024
S004C002P003R001A013
S004C002P008R001A009
S004C002P020R001A003
S004C002P020R001A004
S004C002P020R001A012
S004C002P020R001A020
S004C002P020R001A021
S004C002P020R001A036
S005C002P004R001A001
S005C002P004R001A003
S005C002P010R001A016
S005C002P010R001A017
S005C002P010R001A048
S005C002P010R001A049
S005C002P016R001A009
S005C002P016R001A010
S005C002P018R001A003
S005C002P018R001A028
S005C002P018R001A029
S005C003P016R002A009
S005C003P018R002A013
S005C003P021R002A057
S006C001P001R002A055
S006C002P007R001A005
S006C002P007R001A006
S006C002P016R001A043
S006C002P016R001A051
S006C002P016R001A052
S006C002P022R001A012
S006C002P023R001A020
S006C002P023R001A021
S006C002P023R001A022
S006C002P023R001A023
S006C002P024R001A018
S006C002P024R001A019
S006C003P001R002A013
S006C003P007R002A009
S006C003P007R002A010
S006C003P007R002A025
S006C003P016R001A060
S006C003P017R001A055
S006C003P017R002A013
S006C003P017R002A014
S006C003P017R002A015
S006C003P022R002A013
S007C001P018R002A050
S007C001P025R002A051
S007C001P028R001A050
S007C001P028R001A051
S007C001P028R001A052
S007C002P008R002A008
S007C002P015R002A055
S007C002P026R001A008
S007C002P026R001A009
S007C002P026R001A010
S007C002P026R001A011
S007C002P026R001A012
S007C002P026R001A050
S007C002P027R001A011
S007C002P027R001A013
S007C002P028R002A055
S007C003P007R001A002
S007C003P007R001A004
S007C003P019R001A060
S007C003P027R002A001
S007C003P027R002A002
S007C003P027R002A003
S007C003P027R002A004
S007C003P027R002A005
S007C003P027R002A006
S007C003P027R002A007
S007C003P027R002A008
S007C003P027R002A009
S007C003P027R002A010
S007C003P027R002A011
S007C003P027R002A012
S007C003P027R002A013
S008C002P001R001A009
S008C002P001R001A010
S008C002P001R001A014
S008C002P001R001A015
S008C002P001R001A016
S008C002P001R001A018
S008C002P001R001A019
S008C002P008R002A059
S008C002P025R001A060
S008C002P029R001A004
S008C002P031R001A005
S008C002P031R001A006
S008C002P032R001A018
S008C002P034R001A018
S008C002P034R001A019
S008C002P035R001A059
S008C002P035R002A002
S008C002P035R002A005
S008C003P007R001A009
S008C003P007R001A016
S008C003P007R001A017
S008C003P007R001A018
S008C003P007R001A019
S008C003P007R001A020
S008C003P007R001A021
S008C003P007R001A022
S008C003P007R001A023
S008C003P007R001A025
S008C003P007R001A026
S008C003P007R001A028
S008C003P007R001A029
S008C003P007R002A003
S008C003P008R002A050
S008C003P025R002A002
S008C003P025R002A011
S008C003P025R002A012
S008C003P025R002A016
S008C003P025R002A020
S008C003P025R002A022
S008C003P025R002A023
S008C003P025R002A030
S008C003P025R002A031
S008C003P025R002A032
S008C003P025R002A033
S008C003P025R002A049
S008C003P025R002A060
S008C003P031R001A001
S008C003P031R002A004
S008C003P031R002A014
S008C003P031R002A015
S008C003P031R002A016
S008C003P031R002A017
S008C003P032R002A013
S008C003P033R002A001
S008C003P033R002A011
S008C003P033R002A012
S008C003P034R002A001
S008C003P034R002A012
S008C003P034R002A022
S008C003P034R002A023
S008C003P034R002A024
S008C003P034R002A044
S008C003P034R002A045
S008C003P035R002A016
S008C003P035R002A017
S008C003P035R002A018
S008C003P035R002A019
S008C003P035R002A020
S008C003P035R002A021
S009C002P007R001A001
S009C002P007R001A003
S009C002P007R001A014
S009C002P008R001A014
S009C002P015R002A050
S009C002P016R001A002
S009C002P017R001A028
S009C002P017R001A029
S009C003P017R002A030
S009C003P025R002A054
S010C001P007R002A020
S010C002P016R002A055
S010C002P017R001A005
S010C002P017R001A018
S010C002P017R001A019
S010C002P019R001A001
S010C002P025R001A012
S010C003P007R002A043
S010C003P008R002A003
S010C003P016R001A055
S010C003P017R002A055
S011C001P002R001A008
S011C001P018R002A050
S011C002P008R002A059
S011C002P016R002A055
S011C002P017R001A020
S011C002P017R001A021
S011C002P018R002A055
S011C002P027R001A009
S011C002P027R001A010
S011C002P027R001A037
S011C003P001R001A055
S011C003P002R001A055
S011C003P008R002A012
S011C003P015R001A055
S011C003P016R001A055
S011C003P019R001A055
S011C003P025R001A055
S011C003P028R002A055
S012C001P019R001A060
S012C001P019R002A060
S012C002P015R001A055
S012C002P017R002A012
S012C002P025R001A060
S012C003P008R001A057
S012C003P015R001A055
S012C003P015R002A055
S012C003P016R001A055
S012C003P017R002A055
S012C003P018R001A055
S012C003P018R001A057
S012C003P019R002A011
S012C003P019R002A012
S012C003P025R001A055
S012C003P027R001A055
S012C003P027R002A009
S012C003P028R001A035
S012C003P028R002A055
S013C001P015R001A054
S013C001P017R002A054
S013C001P018R001A016
S013C001P028R001A040
S013C002P015R001A054
S013C002P017R002A054
S013C002P028R001A040
S013C003P008R002A059
S013C003P015R001A054
S013C003P017R002A054
S013C003P025R002A022
S013C003P027R001A055
S013C003P028R001A040
S014C001P027R002A040
S014C002P015R001A003
S014C002P019R001A029
S014C002P025R002A059
S014C002P027R002A040
S014C002P039R001A050
S014C003P007R002A059
S014C003P015R002A055
S014C003P019R002A055
S014C003P025R001A048
S014C003P027R002A040
S015C001P008R002A040
S015C001P016R001A055
S015C001P017R001A055
S015C001P017R002A055
S015C002P007R001A059
S015C002P008R001A003
S015C002P008R001A004
S015C002P008R002A040
S015C002P015R001A002
S015C002P016R001A001
S015C002P016R002A055
S015C003P008R002A007
S015C003P008R002A011
S015C003P008R002A012
S015C003P008R002A028
S015C003P008R002A040
S015C003P025R002A012
S015C003P025R002A017
S015C003P025R002A020
S015C003P025R002A021
S015C003P025R002A030
S015C003P025R002A033
S015C003P025R002A034
S015C003P025R002A036
S015C003P025R002A037
S015C003P025R002A044
S016C001P019R002A040
S016C001P025R001A011
S016C001P025R001A012
S016C001P025R001A060
S016C001P040R001A055
S016C001P040R002A055
S016C002P008R001A011
S016C002P019R002A040
S016C002P025R002A012
S016C003P008R001A011
S016C003P008R002A002
S016C003P008R002A003
S016C003P008R002A004
S016C003P008R002A006
S016C003P008R002A009
S016C003P019R002A040
S016C003P039R002A016
S017C001P016R002A031
S017C002P007R001A013
S017C002P008R001A009
S017C002P015R001A042
S017C002P016R002A031
S017C002P016R002A055
S017C003P007R002A013
S017C003P008R001A059
S017C003P016R002A031
S017C003P017R001A055
S017C003P020R001A059
""".strip().split('\n'))


def is_missing(fname):
    """Return True if this file is on the official missing list."""
    # Strip extension and match bare ID (case-insensitive)
    return fname.split('.')[0].upper() in {m.upper() for m in MISSING_FILES}

# labels for NTU60 classes

NTU60_CLASSES = {
    1:"drink water", 2:"eat meal/snack", 3:"brushing teeth",
    4:"brushing hair", 5:"drop", 6:"pick up", 7:"throw",
    8:"sit down", 9:"stand up", 10:"clapping", 11:"reading",
    12:"writing", 13:"tear up paper", 14:"put on jacket",
    15:"take off jacket", 16:"put on a shoe", 17:"take off a shoe",
    18:"put on glasses", 19:"take off glasses", 20:"put on a hat/cap",
    21:"take off a hat/cap", 22:"cheer up", 23:"hand waving",
    24:"kicking something", 25:"reach into pocket", 26:"hopping",
    27:"jump up", 28:"phone call", 29:"play with phone/tablet",
    30:"type on a keyboard", 31:"point to something", 32:"taking a selfie",
    33:"check time (from watch)", 34:"rub two hands", 35:"nod head/bow",
    36:"shake head", 37:"wipe face", 38:"salute",
    39:"put palms together", 40:"cross hands in front", 41:"sneeze/cough",
    42:"staggering", 43:"falling down", 44:"headache", 45:"chest pain",
    46:"back pain", 47:"neck pain", 48:"nausea/vomiting", 49:"fan self",
    50:"punch/slap", 51:"kicking", 52:"pushing", 53:"pat on back",
    54:"point finger", 55:"hugging", 56:"giving object",
    57:"touch pocket", 58:"shaking hands", 59:"walking towards",
    60:"walking apart",
}

# cross-train subjects from NTU RGB+D 60 dataset
TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 12, 13, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 36, 38, 39, 40}

# NTU 25 joint names
JOINT_NAMES = [
    "base of spine", "mid spine", "neck", "head",
    "left shoulder", "left elbow", "left wrist", "left hand",
    "right shoulder", "right elbow", "right wrist", "right hand",
    "left hip", "left knee", "left ankle", "left foot",
    "right hip", "right knee", "right ankle", "right foot",
    "spine", "left hand tip", "left thumb",
    "right hand tip", "right thumb",
]

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def parse_filename(fname):
    """Extract metadata from NTU filename. Returns dict or None."""
    m = re.match(
        r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})',
        fname, re.IGNORECASE
    )
    if not m:
        return None
    return {
        "setup":   int(m.group(1)),
        "camera":  int(m.group(2)),
        "subject": int(m.group(3)),
        "rep":     int(m.group(4)),
        "action":  int(m.group(5)),
    }


def parse_skeleton_file(filepath):
    """
    Parse a single .skeleton file.
    Returns dict with:
        n_frames, n_bodies, joints (np array F x 25 x 3), has_missing
    Returns None on failure.
    """
    try:
        with open(filepath, "r") as f:
            lines = f.read().splitlines()

        idx = 0
        n_frames = int(lines[idx]); idx += 1
        all_joints = []
        has_missing = False

        for _ in range(n_frames):
            n_bodies = int(lines[idx]); idx += 1
            if n_bodies == 0:
                has_missing = True
                all_joints.append(np.zeros((25, 3)))
                continue

            # Read only the first body
            idx += 1  # body info line
            n_joints = int(lines[idx]); idx += 1
            frame_joints = np.zeros((25, 3))
            for j in range(min(n_joints, 25)):
                vals = lines[idx].split(); idx += 1
                frame_joints[j] = [float(vals[0]),
                                   float(vals[1]),
                                   float(vals[2])]
            # skip remaining joints if any
            if n_joints > 25:
                idx += (n_joints - 25)
            # skip remaining bodies if any
            for _ in range(n_bodies - 1):
                idx += 1  # body info
                nj = int(lines[idx]); idx += 1
                idx += nj

            all_joints.append(frame_joints)

        joints = np.stack(all_joints)  # (F, 25, 3)
        return {
            "n_frames":   n_frames,
            "n_bodies":   n_bodies,
            "joints":     joints,
            "has_missing": has_missing,
        }
    except Exception:
        return None


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# ─── ANALYSIS SECTIONS ───────────────────────────────────────────────────────

def section_1_overview(meta_df):
    print("\n[1/7] Dataset overview")
    print(f"  Clean files        : {len(meta_df)}  (after excluding 302 corrupted)")
    print(f"  Unique actions     : {meta_df['action'].nunique()}")
    print(f"  Unique subjects    : {meta_df['subject'].nunique()}")
    print(f"  Unique cameras     : {meta_df['camera'].nunique()}")
    print(f"  Unique setups      : {meta_df['setup'].nunique()}")
    print(f"  Files per class    : {len(meta_df) // meta_df['action'].nunique()} (avg)")

    # Class distribution bar chart
    counts = meta_df["action"].value_counts().sort_index()
    labels = [f"A{i:03d}" for i in counts.index]

    fig, ax = plt.subplots(figsize=(20, 5))
    bars = ax.bar(range(len(counts)), counts.values, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("Action Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution — NTU RGB+D 60 (302 corrupted files excluded)")
    ax.axhline(counts.mean(), color="red", linestyle="--", linewidth=1.2, label=f"Mean = {counts.mean():.0f}")
    ax.legend()
    fig.tight_layout()
    save(fig, "1_class_distribution.png")


def section_2_metadata(meta_df):
    print("\n[2/7] Subject & camera distribution")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subject distribution
    subj_counts = meta_df["subject"].value_counts().sort_index()
    axes[0].bar(subj_counts.index, subj_counts.values, color="#55A868", width=0.8)
    axes[0].set_title("Samples per Subject")
    axes[0].set_xlabel("Subject ID")
    axes[0].set_ylabel("Count")

    # Camera distribution
    cam_counts = meta_df["camera"].value_counts().sort_index()
    axes[1].bar([f"C{c:03d}" for c in cam_counts.index], cam_counts.values,
                color="#C44E52", width=0.6)
    axes[1].set_title("Samples per Camera View")
    axes[1].set_xlabel("Camera")
    axes[1].set_ylabel("Count")

    # Train/test split preview (cross-subject)
    meta_df["split"] = meta_df["subject"].apply(
        lambda s: "Train" if s in TRAIN_SUBJECTS else "Test"
    )
    split_counts = meta_df["split"].value_counts()
    axes[2].pie(split_counts.values, labels=split_counts.index,
                autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"],
                startangle=90, textprops={"fontsize": 13})
    axes[2].set_title("Cross-Subject Split")

    fig.tight_layout()
    save(fig, "2_metadata_distribution.png")


def section_3_sequence_length(parsed):
    print("\n[3/7] Sequence length analysis")
    lengths = [p["n_frames"] for p in parsed.values()]
    lengths = np.array(lengths)

    print(f"  Min frames  : {lengths.min()}")
    print(f"  Max frames  : {lengths.max()}")
    print(f"  Mean frames : {lengths.mean():.1f}")
    print(f"  Std frames  : {lengths.std():.1f}")
    print(f"  Median      : {np.median(lengths):.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(lengths, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
    axes[0].axvline(lengths.mean(), color="red", linestyle="--", label=f"Mean={lengths.mean():.0f}")
    axes[0].axvline(np.median(lengths), color="orange", linestyle="--", label=f"Median={np.median(lengths):.0f}")
    axes[0].set_title("Sequence Length Distribution")
    axes[0].set_xlabel("Number of Frames")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].boxplot(lengths, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    axes[1].set_title("Sequence Length Box Plot")
    axes[1].set_ylabel("Number of Frames")
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["All samples"])

    fig.tight_layout()
    save(fig, "3_sequence_length.png")


def section_4_quality(parsed, sampled_files):
    print("\n[4/7] Skeleton quality analysis")
    missing_count = sum(1 for p in parsed.values() if p["has_missing"])
    none_count    = sum(1 for p in parsed.values() if p is None)

    print(f"  Files with missing body frames : {missing_count} ({100*missing_count/len(parsed):.1f}%)")
    print(f"  Files failed to parse          : {none_count}")

    labels = ["Valid", "Has Missing Frames", "Parse Failed"]
    sizes  = [len(parsed) - missing_count, missing_count, none_count]
    colors = ["#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        [s for s in sizes if s > 0],
        labels=[l for l, s in zip(labels, sizes) if s > 0],
        autopct="%1.1f%%", colors=[c for c, s in zip(colors, sizes) if s > 0],
        startangle=90, textprops={"fontsize": 12}
    )
    ax.set_title("Skeleton File Quality")
    fig.tight_layout()
    save(fig, "4_skeleton_quality.png")


def section_5_joint_stats(parsed):
    print("\n[5/7] Joint position statistics")

    # Collect mean position of each joint across all sampled files
    # Use first frame only for speed
    joint_means = []
    for p in parsed.values():
        if p is not None and p["joints"].shape[0] > 0:
            joint_means.append(p["joints"][0])  # first frame, shape (25,3)

    joint_means = np.stack(joint_means)  # (N, 25, 3)
    mean_pos = joint_means.mean(axis=0)  # (25, 3)
    std_pos  = joint_means.std(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    coords = ["X", "Y", "Z"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for i, (coord, color) in enumerate(zip(coords, colors)):
        axes[i].barh(range(25), mean_pos[:, i], xerr=std_pos[:, i],
                     color=color, alpha=0.7, ecolor="black", capsize=3)
        axes[i].set_yticks(range(25))
        axes[i].set_yticklabels(JOINT_NAMES, fontsize=8)
        axes[i].set_title(f"Joint Mean {coord} Position (± std)")
        axes[i].set_xlabel(f"{coord} coordinate (meters)")

    fig.tight_layout()
    save(fig, "5_joint_position_stats.png")

    # 2D skeleton mean pose (X vs Y, frontal view)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.scatter(mean_pos[:, 0], mean_pos[:, 1], s=80, zorder=5, color="#4C72B0")
    for j, name in enumerate(JOINT_NAMES):
        ax.annotate(name, (mean_pos[j, 0], mean_pos[j, 1]),
                    fontsize=6, textcoords="offset points", xytext=(4, 2))
    ax.set_title("Mean Skeleton Pose (Frontal View, X vs Y)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "5b_mean_skeleton_pose.png")


def section_6_motion_intensity(parsed, sampled_files):
    print("\n[6/7] Motion intensity per class")

    class_motion = defaultdict(list)
    for fname, p in parsed.items():
        if p is None or p["joints"].shape[0] < 2:
            continue
        meta = parse_filename(fname)
        if meta is None:
            continue
        # Mean frame-to-frame displacement across all joints
        diffs = np.diff(p["joints"], axis=0)          # (F-1, 25, 3)
        motion = np.linalg.norm(diffs, axis=-1).mean() # scalar
        class_motion[meta["action"]].append(motion)

    class_ids   = sorted(class_motion.keys())
    mean_motion = [np.mean(class_motion[c]) for c in class_ids]
    std_motion  = [np.std(class_motion[c])  for c in class_ids]
    labels      = [f"A{c:03d}" for c in class_ids]

    # Sort by motion intensity
    sorted_idx  = np.argsort(mean_motion)[::-1]

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(range(len(class_ids)),
           [mean_motion[i] for i in sorted_idx],
           yerr=[std_motion[i] for i in sorted_idx],
           color="#4C72B0", ecolor="black", capsize=2,
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels([labels[i] for i in sorted_idx], rotation=90, fontsize=7)
    ax.set_title("Mean Motion Intensity per Class (sorted)")
    ax.set_xlabel("Action Class")
    ax.set_ylabel("Mean Joint Displacement (m/frame)")
    fig.tight_layout()
    save(fig, "6_motion_intensity.png")


def section_7_sequence_length_by_class(parsed, sampled_files):
    print("\n[7/7] Sequence length by class")

    class_lengths = defaultdict(list)
    for fname, p in parsed.items():
        if p is None:
            continue
        meta = parse_filename(fname)
        if meta is None:
            continue
        class_lengths[meta["action"]].append(p["n_frames"])

    class_ids = sorted(class_lengths.keys())
    data      = [class_lengths[c] for c in class_ids]
    labels    = [f"A{c:03d}" for c in class_ids]

    fig, ax = plt.subplots(figsize=(22, 6))
    ax.boxplot(data, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.5),
               medianprops=dict(color="red", linewidth=1.5),
               flierprops=dict(marker=".", markersize=2, alpha=0.3),
               whiskerprops=dict(linewidth=0.8),
               capprops=dict(linewidth=0.8))
    ax.set_xticks(range(1, len(class_ids) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title("Sequence Length Distribution per Class")
    ax.set_xlabel("Action Class")
    ax.set_ylabel("Number of Frames")
    fig.tight_layout()
    save(fig, "7_sequence_length_by_class.png")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    random.seed(RANDOM_SEED)

    # ── Step 1: scan all files and build metadata DataFrame ──
    print(f"\n[INIT] Scanning {SKELETON_DIR} ...")
    if not os.path.isdir(SKELETON_DIR):
        print(f"[ERROR] Directory not found: {SKELETON_DIR}")
        print("        Please update SKELETON_DIR at the top of the script.")
        return

    all_files_raw = [f for f in os.listdir(SKELETON_DIR) if f.lower().endswith(".skeleton")]
    print(f"       Found {len(all_files_raw)} .skeleton files total")

    # ── Filter out the 302 officially corrupted files ──
    all_files  = [f for f in all_files_raw if not is_missing(f)]
    n_excluded = len(all_files_raw) - len(all_files)
    print(f"       Excluded {n_excluded} officially corrupted files")
    print(f"       Remaining: {len(all_files)} clean files")

    if n_excluded == 0:
        print("       [WARN] No missing files were matched. "
              "Check that filenames follow the SxxxCxxxPxxxRxxxAxxx pattern.")

    records = []
    for fname in all_files:
        meta = parse_filename(fname)
        if meta:
            meta["filename"] = fname
            records.append(meta)

    meta_df = pd.DataFrame(records)
    print(f"       Parsed metadata for {len(meta_df)} clean files")

    # ── Step 2: sample files for deep analysis ──
    sample_n    = min(SAMPLE_SIZE, len(all_files))
    sampled     = random.sample(all_files, sample_n)
    print(f"\n[INFO] Deep-parsing {sample_n} files for content analysis ...")
    print(f"       (increase SAMPLE_SIZE for more accurate stats)\n")

    parsed = {}
    for fname in tqdm(sampled, desc="Parsing skeletons"):
        fpath = os.path.join(SKELETON_DIR, fname)
        parsed[fname] = parse_skeleton_file(fpath)

    # ── Run all sections ──
    section_1_overview(meta_df)
    section_2_metadata(meta_df)
    section_3_sequence_length(parsed)
    section_4_quality(parsed, sampled)
    section_5_joint_stats(parsed)
    section_6_motion_intensity(parsed, sampled)
    section_7_sequence_length_by_class(parsed, sampled)

    print(f"\n{'='*55}")
    print(f"  EDA complete. All plots saved to: {OUTPUT_DIR}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()