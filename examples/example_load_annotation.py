"""
Example: How to read Xperience-10M data from annotation.hdf5.

This script demonstrates:
  1. Listing contents of annotation.hdf5
  2. Loading full annotation (calibration, SLAM poses, hand/body mocap, depth, IMU)
  3. Accessing calibration and camera extrinsics

Run from package root:
  .venv/bin/python examples/example_load_annotation.py --data_root data/xperience-10m-sample
"""

import sys
import argparse
from pathlib import Path

# Add package root to path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from data_loader import (
    load_from_annotation_hdf5,
    list_annotation_contents,
    get_T_camera_body,
)


def main():
    parser = argparse.ArgumentParser(description="Example: read Xperience-10M annotation and list/use data.")
    parser.add_argument("--data_root", type=str, required=True, help="Episode folder (contains annotation.hdf5)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    annotation_path = data_root / "annotation.hdf5"
    if not annotation_path.exists():
        print(f"Annotation not found: {annotation_path}")
        sys.exit(1)

    # 1) List contents (lightweight)
    print("--- annotation.hdf5 contents (top-level) ---")
    contents = list_annotation_contents(annotation_path)
    for name, shape in sorted(contents.items()):
        print(f"  {name}: {shape}")

    # 2) Load full annotation (all frames)
    ann = load_from_annotation_hdf5(str(annotation_path), 0, None)
    print("\n--- Loaded data summary ---")
    print(f"  Frames (img_names): {len(ann['img_names'])}")
    print(f"  R_c2w_all: {ann['R_c2w_all'].shape}")
    print(f"  t_c2w_all: {ann['t_c2w_all'].shape}")
    if ann.get("hand_left_joints") is not None:
        print(f"  Hand left joints: {ann['hand_left_joints'].shape}")
    if ann.get("hand_right_joints") is not None:
        print(f"  Hand right joints: {ann['hand_right_joints'].shape}")
    if ann.get("smplh_body_joints") is not None:
        print(f"  Full-body keypoints: {ann['smplh_body_joints'].shape}")
    if ann.get("contacts") is not None:
        print(f"  Contacts: {ann['contacts'].shape}")
    if ann.get("depth_loader") is not None:
        print(f"  Depth: lazy loader, {ann['depth_num_frames']} frames, range [{ann['depth_min']}, {ann['depth_max']}]")
    if ann.get("imu_accel_xyz") is not None:
        n_imu = ann["imu_accel_xyz"].shape[0]
        print(f"  IMU: {n_imu} samples (accel, gyro)")

    # 3) Calibration
    calib = ann["calib_data"]
    if calib:
        print("\n--- Calibration ---")
        if "cam01" in calib:
            K = calib["cam01"].get("K")
            if K is not None:
                print(f"  cam01.K shape: {getattr(K, 'shape', 'scalar')}")
        for cam_id in ["cam0", "cam1", "cam2", "cam3"]:
            T_c_b = get_T_camera_body(calib, cam_id)
            if T_c_b is not None:
                print(f"  {cam_id} T_c_b: available")

    print("\nDone. Use these arrays for your own processing or pass to visualization (see example_visualize_rrd.py).")


if __name__ == "__main__":
    main()
