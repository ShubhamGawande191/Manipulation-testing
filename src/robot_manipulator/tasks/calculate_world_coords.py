import json
import os

import numpy as np


def load_calibration_data():
    anchor_file = "anchor_config.json"
    intrinsics_file = "camera_intrinsics.json"

    if not os.path.exists(anchor_file) or not os.path.exists(intrinsics_file):
        print("Missing calibration files! Run find_world_tag.py and calibrate_intrinsics.py first.")
        return None, None

    with open(anchor_file) as f:
        anchor_data = json.load(f)

    with open(intrinsics_file) as f:
        intrinsics_data = json.load(f)

    return anchor_data, intrinsics_data


def get_3d_ray_from_pixel(u, v, camera_matrix):
    """
    Convert a 2D pixel (u,v) to a 3D ray in camera space defined by (X, Y, Z=1)
    """
    K = np.array(camera_matrix)
    K_inv = np.linalg.inv(K)

    # Pixel point in homogeneous coordinates
    p_px = np.array([[u], [v], [1.0]])

    # Ray in camera coordinates (at Z = 1)
    ray_cam = K_inv @ p_px
    return ray_cam.flatten()


def main():
    print("--- 3D World Coordinate Calculator ---")
    anchor_data, intrinsics_data = load_calibration_data()

    if not anchor_data or not intrinsics_data:
        return

    cam_matrix = intrinsics_data["camera_matrix"]
    dist_coeffs = np.array(intrinsics_data["dist_coeff"])

    anchor_x, anchor_y = anchor_data["pixel_center"]

    print("\nCalibration Data Loaded:")
    print(f"- Anchor Pixel Center: ({anchor_x}, {anchor_y})")

    # We will assume a fixed height/Z for our objects relative to the camera for now
    # The actual physical Z offset depends on where the arm is and object height.
    # We define Z_distance as the distance from the camera lens to the desk surface in mm.
    Z_distance = 150.0  # Example: 15 cm above desk

    # Calculate the ray for the anchor point (which is defined as 0,0 on the desk)
    anchor_ray = get_3d_ray_from_pixel(anchor_x, anchor_y, cam_matrix)

    # Using similar triangles: X_real = ray_x * Z_distance
    anchor_world_x = anchor_ray[0] * Z_distance
    anchor_world_y = anchor_ray[1] * Z_distance

    print("\nLet's test converting a new pixel point to physical coordinates!")
    print(f"Assuming camera is hovering exactly {Z_distance}mm above the desk.")

    try:
        u = input("\nEnter target pixel X (u) [default=300]: ").strip()
        v = input("Enter target pixel Y (v) [default=200]: ").strip()

        u = float(u) if u else 300.0
        v = float(v) if v else 200.0

        target_ray = get_3d_ray_from_pixel(u, v, cam_matrix)
        target_camera_x = target_ray[0] * Z_distance
        target_camera_y = target_ray[1] * Z_distance

        # Calculate offset from the anchor (0,0) point
        delta_x = target_camera_x - anchor_world_x
        delta_y = target_camera_y - anchor_world_y

        print(f"\nTarget at Pixel ({u}, {v})")
        print(f"-> Camera-Relative: X={target_camera_x:.2f}mm, Y={target_camera_y:.2f}mm")
        print(f"-> Absolute World Offset from Anchor: dX={delta_x:.2f}mm, dY={delta_y:.2f}mm")

    except ValueError:
        print("Please enter valid numbers.")


if __name__ == "__main__":
    main()
