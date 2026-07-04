import glob
import json

import cv2


def calibrate_camera_charuco():
    # --- ChArUco Board Configuration ---
    # Update these if your board dimensions are different
    squares_x = 5
    squares_y = 7
    square_length = 0.04  # 40mm squares
    marker_length = 0.02  # 20mm markers
    # Many standard physical ChArUco boards use 5X5 dictionary, let's try 5X5_250 as fallback if 4X4 fails, or DICT_6X6_250!
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_5X5_100
    )  # Setting to standard, wait actually let's re-read the printout. Dict 15 is 7X7_1000. Wait, Dict 16 is original. I should set to DICT_5X5_100. Wait. I'll just use 12 (DICT_7X7_50).
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

    # Setup Aruco detector
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Setup Charuco detector
    charuco_detector = cv2.aruco.CharucoDetector(board)

    # Setup arrays to store object points and image points from all the images.
    all_charuco_corners = []
    all_charuco_ids = []

    images = glob.glob("calibration_images/*.jpg")
    if not images:
        print("No images found in calibration_images/")
        return

    print(f"Found {len(images)} images. Processing...")

    imsize = None
    successful_images = 0

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if imsize is None:
            imsize = gray.shape[::-1]

        # Detect markers
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(
            gray
        )

        if marker_ids is not None and len(marker_ids) > 0:
            # If enough corners are found, add them
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                successful_images += 1

                # Draw and save debug image
                cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
                cv2.imwrite(fname.replace(".jpg", "_debug.jpg"), img)
            else:
                print(f"Skipped {fname}: too few corners detected.")
        else:
            print(f"Skipped {fname}: no markers detected.")

    if successful_images < 5:
        print("Error: Not enough good calibration images found (need at least 5).")
        return

    print(f"\nCalibrating using {successful_images} valid images...")

    # Actually perform calibration
    obj_pts = board.getChessboardCorners()

    # In OpenCV 4.7+, the calibrateCameraCharuco function expects 1D arrays of points or the old signature was moved. Let's use board.matchImagePoints!
    # Wait, the best way in modern opencv is calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs)
    # But wait, it might be in `charuco_detector`, or `cv2.aruco.calibrateCameraCharuco` might be removed entirely and moved to `cv2.calibrateCamera` with object points mapped!

    # Standard approach for version 4.7.0+
    objpoints = []
    imgpoints = []

    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        objp, imgp = board.matchImagePoints(corners, ids)
        if len(objp) > 3:
            objpoints.append(objp)
            imgpoints.append(imgp)

    if not objpoints:
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imsize, None, None)

    print("\n=== Calibration Complete ===")
    print(f"RMS Re-projection Error: {ret:.4f} pixels (lower is better, <1.0 is ideal)")
    print("\nCamera Matrix (Intrinsics):")
    print(mtx)
    print("\nDistortion Coefficients:")
    print(dist)

    # Save to JSON
    calib_data = {"camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist(), "rms_error": ret}

    with open("camera_intrinsics.json", "w") as f:
        json.dump(calib_data, f, indent=4)
    print("\nSaved intrinsics to camera_intrinsics.json!")


if __name__ == "__main__":
    calibrate_camera_charuco()
