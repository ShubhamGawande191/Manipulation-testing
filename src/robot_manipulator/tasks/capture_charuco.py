import os

import cv2


def capture_calibration_images():
    output_dir = "calibration_images"
    os.makedirs(output_dir, exist_ok=True)

    print("Opening wrist camera...")
    cap = cv2.VideoCapture(0)  # Open wrist camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\n===============================")
    print("  ChArUco Calibration Capture ")
    print("===============================\n")
    print("Instructions:")
    print("1. Hold the ChArUco board steady in front of the wrist camera.")
    print("2. Move it around slowly: tilt it, rotate it, move it close & far.")
    print("3. Ensure the whole board is visible with good lighting.")
    print("4. Press 'c' to capture an image.")
    print("5. Press 'q' when finished (aim for 15-20 images).\n")

    img_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            cv2.putText(
                display,
                f"Captured: {img_count}/15+",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Calibration Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                filename = os.path.join(output_dir, f"calib_{img_count:02d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Captured: {filename}")
                img_count += 1
            elif key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. Captured {img_count} images to '{output_dir}/'")


if __name__ == "__main__":
    capture_calibration_images()
