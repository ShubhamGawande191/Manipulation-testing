import sys

import cv2
from pupil_apriltags import Detector


def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print("Initializing AprilTag Detector for 36h11 family...")
    at_detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    print(f"Opening camera index: {cam_index}")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Looking for AprilTags... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect
            tags = at_detector.detect(gray)

            for tag in tags:
                # Draw the bounding box
                for idx in range(len(tag.corners)):
                    cv2.line(
                        frame,
                        tuple(tag.corners[idx - 1, :].astype(int)),
                        tuple(tag.corners[idx, :].astype(int)),
                        (0, 255, 0),
                        2,
                    )

                # Draw center
                center = tuple(tag.center.astype(int))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # Annotate tag ID
                cv2.putText(
                    frame,
                    str(tag.tag_id),
                    (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

                print(f"Tag ID: {tag.tag_id} | Center: {center} | Family: {tag.tag_family}")

            cv2.imshow("AprilTag Check", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
