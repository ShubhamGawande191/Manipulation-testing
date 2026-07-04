import asyncio
import logging

import cv2
from pupil_apriltags import Detector

from robot_manipulator.tasks.pick_and_place import BridgeClient

# Setup Logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("find_world_tag")


async def scan_for_tag():
    bridge = BridgeClient()
    await bridge.connect()

    logger.info("Taking picture of AprilTag from current physical position...")

    at_detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    frame = await bridge.image("wrist")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tags = at_detector.detect(gray)
    if tags:
        tag = tags[0]  # Grab first detected tag
        center = (int(tag.center[0]), int(tag.center[1]))  # Force standard int
        logger.info(f"Tag {tag.tag_id} Found at camera pixel: {center}!")

        # Save visual confirmation
        for idx in range(len(tag.corners)):
            cv2.line(
                frame,
                tuple(tag.corners[idx - 1, :].astype(int)),
                tuple(tag.corners[idx, :].astype(int)),
                (0, 255, 0),
                2,
            )
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.imwrite("/tmp/found_apriltag.jpg", frame)

        state = await bridge.state()
        joints = state.get("joints", [])

        logger.info("==================================================")
        logger.info("Current Joint State saved as 'World Origin':")
        logger.info(f"{joints}")
        logger.info("==================================================")

        logger.info("Saved image of tag to /tmp/found_apriltag.jpg")

        import json

        with open("anchor_config.json", "w") as f:
            json.dump({"anchor_joints": joints, "pixel_center": center}, f)

    else:
        logger.error("Could not find AprilTag. Is the arm still looking at it?")

    await bridge.close()


if __name__ == "__main__":
    try:
        asyncio.run(scan_for_tag())
    except KeyboardInterrupt:
        logger.info("Interrupted.")
