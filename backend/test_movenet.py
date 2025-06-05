import sys
from pose_detector import detect_pose, NoKeypointError


def main(img_path: str, pose_id: str = "mountain_pose"):
    with open(img_path, "rb") as f:
        image_bytes = f.read()
    try:
        score, buf = detect_pose(image_bytes, pose_id)
        with open("skeleton_output.png", "wb") as out:
            out.write(buf.getvalue())
        print("Score:", score)
        print("Skeleton saved to skeleton_output.png")
    except NoKeypointError as e:
        print("Failed to detect keypoints:", e)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_movenet.py image_path [pose_id]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "mountain_pose")
