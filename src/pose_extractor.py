import os
import cv2
import csv
import mediapipe as mp


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_pose_from_video(
    input_video_path: str,
    output_video_path: str,
    output_csv_path: str,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> None:
    """
    Extract 2D pose keypoints from a pitching video using MediaPipe Pose.
    Save both a skeleton overlay video and a CSV file of pose keypoints.
    """

    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    ensure_dir(os.path.dirname(output_video_path))
    ensure_dir(os.path.dirname(output_csv_path))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    landmark_names = [landmark.name.lower() for landmark in mp_pose.PoseLandmark]

    csv_header = ["frame_idx", "time_sec"]
    for name in landmark_names:
        csv_header.extend([
            f"{name}_x",
            f"{name}_y",
            f"{name}_z",
            f"{name}_visibility",
        ])

    frame_idx = 0

    with open(output_csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as pose:

            while True:
                success, frame = cap.read()
                if not success:
                    break

                time_sec = frame_idx / fps if fps > 0 else 0.0

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False

                results = pose.process(rgb_frame)

                rgb_frame.flags.writeable = True
                output_frame = frame.copy()

                row = [frame_idx, time_sec]

                if results.pose_landmarks:
                    for lm in results.pose_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z, lm.visibility])

                    mp_drawing.draw_landmarks(
                        output_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                else:
                    for _ in landmark_names:
                        row.extend(["", "", "", ""])

                csv_writer.writerow(row)
                writer.write(output_frame)

                frame_idx += 1

    cap.release()
    writer.release()

    print("Pose extraction completed.")
    print(f"Frames processed: {frame_idx}")
    print(f"Overlay video saved to: {output_video_path}")
    print(f"Keypoints CSV saved to: {output_csv_path}")


if __name__ == "__main__":
    input_video = "data/raw_videos/sample_pitch.mp4"
    output_video = "outputs/overlay_videos/sample_pitch_overlay.mp4"
    output_csv = "data/pose_keypoints/sample_pitch_keypoints.csv"

    extract_pose_from_video(
        input_video_path=input_video,
        output_video_path=output_video,
        output_csv_path=output_csv,
    )