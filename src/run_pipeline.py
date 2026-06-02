import argparse
import os
import sys
import cv2


# Make sure src modules can be imported when running from project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.pose_extractor import extract_pose_from_video
from src.smoothing import smooth_pose_keypoints
from src.release_detection import detect_release_frame
from src.velocity_proxy import calculate_pitch_intensity


def get_video_fps(video_path: str) -> float:
    """
    Read FPS from input video using OpenCV.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        raise ValueError(f"Invalid FPS detected: {fps}")

    return float(fps)


def run_pipeline(
    video_path: str,
    pitch_id: str,
    throwing_side: str,
    fps: float,
    release_offset_frames: int,
    start_time: float,
    end_time: float,
) -> None:
    """
    Run the full pitching motion CV pipeline.

    Steps:
    1. Pose extraction
    2. Keypoint smoothing
    3. Release frame detection
    4. Pitch intensity proxy calculation
    """

    # Output paths
    overlay_video_path = f"outputs/overlay_videos/{pitch_id}_overlay.mp4"

    raw_keypoints_csv = f"data/pose_keypoints/{pitch_id}_keypoints.csv"
    smoothed_keypoints_csv = f"data/pose_keypoints/{pitch_id}_keypoints_smoothed.csv"

    wrist_trajectory_plot = f"outputs/plots/{pitch_id}_{throwing_side}_wrist_trajectory.png"
    wrist_speed_plot = f"outputs/plots/{pitch_id}_{throwing_side}_wrist_speed.png"

    release_summary_csv = f"reports/{pitch_id}_release_summary.csv"
    release_plot = f"outputs/plots/{pitch_id}_{throwing_side}_wrist_speed_release.png"
    release_snapshot = f"outputs/plots/{pitch_id}_release_frame_snapshot.png"

    intensity_summary_csv = f"reports/{pitch_id}_pitch_intensity_summary.csv"
    intensity_plot = f"outputs/plots/{pitch_id}_pitch_intensity_features.png"

    print("=" * 80)
    print("PitchingMotionCV Pipeline")
    print("=" * 80)
    print(f"Input video: {video_path}")
    print(f"Pitch ID: {pitch_id}")
    print(f"Throwing side: {throwing_side}")
    print(f"FPS: {fps}")
    print(f"Release offset frames: {release_offset_frames}")
    print(f"Analysis start time: {start_time}")
    print(f"Analysis end time: {end_time}")
    print("=" * 80)

    # Step 1: Pose extraction
    print("\n[Step 1] Pose extraction")
    extract_pose_from_video(
        input_video_path=video_path,
        output_video_path=overlay_video_path,
        output_csv_path=raw_keypoints_csv,
    )

    # Step 2: Keypoint smoothing
    print("\n[Step 2] Keypoint smoothing")
    smooth_pose_keypoints(
        input_csv_path=raw_keypoints_csv,
        output_csv_path=smoothed_keypoints_csv,
        output_plot_dir="outputs/plots",
        fps=fps,
        throwing_side=throwing_side,
        window_length=11,
        polyorder=2,
    )

    # The current smoothing.py generates generic names:
    # outputs/plots/right_wrist_trajectory.png
    # outputs/plots/right_wrist_speed.png
    # We keep those, but also the core outputs are saved correctly in CSV.
    # Later we can make plot filenames pitch_id-specific inside smoothing.py.

    # Step 3: Release frame detection
    print("\n[Step 3] Release frame detection")
    detect_release_frame(
        input_csv_path=smoothed_keypoints_csv,
        output_summary_path=release_summary_csv,
        output_plot_path=release_plot,
        overlay_video_path=overlay_video_path,
        output_snapshot_path=release_snapshot,
        throwing_side=throwing_side,
        fps=fps,
        start_time=start_time,
        end_time=end_time,
        release_offset_frames=release_offset_frames,
    )

    # Step 4: Pitch intensity proxy
    print("\n[Step 4] Pitch intensity proxy")
    calculate_pitch_intensity(
        keypoints_csv_path=smoothed_keypoints_csv,
        release_summary_path=release_summary_csv,
        output_summary_path=intensity_summary_csv,
        output_plot_path=intensity_plot,
        throwing_side=throwing_side,
        fps=fps,
        window_frames=10,
    )

    print("\n" + "=" * 80)
    print("Pipeline completed successfully.")
    print("=" * 80)
    print("Generated outputs:")
    print(f"- Overlay video: {overlay_video_path}")
    print(f"- Raw keypoints CSV: {raw_keypoints_csv}")
    print(f"- Smoothed keypoints CSV: {smoothed_keypoints_csv}")
    print(f"- Release summary: {release_summary_csv}")
    print(f"- Release plot: {release_plot}")
    print(f"- Release snapshot: {release_snapshot}")
    print(f"- Pitch intensity summary: {intensity_summary_csv}")
    print(f"- Pitch intensity plot: {intensity_plot}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PitchingMotionCV pipeline on a baseball pitching video."
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input pitching video.",
    )

    parser.add_argument(
        "--pitch-id",
        type=str,
        required=True,
        help="Unique ID/name for this pitch video.",
    )

    parser.add_argument(
        "--throwing-side",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Pitcher's throwing side.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Video FPS. If omitted, FPS will be read automatically from the video.",
    )

    parser.add_argument(
        "--release-offset-frames",
        type=int,
        default=5,
        help="Shift wrist-speed peak earlier by this many frames to estimate release frame.",
    )

    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Optional start time in seconds for release peak search.",
    )

    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Optional end time in seconds for release peak search.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    fps = args.fps
    if fps is None:
        fps = get_video_fps(args.video)
        print(f"Auto-detected FPS: {fps}")

    run_pipeline(
        video_path=args.video,
        pitch_id=args.pitch_id,
        throwing_side=args.throwing_side,
        fps=fps,
        release_offset_frames=args.release_offset_frames,
        start_time=args.start_time,
        end_time=args.end_time,
    )