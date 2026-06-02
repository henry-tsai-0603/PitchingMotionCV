import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def detect_release_frame(
    input_csv_path: str,
    output_summary_path: str,
    output_plot_path: str,
    overlay_video_path: Optional[str] = None,
    output_snapshot_path: Optional[str] = None,
    throwing_side: str = "right",
    fps: float = 30.0,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    release_offset_frames: int = 5,
) -> dict:
    """
    Detect an estimated release frame using the wrist speed peak.

    This is a proxy method:
    release frame is estimated as the strongest wrist-speed peak
    within an optional analysis time window.
    """

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    ensure_dir(os.path.dirname(output_summary_path))
    ensure_dir(os.path.dirname(output_plot_path))

    df = pd.read_csv(input_csv_path)

    speed_col = f"{throwing_side}_wrist_speed"

    if speed_col not in df.columns:
        raise ValueError(f"Missing speed column: {speed_col}")

    # Filter analysis window
    analysis_df = df.copy()

    if start_time is not None:
        analysis_df = analysis_df[analysis_df["time_sec"] >= start_time]

    if end_time is not None:
        analysis_df = analysis_df[analysis_df["time_sec"] <= end_time]

    if analysis_df.empty:
        raise ValueError("No frames remain after applying start_time/end_time window.")

    speed = analysis_df[speed_col].to_numpy()
    frames = analysis_df["frame_idx"].to_numpy()
    times = analysis_df["time_sec"].to_numpy()

    # Peak detection settings
    # distance: avoid peaks too close together
    min_peak_distance = max(1, int(0.15 * fps))

    # prominence: require peak to stand out from surrounding motion
    speed_range = np.nanmax(speed) - np.nanmin(speed)
    prominence = max(0.01, 0.10 * speed_range)

    peaks, properties = find_peaks(
        speed,
        distance=min_peak_distance,
        prominence=prominence,
    )

    # Fallback: if no peak found, use global max
    if len(peaks) == 0:
        best_local_idx = int(np.nanargmax(speed))
        peak_method = "global_max_fallback"
    else:
        # Pick the highest detected peak
        best_peak_idx = peaks[np.argmax(speed[peaks])]
        best_local_idx = int(best_peak_idx)
        peak_method = "find_peaks_highest_peak"

    peak_frame = int(frames[best_local_idx])
    peak_time = float(times[best_local_idx])
    peak_speed = float(speed[best_local_idx])

    release_frame = max(0, peak_frame - release_offset_frames)

    release_row = df[df["frame_idx"] == release_frame]
    if release_row.empty:
        closest_idx = (df["frame_idx"] - release_frame).abs().idxmin()
        release_time = float(df.loc[closest_idx, "time_sec"])
        release_speed = float(df.loc[closest_idx, speed_col])
    else:
        release_time = float(release_row.iloc[0]["time_sec"])
        release_speed = float(release_row.iloc[0][speed_col])

    # Compute confidence based on relative peak strength
    median_speed = float(np.nanmedian(speed))
    std_speed = float(np.nanstd(speed))
    z_score = (release_speed - median_speed) / std_speed if std_speed > 0 else 0.0

    if z_score >= 2.0:
        confidence = "High"
    elif z_score >= 1.0:
        confidence = "Medium"
    else:
        confidence = "Low"

    result = {
        "throwing_side": throwing_side,
        "release_frame": release_frame,
        "release_time_sec": release_time,
        "release_wrist_speed": release_speed,
        "peak_method": peak_method,
        "confidence": confidence,
        "z_score": z_score,
        "start_time": start_time,
        "end_time": end_time,
        "peak_frame": peak_frame,
        "peak_time_sec": peak_time,
        "peak_wrist_speed": peak_speed,
        "release_offset_frames": release_offset_frames,
    }

    # Save summary CSV
    pd.DataFrame([result]).to_csv(output_summary_path, index=False)

    # Plot speed curve with release marker
    plt.figure(figsize=(10, 4))
    plt.plot(df["time_sec"], df[speed_col], linewidth=2, label="Wrist speed")

    if start_time is not None:
        plt.axvline(start_time, linestyle=":", label="Start window")
    if end_time is not None:
        plt.axvline(end_time, linestyle=":", label="End window")

    plt.axvline(release_time, linestyle="--", label=f"Estimated release: {release_time:.2f}s")
    plt.scatter([release_time], [release_speed], s=50)

    plt.xlabel("Time (sec)")
    plt.ylabel("Wrist speed (normalized units/sec)")
    plt.title(f"Estimated Release Frame from {throwing_side.capitalize()} Wrist Speed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)
    plt.close()

    # Save release frame snapshot if overlay video is provided
    if overlay_video_path is not None and output_snapshot_path is not None:
        ensure_dir(os.path.dirname(output_snapshot_path))

        if os.path.exists(overlay_video_path):
            cap = cv2.VideoCapture(overlay_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame)
            success, frame = cap.read()
            cap.release()

            if success:
                cv2.putText(
                    frame,
                    f"Estimated Release Frame: {release_frame}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(output_snapshot_path, frame)
            else:
                print(f"Warning: failed to read frame {release_frame} from overlay video.")
        else:
            print(f"Warning: overlay video not found: {overlay_video_path}")

    print("Release frame detection completed.")
    print(f"Estimated release frame: {release_frame}")
    print(f"Estimated release time: {release_time:.3f} sec")
    print(f"Release wrist speed: {release_speed:.6f}")
    print(f"Confidence: {confidence}")
    print(f"Summary saved to: {output_summary_path}")
    print(f"Plot saved to: {output_plot_path}")

    if output_snapshot_path is not None:
        print(f"Snapshot saved to: {output_snapshot_path}")

    return result


if __name__ == "__main__":
    input_csv = "data/pose_keypoints/sample_pitch_keypoints_smoothed.csv"
    output_summary = "reports/release_summary.csv"
    output_plot = "outputs/plots/right_wrist_speed_release.png"
    overlay_video = "outputs/overlay_videos/sample_pitch_overlay.mp4"
    output_snapshot = "outputs/plots/release_frame_snapshot.png"

    fps = 59.687616214206024
    throwing_side = "right"

    # 第一版先用整段影片。
    # 如果偵測點明顯太早或太晚，再設定 start_time / end_time。
    detect_release_frame(
        input_csv_path=input_csv,
        output_summary_path=output_summary,
        output_plot_path=output_plot,
        overlay_video_path=overlay_video,
        output_snapshot_path=output_snapshot,
        throwing_side=throwing_side,
        fps=fps,
        start_time=None,
        end_time=None,
        release_offset_frames=5,
    )