import os
import math
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_point(row: pd.Series, name: str) -> np.ndarray:
    return np.array([row[f"{name}_x"], row[f"{name}_y"]], dtype=float)


def angle_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def compute_arm_slot(row: pd.Series, side: str) -> float:
    shoulder = get_point(row, f"{side}_shoulder")
    wrist = get_point(row, f"{side}_wrist")

    dx = wrist[0] - shoulder[0]
    dy = shoulder[1] - wrist[1]

    return float(math.degrees(math.atan2(dy, dx)))


def compute_elbow_angle(row: pd.Series, side: str) -> float:
    shoulder = get_point(row, f"{side}_shoulder")
    elbow = get_point(row, f"{side}_elbow")
    wrist = get_point(row, f"{side}_wrist")

    return angle_between_points(shoulder, elbow, wrist)


def compute_trunk_tilt(row: pd.Series) -> float:
    left_shoulder = get_point(row, "left_shoulder")
    right_shoulder = get_point(row, "right_shoulder")
    left_hip = get_point(row, "left_hip")
    right_hip = get_point(row, "right_hip")

    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0

    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]

    return float(math.degrees(math.atan2(dx, dy)))


def normalize_score(value: float, low: float, high: float) -> float:
    if np.isnan(value):
        return 0.0

    if high <= low:
        return 0.0

    score = 100.0 * (value - low) / (high - low)
    return float(np.clip(score, 0.0, 100.0))


def calculate_pitch_intensity(
    keypoints_csv_path: str,
    release_summary_path: str,
    output_summary_path: str,
    output_plot_path: str,
    throwing_side: str = "right",
    fps: float = 30.0,
    window_frames: int = 10,
) -> Dict[str, float]:

    if not os.path.exists(keypoints_csv_path):
        raise FileNotFoundError(f"Keypoints CSV not found: {keypoints_csv_path}")

    if not os.path.exists(release_summary_path):
        raise FileNotFoundError(f"Release summary not found: {release_summary_path}")

    ensure_dir(os.path.dirname(output_summary_path))
    ensure_dir(os.path.dirname(output_plot_path))

    df = pd.read_csv(keypoints_csv_path)
    release_df = pd.read_csv(release_summary_path)

    release_frame = int(release_df.loc[0, "release_frame"])
    release_time = float(release_df.loc[0, "release_time_sec"])
    release_confidence = str(release_df.loc[0, "confidence"])

    speed_col = f"{throwing_side}_wrist_speed"

    if speed_col not in df.columns:
        raise ValueError(f"Missing wrist speed column: {speed_col}")

    accel_col = f"{throwing_side}_wrist_acceleration"
    df[accel_col] = df[speed_col].diff().fillna(0.0) * fps

    elbow_angle_col = f"{throwing_side}_elbow_angle"
    elbow_speed_col = f"{throwing_side}_elbow_extension_speed"

    df[elbow_angle_col] = df.apply(lambda row: compute_elbow_angle(row, throwing_side), axis=1)
    df[elbow_speed_col] = df[elbow_angle_col].diff().fillna(0.0) * fps

    arm_slot_col = f"{throwing_side}_arm_slot"
    df[arm_slot_col] = df.apply(lambda row: compute_arm_slot(row, throwing_side), axis=1)
    df["trunk_tilt"] = df.apply(lambda row: compute_trunk_tilt(row), axis=1)

    start_frame = max(0, release_frame - window_frames)
    end_frame = release_frame + window_frames

    window_df = df[(df["frame_idx"] >= start_frame) & (df["frame_idx"] <= end_frame)].copy()

    if window_df.empty:
        raise ValueError("No data found around release frame window.")

    wrist_speed_peak = float(window_df[speed_col].quantile(0.95))
    wrist_accel_peak = float(window_df[accel_col].abs().quantile(0.95))
    elbow_extension_peak = float(window_df[elbow_speed_col].abs().quantile(0.95))

    #wrist_accel_peak = float(np.clip(wrist_accel_peak, 0.0, 50.0))
    #elbow_extension_peak = float(np.clip(elbow_extension_peak, 0.0, 800.0))

    wrist_accel_peak = float(np.clip(wrist_accel_peak, 0.0, 80.0))
    elbow_extension_peak = float(np.clip(elbow_extension_peak, 0.0, 1200.0))

    release_row = df[df["frame_idx"] == release_frame]

    if release_row.empty:
        closest_idx = (df["frame_idx"] - release_frame).abs().idxmin()
        release_row = df.loc[[closest_idx]]

    release_row = release_row.iloc[0]

    arm_slot_at_release = float(release_row[arm_slot_col])
    trunk_tilt_at_release = float(release_row["trunk_tilt"])

    wrist_speed_score = normalize_score(wrist_speed_peak, low=0.5, high=4.5)
    wrist_accel_score = normalize_score(wrist_accel_peak, low=1.0, high=80.0)
    elbow_extension_score = normalize_score(elbow_extension_peak, low=20.0, high=1200.0)

    pitch_intensity_score = (
        0.50 * wrist_speed_score
        + 0.30 * wrist_accel_score
        + 0.20 * elbow_extension_score
    )

    pitch_intensity_score = float(np.clip(pitch_intensity_score, 0.0, 100.0))

    result = {
        "throwing_side": throwing_side,
        "release_frame": release_frame,
        "release_time_sec": release_time,
        "release_confidence": release_confidence,
        "wrist_speed_peak": wrist_speed_peak,
        "wrist_acceleration_peak": wrist_accel_peak,
        "elbow_extension_speed_peak": elbow_extension_peak,
        "arm_slot_at_release": arm_slot_at_release,
        "trunk_tilt_at_release": trunk_tilt_at_release,
        "wrist_speed_score": wrist_speed_score,
        "wrist_acceleration_score": wrist_accel_score,
        "elbow_extension_score": elbow_extension_score,
        "pitch_intensity_score": pitch_intensity_score,
        "note": "Pitch intensity score is a pose-based proxy, not official pitch velocity in mph.",
    }

    pd.DataFrame([result]).to_csv(output_summary_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_sec"], df[speed_col], label="Wrist speed")
    plt.plot(df["time_sec"], df[accel_col], label="Wrist acceleration")
    plt.axvline(release_time, linestyle="--", label="Estimated release")

    plt.xlabel("Time (sec)")
    plt.ylabel("Normalized feature value")
    plt.title(f"Pitch Intensity Features | Score = {pitch_intensity_score:.1f}/100")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)
    plt.close()

    print("Pitch intensity calculation completed.")
    print(f"Release frame: {release_frame}")
    print(f"Release time: {release_time:.3f} sec")
    print(f"Wrist speed peak: {wrist_speed_peak:.6f}")
    print(f"Wrist acceleration peak: {wrist_accel_peak:.6f}")
    print(f"Elbow extension speed peak: {elbow_extension_peak:.6f}")
    print(f"Arm slot at release: {arm_slot_at_release:.2f} degrees")
    print(f"Trunk tilt at release: {trunk_tilt_at_release:.2f} degrees")
    print(f"Pitch intensity score: {pitch_intensity_score:.2f}/100")
    print(f"Summary saved to: {output_summary_path}")
    print(f"Plot saved to: {output_plot_path}")

    return result


if __name__ == "__main__":
    keypoints_csv = "data/pose_keypoints/sample_pitch_keypoints_smoothed.csv"
    release_summary = "reports/release_summary.csv"
    output_summary = "reports/pitch_intensity_summary.csv"
    output_plot = "outputs/plots/pitch_intensity_features.png"

    fps = 59.687616214206024
    throwing_side = "right"

    calculate_pitch_intensity(
        keypoints_csv_path=keypoints_csv,
        release_summary_path=release_summary,
        output_summary_path=output_summary,
        output_plot_path=output_plot,
        throwing_side=throwing_side,
        fps=fps,
        window_frames=10,
    )