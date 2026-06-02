import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def smooth_series(series: pd.Series, window_length: int = 11, polyorder: int = 2) -> pd.Series:
    """
    Interpolate missing values, then smooth using Savitzky-Golay filter.
    """
    s = pd.to_numeric(series, errors="coerce")

    # Interpolate missing values
    s = s.interpolate(method="linear", limit_direction="both")

    # If still all NaN or too short, return as is
    if s.isna().all():
        return s

    n = len(s)

    # window_length must be odd and <= n
    if n < 3:
        return s

    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1

    if window_length < 3:
        return s

    if window_length % 2 == 0:
        window_length -= 1

    if window_length <= polyorder:
        window_length = polyorder + 1
        if window_length % 2 == 0:
            window_length += 1
        if window_length > n:
            return s

    smoothed = savgol_filter(s.to_numpy(), window_length=window_length, polyorder=polyorder)
    return pd.Series(smoothed, index=s.index)


def compute_speed(x: pd.Series, y: pd.Series, fps: float) -> pd.Series:
    """
    Compute frame-to-frame 2D speed from x/y coordinates.
    """
    dx = x.diff()
    dy = y.diff()
    dt = 1.0 / fps if fps > 0 else 1.0
    speed = np.sqrt(dx**2 + dy**2) / dt
    return speed.fillna(0.0)


def plot_wrist_trajectory(df: pd.DataFrame, x_col: str, y_col: str, output_path: str, title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot(df[x_col], df[y_col], linewidth=2)
    plt.gca().invert_yaxis()  # image coordinates: y increases downward
    plt.xlabel("Normalized X")
    plt.ylabel("Normalized Y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_speed_curve(df: pd.DataFrame, time_col: str, speed_col: str, output_path: str, title: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df[time_col], df[speed_col], linewidth=2)
    peak_idx = df[speed_col].idxmax()
    peak_time = df.loc[peak_idx, time_col]
    peak_speed = df.loc[peak_idx, speed_col]

    plt.axvline(peak_time, linestyle="--")
    plt.scatter([peak_time], [peak_speed], s=40)
    plt.text(peak_time, peak_speed, f"  Peak @ {peak_time:.2f}s", va="bottom")

    plt.xlabel("Time (sec)")
    plt.ylabel("Speed (normalized units/sec)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def smooth_pose_keypoints(
    input_csv_path: str,
    output_csv_path: str,
    output_plot_dir: str,
    fps: float,
    throwing_side: str = "right",
    window_length: int = 11,
    polyorder: int = 2,
) -> None:
    """
    Smooth pose keypoints CSV and generate wrist trajectory / speed plots.
    """

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    ensure_dir(os.path.dirname(output_csv_path))
    ensure_dir(output_plot_dir)

    df = pd.read_csv(input_csv_path)

    # Convert numeric columns safely
    for col in df.columns:
        if col not in ["frame_idx"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Smooth all x/y/z/visibility columns
    smoothed_df = df.copy()
    keypoint_cols = [
        col for col in df.columns
        if col.endswith("_x") or col.endswith("_y") or col.endswith("_z") or col.endswith("_visibility")
    ]

    for col in keypoint_cols:
        smoothed_df[col] = smooth_series(df[col], window_length=window_length, polyorder=polyorder)

    # Select wrist columns
    wrist_x = f"{throwing_side}_wrist_x"
    wrist_y = f"{throwing_side}_wrist_y"

    if wrist_x not in smoothed_df.columns or wrist_y not in smoothed_df.columns:
        raise ValueError(f"Missing wrist columns for side='{throwing_side}'")

    # Compute wrist speed
    speed_col = f"{throwing_side}_wrist_speed"
    smoothed_df[speed_col] = compute_speed(smoothed_df[wrist_x], smoothed_df[wrist_y], fps=fps)

    # Save smoothed CSV
    smoothed_df.to_csv(output_csv_path, index=False)

    # Plot trajectory
    trajectory_plot_path = os.path.join(output_plot_dir, f"{throwing_side}_wrist_trajectory.png")
    plot_wrist_trajectory(
        smoothed_df,
        wrist_x,
        wrist_y,
        trajectory_plot_path,
        title=f"{throwing_side.capitalize()} Wrist Trajectory (Smoothed)",
    )

    # Plot speed curve
    speed_plot_path = os.path.join(output_plot_dir, f"{throwing_side}_wrist_speed.png")
    plot_speed_curve(
        smoothed_df,
        time_col="time_sec",
        speed_col=speed_col,
        output_path=speed_plot_path,
        title=f"{throwing_side.capitalize()} Wrist Speed Curve (Smoothed)",
    )

    peak_idx = smoothed_df[speed_col].idxmax()
    peak_frame = int(smoothed_df.loc[peak_idx, "frame_idx"])
    peak_time = float(smoothed_df.loc[peak_idx, "time_sec"])
    peak_speed = float(smoothed_df.loc[peak_idx, speed_col])

    print("Keypoint smoothing completed.")
    print(f"Smoothed CSV saved to: {output_csv_path}")
    print(f"Trajectory plot saved to: {trajectory_plot_path}")
    print(f"Speed plot saved to: {speed_plot_path}")
    print(f"Peak wrist speed frame: {peak_frame}")
    print(f"Peak wrist speed time: {peak_time:.3f} sec")
    print(f"Peak wrist speed value: {peak_speed:.6f}")


if __name__ == "__main__":
    input_csv = "data/pose_keypoints/sample_pitch_keypoints.csv"
    output_csv = "data/pose_keypoints/sample_pitch_keypoints_smoothed.csv"
    output_plot_dir = "outputs/plots"

    # 你先假設這支影片是右投；如果之後是左投，再改成 "left"
    throwing_side = "right"

    # 這裡先直接填你剛剛影片的 FPS
    # 如果不知道，可以先填 30
    fps = 30.0

    smooth_pose_keypoints(
        input_csv_path=input_csv,
        output_csv_path=output_csv,
        output_plot_dir=output_plot_dir,
        fps=fps,
        throwing_side=throwing_side,
        window_length=11,
        polyorder=2,
    )