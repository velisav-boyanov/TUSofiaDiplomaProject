import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.stats import pearsonr

FFT_VIDEO = "file1_fft.mp4"       # video from FFT
ML_VIDEO = "file1_ml.mp4"         # vide from ML
OUT_DIR = Path("video_eval")
VIDEO_DIR = Path("video_out")

SR = 44100       # sample rate (for mapping the Hz to lines)
FMAX = 22050.0   # max Hz

BANDS_HZ = {
    "low": (20, 250),
    "mid": (250, 4000),
    "high": (4000, 20000)
}

SAVE_PLOTS = True


"""
Load video frames in to array

Parameters:
    video_path (str): Path to the video file.
    
Returns:
    numpy.ndarray: A stack of grayscale frames as float32 over axis 0
"""
def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cant open the video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
    cap.release()
    if not frames:
        raise ValueError(f"No frame found: {video_path}")
    return np.stack(frames, axis=0)


"""
Convert a frequency band into row indices for a spectrogram-like image

Parameters:
    sr (float): Sampling rate of the original signal.
    band (tuple): Frequency band as (low_hz, high_hz).
    height (int): Number of rows in the spectrogram image.
    fmax (float, optional): Maximum frequency to consider. Defaults to Nyquist.

Returns:
    tuple: (row_start, row_end) indices corresponding to the frequency band
"""
def band_rows_from_freqs(sr, band, height, fmax=None):
    nyquist = sr / 2.0
    top_freq = fmax if fmax else nyquist
    low_hz, high_hz = max(0, band[0]), min(top_freq, band[1])

    def hz_to_row(hz):
        frac = hz / top_freq
        row_from_bottom = int(round(frac * (height - 1)))
        return height - 1 - row_from_bottom

    row_top = hz_to_row(high_hz)
    row_bottom = hz_to_row(low_hz)
    return (min(row_top, row_bottom), max(row_top, row_bottom) + 1)


"""
Compute average absolute energy in specified rows of an array

Parameters:
    arr (numpy.ndarray): 2D array (e.g., spectrogram frame)
    row_start (int): Starting row index
    row_end (int): Ending row index (exclusive)

Returns:
    float: Mean absolute value of the array in the specified rows
"""
def energy_in_rows(arr, row_start, row_end):
    h = arr.shape[0]
    rs, re = max(0, row_start), min(h, row_end)
    if re <= rs:
        return 0.0
    return float(np.abs(arr[rs:re]).mean())


"""
Compute energy for each frequency band for each frame

Parameters:
    frames (numpy.ndarray): Stack of 2D frames
    bands_rows (dict): Dictionary mapping band names to (row_start, row_end)

Returns:
    pandas.DataFrame: Each row contains frame index and energy per band with band limits
"""
def compute_band_energies(frames, bands_rows):
    records = []
    for i, arr in enumerate(frames):
        per_band = {}
        for name, rows in bands_rows.items():
            per_band[f"{name}"] = energy_in_rows(arr, *rows)
            per_band[f"{name}_hz_low"] = BANDS_HZ[name][0]
            per_band[f"{name}_hz_high"] = BANDS_HZ[name][1]
        records.append({"frame": i, **per_band})
    return pd.DataFrame(records)


"""
Compute error metrics between true and predicted values

Parameters:
    y_true (numpy.ndarray): Ground truth values
    y_pred (numpy.ndarray): Predicted values

Returns:
    dict: Dictionary with MAE, RMSE, MAPE%, and Pearson correlation
"""
def metrics(y_true, y_pred):
    eps = 1e-8
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    mape = float(np.mean(np.abs(diff) / np.clip(np.abs(y_true), eps, None))) * 100
    try:
        r, _ = pearsonr(y_true, y_pred)
    except Exception:
        r = float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "Pearson_r": r}



"""
Main function to load videos, compute band energies, compare, and save results

Steps:
    1. Load FFT and ML videos
    2. Align frame lengths
    3. Compute row mapping for each frequency band
    4. Compute energy per band for each frame
    5. Merge results and save CSV
    6. Compute and save summary metrics
    7. Optionally save plots per band

Returns:
    None
"""
def freq_cmp_main(fft = FFT_VIDEO, mll = ML_VIDEO):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # зареждане на двете видеа
    fft_frames = load_video_frames(str(VIDEO_DIR/fft))
    ml_frames = load_video_frames(str(VIDEO_DIR/mll))

    min_len = min(len(fft_frames), len(ml_frames))
    fft_frames, ml_frames = fft_frames[:min_len], ml_frames[:min_len]

    height = fft_frames.shape[1]
    bands_rows = {name: band_rows_from_freqs(SR, band, height, FMAX) for name, band in BANDS_HZ.items()}

    df_fft = compute_band_energies(fft_frames, bands_rows)
    df_ml = compute_band_energies(ml_frames, bands_rows)

    df = pd.merge(df_fft, df_ml, on="frame", suffixes=("_fft", "_ml"))

    # Запис само на енергията + честотните граници
    df.to_csv(OUT_DIR / "band_energies.csv", index=False)

    # Обобщени метрики за грешките
    summary = {band: metrics(df[f"{band}_fft"].to_numpy(), df[f"{band}_ml"].to_numpy()) for band in BANDS_HZ}
    all_true = np.concatenate([df[f"{b}_fft"] for b in BANDS_HZ])
    all_pred = np.concatenate([df[f"{b}_ml"] for b in BANDS_HZ])
    summary["overall"] = metrics(all_true, all_pred)

    with open(OUT_DIR / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    if SAVE_PLOTS:
        t = df["frame"].to_numpy()
        for band in BANDS_HZ:
            plt.figure()
            plt.plot(t, df[f"{band}_fft"], label="FFT")
            plt.plot(t, df[f"{band}_ml"], label="ML", linestyle="--")
            plt.title(f"{band.upper()} frequency band")
            plt.legend(); plt.xlabel("Frame"); plt.ylabel("Energy (0–255) 8bit color")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"{band}_band.png", dpi=150)
            plt.close()

    print("Results saved in:", OUT_DIR)

# Стойностите в колоните energy всъщност са средната абсолютна яркост на пикселите в спектрограмното видео в рамките на дадена честотна лента.

# Стъпка по стъпка:

# Всеки кадър от .mp4 видеото се конвертира в черно-бяло (стойности 0–255).

# За дадена честотна лента (например low = 20–250 Hz), кодът изчислява кои редове по вертикалата на изображението ѝ отговарят.

# Извлича този вертикален сегмент от кадъра и пресмята: