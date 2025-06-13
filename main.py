import os
import threading
import tkinter as tk
from fft_visualizer import audio_to_fft_video
from ml_visualizer import audio_to_ml_fft_video
from video_helper import DualVideoPlayer

# --- CONFIG ---
audio_files = ["file1.wav", "file2.wav", "file3.wav"]
video_pairs = []

# --- Generate FFT + ML videos ---
for file in audio_files:
    fft_video = f"{os.path.splitext(file)[0]}_fft.mp4"
    ml_video = f"{os.path.splitext(file)[0]}_ml.mp4"
    audio_to_fft_video(file, fft_video)
    audio_to_ml_fft_video(file, ml_video)
    video_pairs.append((fft_video, ml_video))

# --- Launch 3 windows with side-by-side videos ---
windows = []

for idx, (fft_video, ml_video) in enumerate(video_pairs):
    win = tk.Tk()
    win.geometry(f"850x400+{100 + idx * 50}+{100 + idx * 50}")
    player = DualVideoPlayer(win, fft_video, ml_video)
    windows.append(win)

# --- Start GUI event loops ---
for win in windows:
    threading.Thread(target=win.mainloop).start()
