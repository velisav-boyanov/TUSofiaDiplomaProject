# --- GENERIC PYTHON LIBS ---
import os
import threading
import tkinter as tk
from pathlib import Path

# --- CUSTOM IMPORTS ---
from fft import audio_to_fft_video
from lml import audio_to_ml_fft_video
from video_helper import DualVideoPlayer
from makeaudio import generate_wav
from freq_cmp import freq_cmp_main

AUDIO_DIR = Path("audio_out")
VIDEO_DIR = Path("video_out")

# --- Generate Audio ---
generate_wav("file1.wav", seed =101)
generate_wav("file3.wav", seed =222)
generate_wav("file2.wav", seed =330)

audio_files = ["file1.wav", "file2.wav", "file3.wav"]
video_pairs = []

# --- Generate FFT + ML videos ---
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
for file in audio_files:
   fft_video = f"{os.path.splitext(file)[0]}_fft.mp4"
   ml_video = f"{os.path.splitext(file)[0]}_ml.mp4"

   print(f"Meassurement for {fft_video}")
   audio_to_fft_video(str(AUDIO_DIR/file), str(VIDEO_DIR/fft_video))
   
   print(f"Meassurement for {ml_video}")
   audio_to_ml_fft_video(str(AUDIO_DIR/file), str(VIDEO_DIR/ml_video))

   video_pairs.append((VIDEO_DIR/fft_video, VIDEO_DIR/ml_video))

#FOR TESTING ONLY
# video_pairs.append(("file1_fft.mp4","file1_ml.mp4"))
# video_pairs.append(("file2_fft.mp4","file2_ml.mp4"))
# video_pairs.append(("file3_fft.mp4","file3_ml.mp4"))

#Run code to compare the file groups
freq_cmp_main("file1_fft.mp4", "file1_ml.mp4")

# --- Launch 3 windows with side-by-side videos ---
windows = []

if video_pairs != []:
    root = tk.Tk()
    root.geometry("850x400+100+100")
    root.title("Main Window")

    for idx, (fft_video, ml_video) in enumerate(video_pairs):
        # For the first window, reuse root:
        if idx == 0:
            win = root
        else:
            # For others, create Toplevel windows
            win = tk.Toplevel(root)
            win.geometry(f"850x400+{100 + idx * 50}+{100 + idx * 50}")
        
        print(fft_video)
        print(ml_video)
        player = DualVideoPlayer(win, fft_video, ml_video)
        windows.append(win)

    # --- Start GUI event loops ---
    root.mainloop()
