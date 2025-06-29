import os
import threading
import tkinter as tk
from fft import audio_to_fft_video
from lml import audio_to_ml_fft_video
from video_helper import DualVideoPlayer
from makeaudio import generate_wav

# --- CONFIG ---
generate_wav("file1.wav", seed =101)
generate_wav("file3.wav", seed =222)
generate_wav("file2.wav", seed =330)

audio_files = ["file1.wav", "file2.wav", "file3.wav"]
video_pairs = []

# --- Generate FFT + ML videos ---
for file in audio_files:
   fft_video = f"{os.path.splitext(file)[0]}_fft.mp4"
   ml_video = f"{os.path.splitext(file)[0]}_ml.mp4"
   audio_to_fft_video(file, fft_video)
   audio_to_ml_fft_video(file, ml_video)
   video_pairs.append((fft_video, ml_video))

#FOR TESTING ONLY
#video_pairs.append(("file1_fft.mp4","file1_ml.mp4"))
#video_pairs.append(("file2_fft.mp4","file2_ml.mp4"))
#video_pairs.append(("file3_fft.mp4","file3_ml.mp4"))

# --- Launch 3 windows with side-by-side videos ---
windows = []

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
    
    player = DualVideoPlayer(win, fft_video, ml_video)
    windows.append(win)

# --- Start GUI event loops ---
root.mainloop()
