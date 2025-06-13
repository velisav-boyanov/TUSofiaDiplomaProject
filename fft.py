import numpy as np
import librosa
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import os

"""
Load audio and save FFT plots as image frames.

Parameters:
    audio_path (str): Path to the audio file.
    frame_duration (float): Duration of each frame in seconds.
    n_fft (int): FFT size.
    frame_folder (str): Folder to save frame images.
    
Returns:
    frame_files (list): List of saved image frame file paths.
    sr (int): Sample rate of the audio.
"""
def compute_fft_frames(audio_path, frame_duration=0.05, n_fft=2048, frame_folder='frames_fft'):=
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = int(sr * frame_duration)
    hop_length = frame_length

    # Prepare output folder
    os.makedirs(frame_folder, exist_ok=True)

    num_frames = int((len(y) - frame_length) / hop_length)
    frame_files = []

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        segment = y[start:end]
        
        fft = np.fft.fft(segment, n=n_fft)
        magnitude = np.abs(fft)[:n_fft // 2]

        plt.figure(figsize=(8, 4))
        plt.plot(magnitude)
        plt.ylim(0, np.max(magnitude) * 1.1)
        plt.title(f'FFT Frame {i}')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Magnitude')
        plt.tight_layout()
        frame_path = f'{frame_folder}/frame_{i:05d}.png'
        plt.savefig(frame_path)
        plt.close()
        frame_files.append(frame_path)

    return frame_files, sr


"""
Create video from image frames and attach audio.

Parameters:
    frame_files (list): List of image paths.
    audio_path (str): Path to original audio file.
    output_path (str): Output video file path.
    fps (float): Frames per second (derived from frame_duration).
"""
def create_video_from_frames(frame_files, audio_path, output_path, fps):
    clip = mpy.ImageSequenceClip(frame_files, fps=fps)
    clip = clip.set_audio(mpy.AudioFileClip(audio_path))
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    print(f"Video saved to {output_path}")


"""
Full pipeline to convert audio into an FFT video visualization.

Parameters:
    audio_path (str): Path to the input audio file.
    output_path (str): Path to the output video file.
    frame_duration (float): Duration of each FFT frame (in seconds).
    n_fft (int): FFT size.
"""
def audio_to_fft_video(audio_path, output_path='fft_audio_video.mp4', frame_duration=0.05, n_fft=2048):
    frame_folder = 'frames_fft'
    frame_files, sr = compute_fft_frames(audio_path, frame_duration, n_fft, frame_folder)
    create_video_from_frames(frame_files, audio_path, output_path, fps=int(1 / frame_duration))


"""
test function
"""
def test():
  audio_to_fft_video('your_audio_file.wav', 'my_fft_video.mp4', frame_duration=0.05)
