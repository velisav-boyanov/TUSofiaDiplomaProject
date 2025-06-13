import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import moviepy.editor as mpy

"""
Generates waveform and FFT magnitude pairs for training.
"""
def generate_training_data(audio_path, frame_duration=0.05, n_fft=2048, max_samples=1000):
    y, sr = librosa.load(audio_path, sr=None)
    frame_len = int(sr * frame_duration)
    hop_length = frame_len

    X = []
    Y = []

    num_frames = int((len(y) - frame_len) / hop_length)

    for i in range(min(num_frames, max_samples)):
        start = i * hop_length
        end = start + frame_len
        segment = y[start:end]

        if len(segment) < frame_len:
            continue

        fft = np.fft.fft(segment, n=n_fft)
        magnitude = np.abs(fft)[:n_fft // 2]

        X.append(segment)
        Y.append(magnitude)

    return np.array(X), np.array(Y), sr


"""
Trains a linear regression model to mimic FFT magnitude output.
"""
def train_linear_fft_model(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    return model


"""
Uses the trained model to predict FFT-like magnitudes and save plots as frames.
"""
def generate_frames_from_model(model, y, sr, frame_duration, frame_folder, n_fft):
    os.makedirs(frame_folder, exist_ok=True)
    frame_len = int(sr * frame_duration)
    hop_length = frame_len
    num_frames = int((len(y) - frame_len) / hop_length)
    frame_files = []

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_len
        segment = y[start:end]

        if len(segment) < frame_len:
            continue

        predicted_mag = model.predict([segment])[0]

        plt.figure(figsize=(8, 4))
        plt.plot(predicted_mag)
        plt.ylim(0, np.max(predicted_mag) * 1.1)
        plt.title(f'ML FFT Frame {i}')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Magnitude (Predicted)')
        plt.tight_layout()
        frame_path = f'{frame_folder}/frame_{i:05d}.png'
        plt.savefig(frame_path)
        plt.close()
        frame_files.append(frame_path)

    return frame_files


"""
Creates a video from image frames and syncs it with audio.
"""
def create_video_from_frames(frame_files, audio_path, output_path, fps):

    clip = mpy.ImageSequenceClip(frame_files, fps=fps)
    clip = clip.set_audio(mpy.AudioFileClip(audio_path))
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    print(f"Video saved to {output_path}")


"""
Full pipeline using a linear model to generate FFT-like video visualization.
"""
def audio_to_ml_fft_video(audio_path, output_path='ml_fft_video.mp4', frame_duration=0.05, n_fft=2048):
    # Step 1: Prepare data
    X_train, Y_train, sr = generate_training_data(audio_path, frame_duration, n_fft)

    # Step 2: Train model
    model = train_linear_fft_model(X_train, Y_train)

    # Step 3: Predict and generate frames
    y, _ = librosa.load(audio_path, sr=sr)
    frame_folder = 'frames_ml_fft'
    frame_files = generate_frames_from_model(model, y, sr, frame_duration, frame_folder, n_fft)

    # Step 4: Create video
    create_video_from_frames(frame_files, audio_path, output_path, fps=int(1 / frame_duration))

"""
test function
"""
def test2():
   audio_to_ml_fft_video('your_audio_file.wav', 'linear_fft_video.mp4', frame_duration=0.05)
