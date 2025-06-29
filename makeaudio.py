import numpy as np
from scipy.io.wavfile import write

def generate_wav(filename, seed=None):
    # Set random seed for reproducibility / variety
    if seed is not None:
        np.random.seed(seed)

    duration = 20  # seconds
    sample_rate = 44100  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Random frequency ranges for sweep and carrier
    start_freq = np.random.uniform(50, 300)
    end_freq = np.random.uniform(1000, 3000)
    carrier_freq = np.random.uniform(300, 800)
    mod_freq = np.random.uniform(0.5, 5)

    # Frequency sweep
    sweep = np.sin(2 * np.pi * t * (start_freq + (end_freq - start_freq) * t / duration))

    # Amplitude modulation
    mod = np.sin(2 * np.pi * mod_freq * t)
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    am_modulated = (1 + mod) * carrier * 0.5

    # Stereo panning
    left = sweep * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
    right = am_modulated * (0.5 + 0.5 * np.cos(2 * np.pi * 0.1 * t))

    # Combine into stereo
    stereo_signal = np.stack([left, right], axis=1)

    # Normalize to int16
    stereo_signal = (stereo_signal * 32767).astype(np.int16)

    # Write to WAV
    write(filename, sample_rate, stereo_signal)
    print(f"Saved {filename}")