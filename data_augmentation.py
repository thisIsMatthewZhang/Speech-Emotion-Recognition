import librosa
import numpy as np

"""
Data augmentation: create new synthetic data samples by adding slight perturbations on existing training set
augmentation techniques for audio include noise injection, pitch change, shifting time, and speed
the goal is to make our learning model invariant to the perturbations and enhance its ability to generalize
"""
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    temp = data + noise_amp * np.random.normal(size=data.shape[0])
    return temp

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)