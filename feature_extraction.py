import numpy as np
import librosa
from data_augmentation import *

"""
Feature extraction: analyze the relations between different things
however, we must convert the audio into a format that the machine can understand
audio signal is a three-dimensional signal in which three axes represent time, amplitude and frequency.
Features:
1) Zero Crossing Rate : The rate of sign-changes of the signal during the duration of a particular frame.
    - used to discern between voiced (e.g. vowels) and unvoiced (e.g. consonants) sounds
    - voiced sounds have low zcr b/c they are more periodic; unvoiced sounds have high zcr b/c they have more noise-like characteristics
2) Energy : The sum of squares of the signal values, normalized by the respective frame length.
3) Entropy of Energy : The entropy of sub-frames’ normalized energies. It can be interpreted as a measure of abrupt changes.
4) Spectral Centroid : The center of gravity of the spectrum.
5) Spectral Spread : The second central moment of the spectrum.
6) Spectral Entropy : Entropy of the normalized spectral energies for a set of sub-frames.
7) Spectral Flux : The squared difference between the normalized magnitudes of the spectra of the two successive frames.
8) Spectral Rolloff : The frequency below which 90% of the magnitude distribution of the spectrum is concentrated.
9) MFCCs Mel Frequency Cepstral Coefficients form a cepstral representation where the frequency bands are not linear but distributed according to the mel-scale.
10) Chroma Vector : A 12-element representation of the spectral energy where the bins represent the 12 equal-tempered pitch classes of western-type music (semitone spacing).
11) Chroma Deviation : The standard deviation of the 12 chroma coefficients.
12) Chroma STFT : captures the harmonic content of an audio signal
    - maps the energy of frequencies to their respective pitch classes, regardless of the octave.
    - focuses on harmonic structure rather than the full spectrum of frequencies.
13) Root Mean Square Value : measure of energy or loudness of an audio signal
    - provides a scalar value representing the signal's average amplitude over time
and more...
    
"""

""" TODO: leverage more feature extraction techniques to produce more informed results (i.e. spectral rolloff, chroma vector, chroma deviation) """
def extract_features(data, sample_rate):
    # Zero Crossing Rate
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally so that multiple features can be concatenated together
    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))
    # Root Mean Square Value
    rmsv = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rmsv))
    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    # spectral rolloff
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spec_rolloff))
    # tonal centroid
    tonal_cent = np.mean(librosa.feature.tonnetz(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, tonal_cent))
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    stretch_data = stretch(data)
    data_stretch_pitch = pitch(data=stretch_data, sr=sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))
    return result
