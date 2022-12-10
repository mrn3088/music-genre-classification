# Author: Ruinan Ma
# Email: r7ma@ucsd.edu
# In this file I implemented the feature extraction function.
# Since there is a bug in my system to include a certain library, 
# I directly copy this code to my kaggle workspace to do feature extraction.

import librosa
import numpy as np

def feature(path, audio):
    # Load the audio data
    y, sr = librosa.load(path)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    harmony = librosa.effects.harmonic(y=y)
    tempo = librosa.beat.tempo(y=y,sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    label = audio.split('.')[0]
    # Create a dataframe with the features
    data = {"filename": audio,
            "chroma_stft_mean": np.mean(chroma_stft),
            "chroma_stft_var": np.var(chroma_stft),
            "rms_mean": np.mean(rms),
            "rms_var": np.var(rms),
            "spectral_centroid_mean": np.mean(spectral_centroid),
            "spectral_centroid_var": np.var(spectral_centroid),
            "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
            "spectral_bandwidth_var": np.var(spectral_bandwidth),
            "rolloff_mean": np.mean(rolloff),
            "rolloff_var": np.var(rolloff),
            "zero_crossing_rate_mean": np.mean(zero_crossing_rate),
            "zero_crossing_rate_var": np.var(zero_crossing_rate),
            "harmony_mean": np.mean(harmony),
            "harmony_var": np.var(harmony),
            "tempo_mean": np.mean(tempo),
            "tempo_var": np.var(tempo),
            "flatness_mean": np.mean(flatness),
            "flatness_var": np.mean(flatness),
            "mfcc1_mean": np.mean(mfccs[0]),
            "mfcc1_var": np.var(mfccs[0]),
            "mfcc2_mean": np.mean(mfccs[1]),
            "mfcc2_var": np.var(mfccs[1]),
            "mfcc3_mean": np.mean(mfccs[2]),
            "mfcc3_var": np.var(mfccs[2]),
            "mfcc4_mean": np.mean(mfccs[3]),
            "mfcc4_var": np.var(mfccs[3]),
            "mfcc5_mean": np.mean(mfccs[4]),
            "mfcc5_var": np.var(mfccs[4]),
            "mfcc6_mean": np.mean(mfccs[5]),
            "mfcc6_var": np.var(mfccs[5]),
            "mfcc7_mean": np.mean(mfccs[6]),
            "mfcc7_var": np.var(mfccs[6]),
            "mfcc8_mean": np.mean(mfccs[7]),
            "mfcc8_var": np.var(mfccs[7]),
            "mfcc9_mean": np.mean(mfccs[8]),
            "mfcc9_var": np.var(mfccs[8]),
            "mfcc10_mean": np.mean(mfccs[9]),
            "mfcc10_var": np.var(mfccs[9]),
            "mfcc11_mean": np.mean(mfccs[10]),
            "mfcc11_var": np.var(mfccs[10]),
            "mfcc12_mean": np.mean(mfccs[11]),
            "mfcc12_var": np.var(mfccs[11]),
            "mfcc13_mean": np.mean(mfccs[12]),
            "mfcc13_var": np.var(mfccs[12]),
            "mfcc14_mean": np.mean(mfccs[13]),
            "mfcc14_var": np.var(mfccs[13]),
            "mfcc15_mean": np.mean(mfccs[14]),
            "mfcc15_var": np.var(mfccs[14]),
            "mfcc16_mean": np.mean(mfccs[15]),
            "mfcc16_var": np.var(mfccs[15]),
            "mfcc17_mean": np.mean(mfccs[16]),
            "mfcc17_var": np.var(mfccs[16]),
            "mfcc18_mean": np.mean(mfccs[17]),
            "mfcc18_var": np.var(mfccs[17]),
            "mfcc19_mean": np.mean(mfccs[18]),
            "mfcc19_var": np.var(mfccs[18]),
            "mfcc20_mean": np.mean(mfccs[19]),
            "mfcc20_var": np.var(mfccs[19]),
            "label": label
            }
    return data