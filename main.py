import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# -------- Load audio files --------
audio_path_1 = "audio1.wav"
audio_path_2 = "audio2.wav"

y1, sr1 = librosa.load(audio_path_1, sr=None, mono=True)
y2, sr2 = librosa.load(audio_path_2, sr=None, mono=True)

# -------- Make same length --------
min_len = min(len(y1), len(y2))
y1 = y1[:min_len]
y2 = y2[:min_len]

# -------- FFT (Frequency-domain) --------
fft1 = np.abs(np.fft.fft(y1))
fft2 = np.abs(np.fft.fft(y2))

freqs = np.fft.fftfreq(len(fft1), 1/sr1)

# -------- Plot frequency-domain graph --------
plt.figure()
plt.plot(freqs, fft1, label="Audio 1")
plt.plot(freqs, fft2, label="Audio 2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Domain Comparison")
plt.legend()
plt.show()

# -------- Feature Extraction (MFCC) --------
mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

mfcc1_mean = np.mean(mfcc1, axis=1)
mfcc2_mean = np.mean(mfcc2, axis=1)

# -------- Similarity Calculation --------
similarity = 1 - cosine(mfcc1_mean, mfcc2_mean)

print("Voice Similarity Score:", similarity)
