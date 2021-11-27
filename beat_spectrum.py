from librosa import beat
import numpy as np
from pathlib import Path
import librosa
import scipy.fftpack as scifft
import matplotlib.pyplot as plt

sources = [x for x in Path('./audio').rglob('*.wav')]

def beat_spectrum(path:str):
    y, sr = librosa.load(path)
    fft = np.abs(librosa.stft(y))
    power_spectrogram = librosa.amplitude_to_db(fft, ref=np.max)
    # ssm = np.dot(fft, fft.transpose)

    freq_bins, time_bins = power_spectrogram.shape

    # row-wise autocorrelation according to the Wiener-Khinchin theorem
    power_spectrogram = np.vstack([power_spectrogram, np.zeros_like(power_spectrogram)])

    nearest_power_of_two = 2 ** np.ceil(np.log(power_spectrogram.shape[0]) / np.log(2))
    pad_amount = int(nearest_power_of_two - power_spectrogram.shape[0])
    power_spectrogram = np.pad(power_spectrogram, ((0, pad_amount), (0, 0)), 'constant')
    fft_power_spec = scifft.fft(power_spectrogram, axis=0)
    abs_fft = np.abs(fft_power_spec) ** 2
    autocorrelation_rows = np.real(
        scifft.ifft(abs_fft, axis=0)[:freq_bins, :])  # ifft over columns

    # normalization factor
    norm_factor = np.tile(np.arange(freq_bins, 0, -1), (time_bins, 1)).T
    autocorrelation_rows = autocorrelation_rows / norm_factor

    # compute the beat spectrum
    beat_spectrum = np.mean(autocorrelation_rows, axis=1)
    print(beat_spectrum)
    # average over frequencies
    fig, ax = plt.subplots()
    t = np.arange(0.0, 1025, 1)
    plt.title(str(path))
    ax.plot(t, beat_spectrum)
    plt.show()
    return beat_spectrum 

for src in sources:
    beat_spectrum(src)