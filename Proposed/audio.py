import librosa
import librosa.filters
import math
import numpy as np
from scipy import signal
from hparams import hparams
from scipy.io import wavfile
import matplotlib.pyplot as plt

import lws


def load_wav(path):
    return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))


def trim(quantized):
    start, end = start_and_end_indices(quantized, hparams.silence_threshold)
    return quantized[start:end]


def adjust_time_resolution(quantized, mel):
    """Adjust time resolution by repeating features

    Args:
        quantized (ndarray): (T,)
        mel (ndarray): (N, D)

    Returns:
        tuple: Tuple of (T,) and (T, D)
    """
    assert len(quantized.shape) == 1
    assert len(mel.shape) == 2

    upsample_factor = quantized.size // mel.shape[0]
    mel = np.repeat(mel, upsample_factor, axis=0)
    n_pad = quantized.size - mel.shape[0]
    if n_pad != 0:
        assert n_pad > 0
        mel = np.pad(mel, [(0, n_pad), (0, 0)], mode="constant", constant_values=0)

    # trim
    start, end = start_and_end_indices(quantized, hparams.silence_threshold)

    return quantized[start:end], mel[start:end, :]
adjast_time_resolution = adjust_time_resolution  # 'adjust' is correct spelling, this is for compatibility


def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end



def linearspectrogram(y):
    y_pad = np.pad(y, (0, hparams.fft_size-get_hop_size()), mode="constant", constant_values=0)
    D = librosa.stft(y=y_pad, n_fft=hparams.fft_size, hop_length=get_hop_size(), win_length=hparams.fft_size,
                     center=False, pad_mode='constant')
    S = _amp_to_db(np.abs(D)) - hparams.ref_level_db_linear
    # plt.figure(17)
    # temp = np.reshape(S, (np.product(S.shape),))
    # plt.hist(temp, normed=True, bins=30)
    # plt.ylabel('Probability')
    # plt.show()
    if not hparams.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - hparams.min_level_db_linear >= 0
    # NOTE in librosa.stft, the real phase at each TF unit is reversed to match with DPWE code, now we change it back
    return _normalize_linear(S), -np.angle(D)


def melspectrogram(y):
    D = _lws_processor().stft(y).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
    if not hparams.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    return _normalize(S)


def get_hop_size():
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size


def _lws_processor():
    return lws.lws(hparams.fft_size, get_hop_size(), mode="speech")


def lws_num_frames(length, fsize, fshift):
    """Compute number of time frames of lws spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def lws_pad_lr(x, fsize, fshift):
    """Compute left and right padding lws internally uses
    """
    M = lws_num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


_linear_basis = None

def _mel_to_linear(spectrogram):
    global _linear_basis, _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()

    if _linear_basis is None:
        _linear_basis = np.linalg.pinv(_mel_basis)

        # plt.figure(1)
        # ax1 = plt.subplot(211)
        # ax1.plot(_mel_basis[0::5].T)
        # ax2 = plt.subplot(212)
        # ax2.plot(_linear_basis[0::10].T)
        # plt.show()
    return np.clip(np.dot(_linear_basis, spectrogram),0,a_max=None)


def _build_mel_basis():
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.fft_size,
                               fmin=hparams.fmin, fmax=hparams.fmax,
                               n_mels=hparams.num_mels)


def _amp_to_db(x):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize_linear(S):
    return np.clip((S - hparams.min_level_db_linear) / -hparams.min_level_db_linear, 0, 1)

def _denormalize_linear(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db_linear) + hparams.min_level_db_linear


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db









def stft_for_reconstruction(x, fft_size, hopsamp):
    """Compute and return the STFT of the supplied time domain signal x.
    Args:
        x (1-dim Numpy array): A time domain signal.
        fft_size (int): FFT size. Should be a power of 2, otherwise DFT will be used.
        hopsamp (int):
    Returns:
        The STFT. The rows are the time slices and columns are the frequency bins.
    """
    window = np.hanning(fft_size)
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    return np.array([np.fft.rfft(window*x[i:i+fft_size])
                     for i in range(0, len(x)-fft_size, hopsamp)])


def istft_for_reconstruction(X, fft_size, hopsamp):
    """Invert a STFT into a time domain signal.
    Args:
        X (2-dim Numpy array): Input spectrogram. The rows are the time slices and columns are the frequency bins.
        fft_size (int):
        hopsamp (int): The hop size, in samples.
    Returns:
        The inverse STFT.
    """
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    window = np.hanning(fft_size)
    time_slices = X.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n,i in enumerate(range(0, len(x)-fft_size, hopsamp)):
        x[i:i+fft_size] += window*np.real(np.fft.irfft(X[n]))
    return x

def reconstruct_signal_griffin_lim(magnitude_spectrogram, fft_size, hopsamp, iterations):
    """Reconstruct an audio signal from a magnitude spectrogram.
    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the paper:
    "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
    in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.
    Args:
        magnitude_spectrogram (2-dim Numpy array): The magnitude spectrogram. The rows correspond to the time slices
            and the columns correspond to frequency bins.
        fft_size (int): The FFT size, which should be a power of 2.
        hopsamp (int): The hope size in samples.
        iterations (int): Number of iterations for the Griffin-Lim algorithm. Typically a few hundred
            is sufficient.
    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    time_slices = magnitude_spectrogram.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    # Initialize the reconstructed signal to noise.
    x_reconstruct = np.random.randn(len_samples)
    n = iterations # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        reconstruction_spectrogram = stft_for_reconstruction(x_reconstruct, fft_size, hopsamp)
        reconstruction_angle = np.angle(reconstruction_spectrogram)
        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram*np.exp(1.0j*reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = istft_for_reconstruction(proposal_spectrogram, fft_size, hopsamp)
        diff = np.sqrt(sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        # print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct
