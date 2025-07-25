import numpy as np
import os
from numpy import inf
import librosa

def stft(data, fft_size=512, step=160, padding=True):
    if padding is True:
        pd = np.zeros(192, )
        data = np.concatenate((data, pd), axis=0)
    windows = np.concatenate((np.zeros((56,)), np.hanning(fft_size - 112), np.zeros((56,))), axis=0)
    windows_num = (len(data) - fft_size) // step
    output = np.ndarray((windows_num, fft_size), dtype=data.dtype)
    for window in range(windows_num):
        start = int(window * step)
        end = int(start + fft_size)
        output[window] = data[start:end] * windows
    M = np.fft.rfft(output, axis=1)
    return M


def istft(M, fft_size=512, step=160, padding=True):
    # Inverse FFT along the last axis of M
    data = np.fft.irfft(M, n=fft_size, axis=-1)  # Use irfft to handle real-valued output with the correct length
    window = np.concatenate((np.zeros((56,)), np.hanning(fft_size - 112), np.zeros((56,))), axis=0)
    windows_num = M.shape[0]
    total_length = step * (windows_num - 1) + fft_size
    Total = np.zeros(total_length)
    # Overlap-add process to reconstruct the time-domain signal
    for i in range(windows_num):
        start = i * step
        end = start + fft_size
        Total[start:end] += data[i] * window
    if padding:
        Total = Total[:48000]

    return Total


def power_law(data, power=0.3):
    mask = np.zeros((data.shape))
    mask[data > 0] = 1
    mask[data < 0] = -1
    data = np.power(np.abs(data), power)
    data = data * mask
    return data

def real_imag_expand(c_data,dim='new'):
    if dim == 'new':
        D = np.zeros((c_data.shape[0],c_data.shape[1],2))
        D[:,:,0] = np.real(c_data)
        D[:,:,1] = np.imag(c_data)
        return D
    if dim =='same':
        D = np.zeros((c_data.shape[0],c_data.shape[1]*2))
        D[:,::2] = np.real(c_data)
        D[:,1::2] = np.imag(c_data)
        return D

def real_imag_shrink(M, dim='new'):
    M_shrink = np.zeros((M.shape[0], M.shape[1]))
    if dim == 'new':
        M_shrink = M[:, :, 0] + M[:, :, 1] * 1j
    if dim == 'same':
        M_shrink = M[:, ::2] + M[:, 1::2] * 1j
    return M_shrink


def fast_stft(data, power=False):
    if power:
        data = power_law(data)

    return real_imag_expand(stft(data))


def fast_istft(data, power=False):
    data = istft(real_imag_shrink(data))
    if power:
        data = power_law(data, (1.0 / 0.3))
    return data


def generate_cRM(Y, S):
    M = np.zeros(Y.shape)
    epsilon = 1e-8
    # real part
    M_real = np.multiply(Y[:, :, 0], S[:, :, 0]) + np.multiply(Y[:, :, 1], S[:, :, 1])
    square_real = np.square(Y[:, :, 0]) + np.square(Y[:, :, 1])
    M_real = np.divide(M_real, square_real + epsilon)
    M[:, :, 0] = M_real
    # imaginary part
    M_img = np.multiply(Y[:, :, 0], S[:, :, 1]) - np.multiply(Y[:, :, 1], S[:, :, 0])
    square_img = np.square(Y[:, :, 0]) + np.square(Y[:, :, 1])
    M_img = np.divide(M_img, square_img + epsilon)
    M[:, :, 1] = M_img
    return M


def cRM_tanh_compress(M, K=10, C=0.1):
    numerator = 1 - np.exp(-C * M)
    numerator[numerator == inf] = 1
    numerator[numerator == -inf] = -1
    denominator = 1 + np.exp(-C * M)
    denominator[denominator == inf] = 1
    denominator[denominator == -inf] = -1
    crm = K * np.divide(numerator, denominator)

    return crm


def cRM_tanh_recover(O, K=10, C=0.1):
    numerator = K - O
    denominator = K + O
    M = -np.multiply((1.0 / C), np.log(np.divide(numerator, denominator)))

    return M


def fast_cRM(Fclean, Fmix, K=10, C=0.1):
    M = generate_cRM(Fmix, Fclean)
    crm = cRM_tanh_compress(M, K, C)
    return crm


def fast_icRM(Y, crm, K=10, C=0.1):
    M = cRM_tanh_recover(crm, K, C)
    S = np.zeros(np.shape(M))
    S[:, :, 0] = np.multiply(M[:, :, 0], Y[:, :, 0]) - np.multiply(M[:, :, 1], Y[:, :, 1])
    S[:, :, 1] = np.multiply(M[:, :, 0], Y[:, :, 1]) + np.multiply(M[:, :, 1], Y[:, :, 0])
    return S

def generate_spectrogram_magphase(audio, stft_frame, stft_hop, with_phase=True):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)
    spectro_mag = np.expand_dims(spectro_mag, axis=0)
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
        return spectro_mag, spectro_phase
    else:
        return spectro_mag

import librosa
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg') 
plt.ioff()

def visualizeSpectrogram(spectrogram, save_path):
	fig,ax = plt.subplots(1,1)
	plt.axis('off')
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	plt.pcolormesh(librosa.amplitude_to_db(spectrogram))
	plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
	plt.close()
