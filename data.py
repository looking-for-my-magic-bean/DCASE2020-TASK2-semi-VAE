# date:2020.3
# author:tianke
import sys

import librosa
import numpy as np
import os
import scipy.io.wavfile as wav
from python_speech_features import *


def one_dimensional_data(datadir='C:/DCASE2020/', type='fan', ID ='id_00', winlen=0.064, winstep=0.032, numcep=128, ratio_train=1):
    frequency_spectrum_train = []
    frequency_spectrum_test_nor = []
    frequency_spectrum_test_anor = []
    datadir = datadir+type
    files1 = os.listdir(datadir + '/train')
    files2 = os.listdir(datadir + '/test')
    i = 0
    j = 0
    k = 0
    for everyone in files1:
        if ID in everyone:
            print(everyone)
            samplerate, train_signal = wav.read(datadir + '/train/'+everyone)
            train_frqe = mfcc(train_signal, samplerate, winlen=winlen, winstep=winstep, numcep=numcep,
             nfilt=numcep, nfft=int(16000*winlen), lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=False,
             winfunc=numpy.hamming)  # (312,128)  新 (312,128).T=(128,312) wi=0.064 ws=0.032 nu=128
            # train_frqe = train_frqe[:310, :]
            # train_frqe = np.expand_dims(train_frqe, 0)  # 在最后位置添加数据(扩维)

            if 'normal' in everyone:
                label = 1
            else:
                label = 2

            if i == 0:
                frequency_spectrum_train = train_frqe
            else:
                frequency_spectrum_train = np.concatenate((frequency_spectrum_train, train_frqe), axis=0)  # 上下拼接

            i += 1

    for everyone in files2:
        if ID in everyone:
            print(everyone)
            samplerate, test_signal = wav.read(datadir + '/test/'+everyone)
            test_frqe = mfcc(test_signal, samplerate, winlen=winlen, winstep=winstep, numcep=numcep,
             nfilt=numcep, nfft=int(16000*winlen), lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True,
             winfunc=numpy.hamming)
            # test_frqe = test_frqe[:310, :]
            # test_frqe = np.expand_dims(test_frqe, 0)  # (扩维)

            if 'normal' in everyone:
                label = 1
                if j == 0:
                    frequency_spectrum_test_nor = test_frqe
                else:
                    frequency_spectrum_test_nor = np.concatenate((frequency_spectrum_test_nor, test_frqe), axis=0)  # 上下拼接
                j += 1
            else:
                label = 2
                if k == 0:
                    frequency_spectrum_test_anor = test_frqe
                else:
                    frequency_spectrum_test_anor = np.concatenate((frequency_spectrum_test_anor, test_frqe), axis=0)  # 上下拼接
                k += 1

    data1 = np.reshape(frequency_spectrum_train, (-1, 8, 128))
    data2 = np.reshape(frequency_spectrum_test_nor, (-1, 8, 128))
    data3 = np.reshape(frequency_spectrum_test_anor, (-1, 8, 128))

    # data = np.concatenate((frequency_spectrum_train, frequency_spectrum_test), axis=0)  # 上下拼接
    # data = np.expand_dims(data, -1)  # 在最后位置添加数据

    # data1 = np.expand_dims(data_train, -1)  # 在最后位置添加数据
    # data2 = np.expand_dims(data_test_nor, -1)  # 在最后位置添加数据
    # data3 = np.expand_dims(data_test_anor, -1)  # 在最后位置添加数据

    N, D, _ = data1.shape

    # ind_cut = int(ratio_train * N)  # 训练样本与测试样本个数分界线
    # ind = np.random.permutation(N)  # 产生N个随机数，用于将样本随机化
    # return data1[ind[:ind_cut], 1:, :, :], data1[ind[ind_cut:], 1:, :, :], data2[:, 1:, :, :], data3[:, 1:, :, :], \
    #        data1[ind[:ind_cut], 0, :, :], data1[ind[ind_cut:], 0, :, :], data2[:, 0, :, :], data3[:, 0, :, :], D - 1
    # return data1[ind[:ind_cut], 1:, :, :], data1[ind[ind_cut:], 1:, :, :], data2[:, 1:, :, :], data3[:, 1:, :, :]
    return data1, data1, data2, data3


def file_to_vector_array(y, sr,  n_mels=128, frames=5, n_fft=1024, hop_length=512, power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)  # [128,313]

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array  # [309,640]


def dnn_data(datadir='C:/DCASE2020/', type='fan', ID='id_00', frames=5, numcep=128):
    frequency_spectrum_train = []
    frequency_spectrum_test_nor = []
    frequency_spectrum_test_anor = []
    datadir = datadir+type
    files1 = os.listdir(datadir + '/train')
    files2 = os.listdir(datadir + '/test')
    i = 0
    j = 0
    k = 0
    for everyone in files1:
        if ID in everyone:
            print(everyone)
            train_signal, samplerate = librosa.load(datadir + '/train/'+everyone, sr=None, mono=False)
            train_frqe = file_to_vector_array(train_signal, samplerate, n_mels=numcep, frames=frames, n_fft=1024, hop_length=512)
            # (312,128)  新 (312,128).T=(128,312) wi=0.064 ws=0.032 nu=128
            # train_frqe = train_frqe[:310, :]
            # train_frqe = np.expand_dims(train_frqe, 0)  # 在最后位置添加数据(扩维)

            if i == 0:
                frequency_spectrum_train = train_frqe
            else:
                frequency_spectrum_train = np.concatenate((frequency_spectrum_train, train_frqe), axis=0)  # 上下拼接

            i += 1

    for everyone in files2:
        if ID in everyone:
            print(everyone)
            test_signal, samplerate = librosa.load(datadir + '/test/'+everyone, sr=None, mono=False)
            test_frqe = file_to_vector_array(test_signal, samplerate, n_mels=numcep, frames=frames, n_fft=1024, hop_length=512)
            # test_frqe = test_frqe[:310, :]
            # test_frqe = np.expand_dims(test_frqe, 0)  # (扩维)

            if 'normal' in everyone:
                label = 1
                if j == 0:
                    frequency_spectrum_test_nor = test_frqe
                else:
                    frequency_spectrum_test_nor = np.concatenate((frequency_spectrum_test_nor, test_frqe), axis=0)  # 上下拼接
                j += 1
            else:
                label = 2
                if k == 0:
                    frequency_spectrum_test_anor = test_frqe
                else:
                    frequency_spectrum_test_anor = np.concatenate((frequency_spectrum_test_anor, test_frqe), axis=0)  # 上下拼接
                k += 1

    data1 = frequency_spectrum_train
    data2 = frequency_spectrum_test_nor
    data3 = frequency_spectrum_test_anor

    N, D = data1.shape

    return data1, data1, data2, data3


def dnn_data_2test(datadir='C:/DCASE2020/', type='fan', ID='id_00', frames=5, numcep=128):
    frequency_spectrum_test_nor = []
    frequency_spectrum_test_anor = []
    datadir = datadir+type
    files2 = os.listdir(datadir + '/test')
    j = 0
    k = 0
    for everyone in files2:
        if ID in everyone:
            print(everyone)
            test_signal, samplerate = librosa.load(datadir + '/test/'+everyone, sr=None, mono=False)
            test_frqe = file_to_vector_array(test_signal, samplerate, n_mels=numcep, frames=frames, n_fft=1024, hop_length=512)
            # test_frqe = test_frqe[:310, :]
            # test_frqe = np.expand_dims(test_frqe, 0)  # (扩维)

            if 'normal' in everyone:
                label = 1
                if j == 0:
                    frequency_spectrum_test_nor = test_frqe
                else:
                    frequency_spectrum_test_nor = np.concatenate((frequency_spectrum_test_nor, test_frqe), axis=0)  # 上下拼接
                j += 1
            else:
                label = 2
                if k == 0:
                    frequency_spectrum_test_anor = test_frqe
                else:
                    frequency_spectrum_test_anor = np.concatenate((frequency_spectrum_test_anor, test_frqe), axis=0)  # 上下拼接
                k += 1

    data2 = frequency_spectrum_test_nor
    data3 = frequency_spectrum_test_anor

    return data2, data3


def dnn_data_train(datadir='C:/DCASE2020/', type='fan', ID='id_00', frames=5, numcep=128):
    frequency_spectrum_train = []
    frequency_spectrum_test = []
    datadir = datadir+type
    files1 = os.listdir(datadir + '/train')
    # files2 = os.listdir(datadir + '/test')
    i = 0
    j = 0
    k = 0
    for everyone in files1:
        if ID in everyone:
            print(everyone)
            train_signal, samplerate = librosa.load(datadir + '/train/'+everyone, sr=None, mono=False)
            train_frqe = file_to_vector_array(train_signal, samplerate, n_mels=numcep, frames=frames, n_fft=1024, hop_length=512)
            # (312,128)  新 (312,128).T=(128,312) wi=0.064 ws=0.032 nu=128
            # train_frqe = train_frqe[:310, :]
            # train_frqe = np.expand_dims(train_frqe, 0)  # 在最后位置添加数据(扩维)

            if i == 0:
                frequency_spectrum_train = train_frqe
            else:
                frequency_spectrum_train = np.concatenate((frequency_spectrum_train, train_frqe), axis=0)  # 上下拼接

            i += 1

    data1 = frequency_spectrum_train
    data2 = frequency_spectrum_test

    return data1


def dnn_data_test(datadir='C:/DCASE2020/', type='fan', ID='id_00', frames=5, numcep=128):
    frequency_spectrum_train = []
    frequency_spectrum_test = []
    datadir = datadir+type
    # files1 = os.listdir(datadir + '/train')
    files2 = os.listdir(datadir + '/test')
    i = 0
    j = 0
    k = 0
    for everyone in files2:
        if ID in everyone:
            print(everyone)
            test_signal, samplerate = librosa.load(datadir + '/test/'+everyone, sr=None, mono=False)
            test_frqe = file_to_vector_array(test_signal, samplerate, n_mels=numcep, frames=frames, n_fft=1024, hop_length=512)
            # test_frqe = test_frqe[:310, :]
            # test_frqe = np.expand_dims(test_frqe, 0)  # (扩维)

            if j == 0:
                frequency_spectrum_test = test_frqe
            else:
                frequency_spectrum_test = np.concatenate((frequency_spectrum_test, test_frqe), axis=0)  # 上下拼接
            j += 1

    data1 = frequency_spectrum_train
    data2 = frequency_spectrum_test

    return data2