# encoding: utf-8
import os
import glob
import torch
import numpy as np
import scipy.io.wavfile as wav
from scipy.io.wavfile import read
from utils_mfcc import compute_mfcc
from torch.utils.data import Dataset

def padding(array):
    array = [array[_] for _ in range(array.shape[0])]
    size = array[0].shape
    for i in range(180000 - len(array)):
        array.append(np.zeros(size))
    return np.stack(array, axis=0)

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        path = '/home/lms/Documents/speech-music-classify-lms'
        speech_wav_files = []
        speech_wav_files_xinwen = glob.glob(os.path.join(path, 'speech', '*.wav'))
        speech_wav_files.append(speech_wav_files_xinwen)
        speech_wav_files_timit = 
        music_wav_files = glob.glob(os.path.join(path, 'music', '*.wav'))
   
        ##speech label: 0
        for speech in speech_wav_files:
            fs, signal = wav.read(speech)
            if len(signal) < 180000:
            #     print(speech)
                signal = padding(signal)
            else:
                signal = signal
            n = int(len(signal)/180000)
            #print(n)
            for i in range(n):
                signal_i = signal[i*180000:(i+1)*180000]
                mfcc_features = compute_mfcc(signal_i, 16000)
                self.data.append((mfcc_features, 0))
        print('the number of speech for training: {}'.format(len(self.data)))
        speech_data = len(self.data)
        
        ##music label: 1
        for music in music_wav_files:
            fs, signal = wav.read(music)
            if len(signal) < 180000:
            #     print(speech)
                signal = padding(signal)
            else:
                signal = signal
            n = int(len(signal)/180000)
            #print(n)
            for i in range(n):
                signal_i = signal[i*180000:(i+1)*180000]
                mfcc_features = compute_mfcc(signal_i, 16000)
                self.data.append((mfcc_features, 1))
        print('the number of music for training: {}'.format(len(self.data)-speech_data)) 
    
    def __getitem__(self, idx):
        (mfcc_features, label) = self.data[idx]
        #print(mfcc_features.shape)
        return {'inputs':torch.FloatTensor(mfcc_features), 'label':label}
            
    def __len__(self):
        return len(self.data)

