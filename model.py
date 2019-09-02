#coding:utf-8
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from python_speech_features import mfcc, delta
import options as opt

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda()
        _, hidden = self.gru(x, h0)
        hidden = hidden.cuda()
        #print(hidden.size())
        hidden_out = hidden[-2:]
        #print(hidden_out.size())
        hidden_out = hidden_out.transpose(1,0)
        #print(hidden_out.size())
        hidden_out = hidden_out.reshape(-1, self.hidden_size*2)
        #print(hidden_out.size())
        # predicitions based on every time step
        out = self.fc(hidden_out)  # predictions based on the last time step
        return  out


class Speech_Music_Classify(nn.Module):
    def __init__(self, inputDim=256, hiddenDim=512, nClasses=500, frameLen=29):
        super(Speech_Music_Classify, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = 2
        # frontend1D
        self.fronted1D = nn.Sequential(
                nn.Conv1d(39, 39, kernel_size=4, stride=2, bias=False),
                nn.BatchNorm1d(39),
                nn.ReLU(True),
                nn.Conv1d(39, 39, kernel_size=2, stride=1, bias=False),
                nn.BatchNorm1d(39),
                nn.ReLU(True),
                nn.Conv1d(39, 39, kernel_size=4, stride=2, bias=False),
                nn.BatchNorm1d(39),
                nn.ReLU(True),
                nn.Conv1d(39, 39, kernel_size=4, stride=2, bias=False),
                nn.BatchNorm1d(39),
                nn.ReLU(True),
                nn.Conv1d(39, 39, kernel_size=4, stride=2, bias=False),
                nn.BatchNorm1d(39),
                nn.ReLU(True),
                nn.Conv1d(39, 39, kernel_size=4, stride=2, bias=False),
                nn.BatchNorm1d(39),
                nn.ReLU(True)
                #nn.Conv1d
                )
        # fc_layers
        self.fc = nn.Sequential(
            nn.Linear(39, 112),
            nn.BatchNorm1d(112),
            nn.ReLU(True),
            nn.Linear(112, 256))
        # backend_gru
        self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses)
        # initialize
        self._initialize_weights()
        
    def forward(self, x):
        m = Variable(torch.zeros(opt.batch_size, 33, 256)).cuda()
        #hidden = Variable(torch.zeros(self.nLayers*2, x.size(0), self.hiddenDim)).cuda(0)
        #x = x.view(-1, x.size(0), x.size(1))
        #print(x.size())
        x = x.view(-1, x.size(1), x.size(2))
        x = self.fronted1D(x)
        x = x.contiguous()
        #print('fronted1D: ', x.size())
        x = x.transpose(2,1).contiguous()
        x = x.view(-1, x.size(1), x.size(2))
        #print(x.size())
        for i in range(opt.batch_size):
            #print(x[i].size())
            m[i] = self.fc(x[i])
            #print(x[i].size())
        #x = self.fc()
        #print('fc: ', x.size())
        x = m.view(-1, self.frameLen, self.inputDim)
        x = self.gru(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def speech_music_classify(inputDim=256, hiddenDim=512, nClasses=2, frameLen=33):
    model = Speech_Music_Classify(inputDim=inputDim, hiddenDim=hiddenDim, nClasses=nClasses, frameLen=frameLen)
    return model


# model = speech_music_classify()

# import scipy.io.wavfile as wav
# from scipy.io.wavfile import read
# from utils_mfcc import compute_mfcc

# wav_name = './11_1.wav'

# fs, signal = wav.read(wav_name)
# print(len(signal))
# mfcc_features = compute_mfcc(signal, 16000)
# print('inputs: ',mfcc_features.shape)

# signal = torch.FloatTensor(mfcc_features)

# hidden = model(signal)

# print(hidden.size())