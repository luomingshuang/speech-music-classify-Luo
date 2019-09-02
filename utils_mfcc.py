import numpy as np
import scipy.io.wavfile as wav
from scipy.io.wavfile import read
from python_speech_features import mfcc, delta




def compute_mfcc(sample,sample_rate,stride_ms=10.0,window_ms=20.0, max_freq=None):
        """Compute mfcc from samples."""
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             "sample rate.")
        if stride_ms > window_ms:
            raise ValueError("Stride size must not be greater than "
                             "window size.")
        # compute the 13 cepstral coefficients, and the first one is replaced
        # by log(frame energy)
        mfcc_feat = mfcc(
            signal=sample,
            samplerate=sample_rate,
            winlen=0.001 * window_ms,
            winstep=0.001 * stride_ms,
            highfreq=max_freq)
        # Deltas
        d_mfcc_feat = delta(mfcc_feat, 2)
        # Deltas-Deltas
        dd_mfcc_feat = delta(d_mfcc_feat, 2)
        # transpose
        mfcc_feat = np.transpose(mfcc_feat)
        d_mfcc_feat = np.transpose(d_mfcc_feat)
        dd_mfcc_feat = np.transpose(dd_mfcc_feat)
        # concat above three features
        concat_mfcc_feat = np.concatenate(
            (mfcc_feat, d_mfcc_feat, dd_mfcc_feat))
        return concat_mfcc_feat

# print(mfcc_features.shape)

# mfcc_feats = compute_mfcc(signal, 16000)
# print(mfcc_feats.shape)


