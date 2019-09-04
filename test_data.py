class MyDataset_test(Dataset):
    def __init__(self):
        self.data = []
        test_files = glob.glob(os.path.join(path, 'test_5000', '*.wav'))

        for wav_file in test_files:
            fs, signal = wav.read(speech)
            if len(signal) < 180000:
            #     print(speech)
                signal = padding(signal)
                mfcc_features = compute_mfcc(signal, 16000)
                self.data.append((wav_file, mfcc_features, 0))
            else:
                signal_i = signal[0:180000]
                mfcc_features = compute_mfcc(signal_i, 16000)
                self.data.append((wav_file, mfcc_features, 0))
        print('the number of test_files for testing: {}'.format(len(self.data)))
        speech_data = len(self.data)
    
    def __getitem__(self, idx):
        (wav_file, mfcc_features, label) = self.data[idx]
        #print(mfcc_features.shape)
        return {'name':wav_file, 'inputs':torch.FloatTensor(mfcc_features), 'label':label}
            
    def __len__(self):
        return len(self.data)