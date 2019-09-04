import os
os.environ['OPENBLAS_NUM_THREADS'] = '0,1'
import time
import torch
import torch.nn as nn
from torch import optim

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from test_data import MyDataset_test

from model import speech_music_classify

if(__name__ == '__main__'):
    torch.manual_seed(55)
    torch.cuda.manual_seed_all(55)
    opt = __import__('options')

speech_classify_path = './speech_classify.txt'
music_classify_path = './music_classify.txt'
speech_file = open(speech_classify_path， 'w')
music_file = open(music_classify_path， 'w')

def data_from_test():
    dataset = MyDataset_test()
    print('num_data:{}'.format(len(dataset.data)))
    
    loader = DataLoader(dataset, 
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        drop_last=False,
        shuffle=True)   
    return (dataset, loader)

if(__name__ == '__main__'):
    model = speech_music_classify().cuda()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        # print(model_dict.keys())
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)       

    (test_dataset, test_loader) = data_from_test()
    #(tst_dataset, tst_loader) = data_from_opt(opt.tst_txt_path, 'test')
 
    iteration = 0
    m = 0
    for epoch in range(opt.max_epoch):
        start_time = time.time()
        exp_lr_scheduler.step()

        for (i, batch) in enumerate(train_loader):
            #print(batch['inputs'].size())
            (names, inputs, label) = batch['name'], batch['inputs'].cuda(), batch['label']
            hidden = model(inputs)            
            _, predict_txt = torch.max(F.softmax(hidden, dim=1).data, 1)
            for idx in range(opt.batch_size):
                if predict_txt[idx] == 0:
                    speech_file.write(names[idx] + '\n')
                else:
                    music_file.write(names[idx] + '\n')