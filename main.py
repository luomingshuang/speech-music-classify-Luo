import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
import tensorboardX
from data import MyDataset
from model import speech_music_classify
import torch.nn as nn
from torch import optim

from tensorboardX import SummaryWriter
import tensorflow as tf

if(__name__ == '__main__'):
    torch.manual_seed(55)
    torch.cuda.manual_seed_all(55)
    opt = __import__('options')


def data_from_opt():
    dataset = MyDataset()
    print('num_data:{}'.format(len(dataset.data)))
    
    loader = DataLoader(dataset, 
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        drop_last=False,
        shuffle=True)   
    return (dataset, loader)

if(__name__ == '__main__'):
    model = speech_music_classify().cuda()
    
    writer = SummaryWriter()

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

    (train_dataset, train_loader) = data_from_opt()
    #(tst_dataset, tst_loader) = data_from_opt(opt.tst_txt_path, 'test')
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
             lr=opt.lr,
             weight_decay=0)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    iteration = 0
    for epoch in range(opt.max_epoch):
        start_time = time.time()
        exp_lr_scheduler.step()

        for (i, batch) in enumerate(train_loader):
            #print(batch['inputs'].size())
            (inputs, label) = batch['inputs'].cuda(), batch['label'].cuda()
            inputs = inputs.cuda(0)
            hidden = model(inputs)            
            loss = criterion(hidden, label)

            optimizer.zero_grad()   
            iteration += 1

            loss.backward()
            optimizer.step()
            tot_iter = epoch*len(train_loader)+i
            
            train_loss = loss.item()
            print('iteration:%d, epoch:%d, train_loss:%.6f'%(iteration, epoch, train_loss))
            n = int(len(train_dataset.data) / opt.batch_size)
            if (i+1)==n:
                break