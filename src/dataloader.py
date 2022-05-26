from calendar import c
from email.errors import ObsoleteHeaderDefect
from socketserver import DatagramRequestHandler
import sys
sys.path.append('..')
import torch

from torch.utils.data.dataset import Dataset
#from torchvision.transforms import Normalize
from pathlib import Path
#import soundfile as sf
import pickle
import pdb
#import cv2
import numpy as np
import torchvision
import os
import random
from glob import glob
#from augmentation import augmentation
import torchaudio
import torch.nn.functional as F
# import albumentations as A
def get_data_loaders(data_dir,batch_size,data_size,data_len,kwargs):
    #data_dir = '/home/data/jinyoung/source_separation/IRMAS/spectrogram/'
    train_set = MUSDB(data_dir, data_partition = 'train', time_duration = 128, sample_rate=44100, target = 'vocal', random_mix = True, seed=1234, data_size = data_size, data_len = data_len)
    test_set = MUSDB(data_dir, data_partition = 'test', time_duration = 128, sample_rate=44100, target = 'vocal', random_mix = True, seed=1234, data_size = data_size, data_len = data_len)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


class MUSDB(Dataset):
    def __init__(self, data_dir, data_partition = 'train', time_duration = 128, sample_rate=44100, target = 'vocal', random_mix = True, seed=1234, data_size = 256, data_len =32):
        super(MUSDB, self).__init__()
        #data_path = /home/data/jinyoung/source_separation/Libri2Mix/wav16k/max/train-360/s1,s2,mix_clean,mix_both,mix_single
        self.data_dir = data_dir
        self.data_partition = data_partition    
        self.duration_len = time_duration
        self.sample_rate = sample_rate
        self.sources_bass = []
        self.sources_drum = []
        self.sources_mixture = []
        self.sources_others = []
        self.sources_vocal = []
        self.data_size = data_size
        self.data_len = data_len
        if data_partition == 'train':
            self.data_path = data_dir + 'train/'
        else:
            self.data_path = data_dir + 'test/'
        data_path = self.data_path + "*"
        
        #source1_data_path =data_path + "/s1/*/"
        #source2_data_path = data_path + "/s2/*/"
        #print(data_path)
        self.music_list = glob(data_path)
        self.music_list.sort()
        print(data_path, len(self.music_list))
        self.seed = seed
        self.target = target
        self.random_mix = random_mix
        #print(self.music_list)
        i = 0
        #Seed 고정.. Required?
        #random.seed(self.seed)
        #Read data path
            
    def __len__(self):
        if self.data_partition == 'train':
            return len(self.music_list) * 10
        elif self.data_partition == 'test':
            return len(self.music_list)

    def __getitem__(self, idx):
        idx = idx % len(self.music_list)
        data_path = self.music_list[idx] + "/mixture.npy"
        data = np.load(data_path)
        '''
        if self.data_partition == 'test':
            s1_p = self.music_list[idx] + "/bass.npy"
            s2_p = self.music_list[idx] + "/drums.npy"
            s3_p = self.music_list[idx] + "/other.npy"
            s4_p = self.music_list[idx] + "/vocals.npy"
            
            source_1 = np.load(s1_p)
            source_2 = np.load(s2_p)
            source_3 = np.load(s3_p)
            source_4 = np.load(s4_p)
                
            if source_1.ndim ==3:
                source_1_real = source_1[:,:,0] + source_1[:,:,1] * 1j
                source_1 = np.absolute(source_1_real)
            else:
                source_1 = np.absolute(source_1)
            if source_2.ndim ==3:
                source_2_real = source_2[:,:,0] + source_2[:,:,1] * 1j
                source_2 = np.absolute(source_2_real)
            else:
                source_2 = np.absolute(source_2)
                
            if source_3.ndim ==3:
                source_3_real = source_3[:,:,0] + source_3[:,:,1] * 1j
                source_3 = np.absolute(source_3_real)
            else:
                source_3 = np.absolute(source_3)
            if source_4.ndim ==3:
                source_4_real = source_4[:,:,0] + source_4[:,:,1] * 1j
                source_4 = np.absolute(source_4_real)
            else:
                source_4 = np.absolute(source_4)
            source_1 = source_1[:self.data_size, :]
            source_2 = source_2[:self.data_size, :]
            source_3 = source_3[:self.data_size, :]
            source_4 = source_4[:self.data_size, :]
            source_1 = torch.Tensor(source_1)
            source_2 = torch.Tensor(source_2)
            source_3 = torch.Tensor(source_3)
            source_4 = torch.Tensor(source_4)
        '''
            
        if data.ndim == 3:
            data_real = data[:,:,0] + data[:,:,1] * 1j
            data = np.absolute(data_real)
        else:
            data = np.absolute(data)
        #print(data.shape, source_1.shape, source_2.shape)

        #Output size 문제 때문에....
        data = data[:self.data_size, :]
        data = torch.Tensor(data)
        if len(data[0]) - self.data_len -1 < 0 :
            start = 0
            #print(data.shape, source_1.shape, source_2.shape)
            pad_to = (0,self.data_len - len(data[0]))
            data = F.pad(data, pad_to, "constant", 0)
            '''
            if self.data_partition == 'test':
                source_1 = F.pad(source_1, pad_to, "constant", 0)
                source_2 = F.pad(source_2, pad_to, "constant", 0)
                source_3 = F.pad(source_3, pad_to, "constant", 0)
                source_4 = F.pad(source_4, pad_to, "constant", 0)
            '''
            #print(data.shape, source_1.shape, source_2.shape)
        else:
            start = random.randint(0, len(data[0]) - self.data_len -1)
            data = data[:, start:start + self.data_len]
            '''
            if self.data_partition == 'test':
                source_1 = source_1[:, start:start + self.data_len]
                source_2 = source_2[:, start:start + self.data_len]
                source_3 = source_3[:, start:start + self.data_len]
                source_4 = source_4[:, start:start + self.data_len]
            '''
        
        source_out = self.normalize(data)
        #print(start, source_out.shape, source_1.shape, source_2.shape)
        #print(torch.max(source_1), torch.max(source_2), torch.max(source_out), torch.min(source_1), torch.min(source_2), torch.min(source_out))
        if self.data_partition == 'train':
            return source_out
        elif self.data_partition == 'test':
            return self.music_list[idx],start,source_out


    def normalize(self, data):
        #print("Norm")
        

        data = data.float()
        #mean = torch.mean(data.float())
        #std = torch.std(data.float())
        #data = (data - mean) / std
        max_ = torch.max(data)
        min_ = torch.min(data)
        if max_ == 0 or min_ == 0:
            pass
            #print("NULL!!!!!!!!!!!!!!!!!!!!!!!!..", max_, min_)
        else:
            if max_ > (-1 * min_):
                data = data / (max_)
            else:
                data = data / (-1 * min_)
        #print(max_, min_, torch.max(data), torch.min(data), vocal_idx, bass_idx, drum_idx, others_idx)
        return data
