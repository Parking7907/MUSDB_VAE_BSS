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
def get_data_loaders(data_dir,save_path,batch_size,data_size,data_len,kwargs):
    #data_dir = '/home/data/jinyoung/source_separation/IRMAS/spectrogram/'
    train_set = MUSDB(data_dir, save_path, data_partition = 'train', time_duration = 128, sample_rate=44100, target = 'vocal', random_mix = True, seed=1234, data_size = data_size, data_len = data_len)
    test_set = MUSDB(data_dir, save_path, data_partition = 'test', time_duration = 128, sample_rate=44100, target = 'vocal', random_mix = True, seed=1234, data_size = data_size, data_len = data_len)
    #pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


class MUSDB(Dataset):
    def __init__(self, data_dir, save_path, data_partition = 'train', time_duration = 128, sample_rate=44100, target = 'vocal', random_mix = True, seed=1234, data_size = 256, data_len =32):
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
        self.save_path = save_path
        if data_partition == 'train':
            self.data_path = data_dir + 'train/'
        elif data_partition == 'test':
            self.data_path = data_dir + 'test/'
        data_path = self.data_path + "*/*"
        win_len = 2048
        hop_length = 512
        n_fft = 2048
        
        self.window=torch.hann_window(window_length=win_len, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    
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
        return len(self.music_list)

    def __getitem__(self, idx):
        save_path = self.save_path
        data_path = self.music_list[idx]
        print(data_path)
        window = self.window
        win_len = 2048
        hop_length = 512
        n_fft = 2048
        music_n = data_path.split('/')[-1] #'008__[cel][nod][cla]0058__1'
        dir_n = data_path.split('/')[-2] #'cel'
        music_n = music_n.split('.wav')[0] # bass
        vocal_signal, vocal_fs = torchaudio.load(data_path) # 2, 132299
        vocal_spectrogram = torchaudio.functional.spectrogram(waveform=vocal_signal, pad=0, window=window, n_fft=n_fft, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False)#, return_complex = False)
        #vocal_spectrogram = 2 X 1025 X 259
        vocal_real_0 = vocal_spectrogram[0, :256, :] # 256, 259

        
        vocal_real_0 = vocal_real_0.numpy()
        #print(vocal_real_0.shape) # 256, 259
        if self.data_partition == 'train':
            os.makedirs(save_path + 'train/' + dir_n + '/', exist_ok=True)
            print(save_path + 'train/' + dir_n + '/')
            out_dir = save_path + 'train/' + dir_n + '/' + music_n + '.npy'
        elif self.data_partition == 'test':
            os.makedirs(save_path + 'test/' + dir_n + '/', exist_ok=True)
            print(save_path + 'test/' + dir_n + '/')
            out_dir = save_path + 'test/' + dir_n + '/' + music_n + '.npy'
        
        #print("outdir =", out_dir)
        #pdb.set_trace()
        np.save(out_dir, vocal_real_0)
        if idx % 100 ==0 :
            print("Done %i/%i"%(idx, len(self.music_list)))
        return idx


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

print("new")

data_dir = '/home/data/musdb/'
batch_size = 16
kwargs = {'num_workers': 80, 'pin_memory': True}
data_size = 256
data_len = 32
save_path = '/home/data/musdb/spectrogram/'
train_loader, test_loader = get_data_loaders(data_dir, save_path, batch_size, data_size, data_len, kwargs)
'''
for idx in train_loader:
    print(idx)
'''
for idx in test_loader:
    print(idx)
#pdb.set_trace()
print("done")
