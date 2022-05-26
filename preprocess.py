import sys
sys.path.append('./')
import argparse
import pdb
import argparse
import yaml 
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np
from tqdm import tqdm
#from train import Trainer
#from test import total_test
#from setup import setup_solver

import torch
import torchaudio
import torchaudio.functional
import logging
from glob import glob
data_path = '/home/data/jinyoung/musdb/train/*/*.wav'
#/home/data/jinyoung/source_separation/Libri2Mix/wav16k/max/train-360/s1/
music_list = glob(data_path)
music_list.sort()
data_path2 = '/home/data/jinyoung/musdb/test/*/*.wav'
music_list2 = glob(data_path2)
music_list2.sort()
#pdb.set_trace()
print("music list :", len(music_list), len(music_list2))
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 기본적으로 44.1kHz, => Win_len = 40ms, n_fft = win_len보다 크게. hop_len = 10ms.. 굳이?
win_len = 2048
hop_length = 512
n_fft = 2048
save_path = '/home/data/jinyoung/musdb/spectrogram/'
window=torch.hann_window(window_length=win_len, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False)
sampling = 44100
source_list = ['bass.wav', 'drums.wav', 'mixture.wav', 'other.wav', 'vocals.wav']
pkl_list = ['bass.pkl', 'drums.pkl', 'mixture.pkl', 'other.pkl', 'vocals.pkl']
i= 0 

for music_name in music_list:
    #if "dru" in music_name:
    #    print("pass by", music_name)
    #    continue
    music_n = music_name.split('/')[-1] #'008__[cel][nod][cla]0058__1'
    dir_n = music_name.split('/')[-2] #'cel'
    music_n = music_n.split('.wav')[0] # bass
    vocal_signal, vocal_fs = torchaudio.load(music_name) # 2, 132299
    vocal_spectrogram = torchaudio.functional.spectrogram(waveform=vocal_signal, pad=0, window=window, n_fft=n_fft, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False)#, return_complex = False)
    #vocal_spectrogram = 2 X 1025 X 259
    vocal_real_0 = vocal_spectrogram[0, :256, :] # 256, 259

    
    vocal_real_0 = vocal_real_0.numpy()
    #print(vocal_real_0.shape) # 256, 259
    os.makedirs(save_path + 'train/' + dir_n + '/', exist_ok=True)
    out_dir = save_path + 'train/' + dir_n + '/' + music_n + '.npy'
    
    #print("outdir =", out_dir)
    #pdb.set_trace()
    np.save(out_dir, vocal_real_0)
    if i % 100 ==0 :
        print("Done %i/%i"%(i, len(music_list)))
    i+=1

for music_name in music_list2:
    music_name_list = music_name.split('/')
    music_n = music_name.split('/')[-1] #'008__[cel][nod][cla]0058__1'
    #dir_n = music_name.split('/')[-2] #'cel'
    music_n = music_n.split('.wav')[0]
    label_name = ('/').join(music_name_list[:-1]) + '/' + music_n + '.txt'
    with open(label_name, 'r') as f:
        line_ = f.read()
    dir_n = line_.split('\t')[0]
    #print(dir_n)
    vocal_signal, vocal_fs = torchaudio.load(music_name) # 2, 132299
    vocal_spectrogram = torchaudio.functional.spectrogram(waveform=vocal_signal, pad=0, window=window, n_fft=n_fft, hop_length=int(win_len/4), win_length=win_len, power=None, normalized=False)#, return_complex = False)
    #vocal_spectrogram = 2 X 1025 X 259
    
    vocal_real_0 = vocal_spectrogram[0, :256, :] # 256, 259
    vocal_real_0 = vocal_real_0.numpy()
    os.makedirs(save_path + 'test/' + dir_n + '/', exist_ok=True)
    out_dir = save_path + 'test/' + dir_n + '/' + music_n + '.npy'
    #print("outdir =", out_dir)
    #pdb.set_trace()
    np.save(out_dir, vocal_real_0)
    if i % 1000 ==0 :
        print("Done %i/%i"%(i, len(music_list2)))
    i+=1