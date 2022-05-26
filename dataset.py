from calendar import c
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
# import albumentations as A
class Libri2Mix(Dataset):
    def __init__(self, data_dir, data_partition, duration_len, sample_rate=44100, target = 'vocal', random_mix = False, seed=1234):
        super(Libri2Mix, self).__init__()
        #data_path = /home/data/jinyoung/source_separation/Libri2Mix/wav16k/max/train-360/s1,s2,mix_clean,mix_both,mix_single
        self.data_partition = data_partition    
        self.duration_len = duration_len
        self.sample_rate = sample_rate
        self.sources_bass = []
        self.sources_drum = []
        self.sources_mixture = []
        self.sources_others = []
        self.sources_vocal = []
        data_path = data_dir + "/*/"
        print(data_path)
        self.music_list = glob(data_path)
        self.music_list.sort()
        self.data_len = len(self.music_list)
        self.seed = seed
        self.target = target
        self.random_mix = random_mix
        print(self.music_list)
        i = 0
        #Seed 고정.. Required?
        #random.seed(self.seed)
        
        
        #Read data path
        for music in self.music_list:
            bass_n = os.path.join(music, 'bass.wav')
            drum_n = os.path.join(music, 'drums.wav')
            mixture_n = os.path.join(music, 'mixture.wav')
            others_n = os.path.join(music, 'other.wav')
            vocal_n = os.path.join(music, 'vocals.wav')
            #print(bass_n)
            self.sources_bass.append(bass_n)
            self.sources_drum.append(drum_n)
            self.sources_mixture.append(mixture_n)
            self.sources_others.append(others_n)
            self.sources_vocal.append(vocal_n)
            i += 1
        #print(self.sources_bass)
        print('%i complete %i'%(i, len(self.sources_bass)))                
        #pdb.set_trace()
    def __len__(self):
        return len(self.music_list)

    def __getitem__(self, idx):
        if self.random_mix:
            vocal_idx = random.randint(0,self.data_len - 1)
            bass_idx = random.randint(0,self.data_len - 1)
            drum_idx = random.randint(0,self.data_len - 1)
            others_idx = random.randint(0,self.data_len - 1)
            if self.target == 'vocal':
                vocal_idx = idx
            elif self.target == 'bass':
                bass_idx = idx
            elif self.target == 'drum':
                drum_idx = idx
            elif self.target == 'others':
                others_idx = idx
        else:
            vocal_idx = idx
            bass_idx = idx
            drum_idx = idx
            others_idx = idx

        vocal_signal, vocal_fs = torchaudio.load(str(self.sources_vocal[vocal_idx]))
        
        bass_signal, bass_fs = torchaudio.load(str(self.sources_bass[bass_idx]))
        drum_signal, drum_fs = torchaudio.load(str(self.sources_drum[drum_idx]))
        others_signal, others_fs = torchaudio.load(str(self.sources_others[others_idx]))
        #print(vocal_signal.shape)
        #print(self.duration_len)
        if self.random_mix:
            data, vocal_data, vocal_duration = self.random_mixing(vocal_signal, bass_signal, drum_signal, others_signal, vocal_fs, self.duration_len, vocal_idx, bass_idx, drum_idx, others_idx)
            #print("random_mixing")
        else:
            data, vocal_data, vocal_duration = self.mixing(vocal_signal, bass_signal, drum_signal, others_signal, vocal_fs, self.duration_len, vocal_idx, bass_idx, drum_idx, others_idx)
            #print("just mixing")
            '''
            duration_length = self.duration_len
            period = 0
            data, data_fs = torchaudio.load(str(self.sources_mixture[idx]))
            vocal_data, vocal_fs = torchaudio.load(str(self.sources_vocal[idx]))
            data = data[:, period:period+duration_length*data_fs]
            vocal_data = vocal_data[:, period:period+duration_length*vocal_fs]
            '''
        data = self.normalize(data, vocal_idx, bass_idx, drum_idx, others_idx)
        vocal_data = self.normalize(vocal_data, vocal_idx, bass_idx, drum_idx, others_idx)
        #print("MAX:", torch.max(data), torch.min(data), torch.max(vocal_data), torch.min(vocal_data))
        #data_name = "%s_%s_%s_%s_.wav"%(vocal_idx, bass_idx, drum_idx, others_idx)
        #torchaudio.save(data_name, data, vocal_fs)
        
        #print(torch.max(output))
        #print(output.type()) # torch. DoubleTensor, 이전꺼는 torch.ByteTensor
        #pdb.set_trace()
        #print(data.type)
        return data, vocal_data, idx, vocal_duration
    
    def random_mixing(self, vocal_signal, bass_signal, drum_signal, others_signal, fs, duration_length, vocal_idx, bass_idx, drum_idx, others_idx):
        #print("Rand")
        vocal_ratio = random.uniform(0.2,1)
        bass_ratio = random.uniform(0.2,1)
        drum_ratio = random.uniform(0.2,1)
        others_ratio = random.uniform(0.2,1)
        #print(others_signal.shape)
        vocal_dur = random.randint(0, len(vocal_signal[0]) - duration_length * fs-1)
        bass_dur = random.randint(0, len(bass_signal[0]) - duration_length * fs-1)
        drum_dur = random.randint(0, len(drum_signal[0]) - duration_length * fs-1)
        others_dur = random.randint(0, len(others_signal[0]) - duration_length * fs-1)
        vocal_sig = vocal_signal[:, vocal_dur:vocal_dur+duration_length*fs] * vocal_ratio
        bass_sig =bass_signal[:, bass_dur:bass_dur+duration_length*fs] * bass_ratio
        drum_sig =drum_signal[:, drum_dur:drum_dur+duration_length*fs] * drum_ratio
        others_sig =others_signal[:, others_dur:others_dur+duration_length*fs] * others_ratio
        
        #print(torch.max(vocal_sig), vocal_ratio, torch.max(bass_sig), bass_ratio, torch.max(drum_sig), drum_ratio, torch.max(others_sig), others_ratio)
        output = bass_sig + drum_sig + others_sig + vocal_sig
        #print((bass_sig * bass_ratio).shape)
        #print((drum_ratio * drum_sig).shape)
        #print((vocal_sig * vocal_ratio).shape)
        #print((others_sig * others_ratio).shape)
        #print(output[0].shape)
        #print("random :", torch.max(vocal_sig), torch.max(bass_sig), torch.max(drum_sig), torch.max(others_sig), torch.max(output), vocal_idx, bass_idx, drum_idx, others_idx)
        return output, vocal_sig, vocal_dur
    def mixing(self, vocal_signal, bass_signal, drum_signal, others_signal, fs, duration_length, vocal_idx, bass_idx, drum_idx, others_idx):
        #print("Rand")
        vocal_ratio = random.uniform(0.2,1)
        bass_ratio = random.uniform(0.2,1)
        drum_ratio = random.uniform(0.2,1)
        others_ratio = random.uniform(0.2,1)
        #print(others_signal.shape)
        vocal_dur = random.randint(0, len(vocal_signal[0]) - duration_length * fs-1)
        bass_dur = vocal_dur
        drum_dur = vocal_dur
        others_dur = vocal_dur
        #bass_dur = random.randint(0, len(bass_signal[0]) - duration_length * fs-1)
        #drum_dur = random.randint(0, len(drum_signal[0]) - duration_length * fs-1)
        #others_dur = random.randint(0, len(others_signal[0]) - duration_length * fs-1)
        vocal_sig = vocal_signal[:, vocal_dur:vocal_dur+duration_length*fs]
        bass_sig =bass_signal[:, bass_dur:bass_dur+duration_length*fs]
        drum_sig =drum_signal[:, drum_dur:drum_dur+duration_length*fs]
        others_sig =others_signal[:, others_dur:others_dur+duration_length*fs]
        
        #print(torch.max(vocal_sig), vocal_ratio, torch.max(bass_sig), bass_ratio, torch.max(drum_sig), drum_ratio, torch.max(others_sig), others_ratio)
        output = bass_sig + drum_sig + others_sig + vocal_sig
        #print((bass_sig * bass_ratio).shape)
        #print((drum_ratio * drum_sig).shape)
        #print((vocal_sig * vocal_ratio).shape)
        #print((others_sig * others_ratio).shape)
        #print(output[0].shape)
        #print("random :", torch.max(vocal_sig), torch.max(bass_sig), torch.max(drum_sig), torch.max(others_sig), torch.max(output), vocal_idx, bass_idx, drum_idx, others_idx)
        return output, vocal_sig, vocal_dur

    def normalize(self, data, vocal_idx, bass_idx, drum_idx, others_idx):
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



'''


class Dataset:
    def __init__(
        self,
        input_source_types: List[str],
        target_source_types: List[str],
        paired_input_target_data: bool,
        input_channels: int,
        augmentor: augmentation,
        segment_samples: int,
    ):
        r"""Used for getting data according to a meta.
        Args:
            input_source_types: list of str, e.g., ['vocals', 'accompaniment']
            target_source_types: list of str, e.g., ['vocals']
            input_channels: int
            augmentor: Augmentor
            segment_samples: int
        """
        self.input_source_types = input_source_types
        self.paired_input_target_data = paired_input_target_data
        self.input_channels = input_channels
        self.augmentor = augmentor
        self.segment_samples = segment_samples

        if paired_input_target_data:
            self.source_types = list(set(input_source_types) | set(target_source_types))

        else:
            self.source_types = input_source_types

    def __getitem__(self, meta: Dict) -> Dict:
        r"""Return data according to a meta. E.g., an input meta looks like: {
            'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
            'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}.
        }
        Then, vocals segments of song_A and song_B will be mixed (mix-audio augmentation).
        Accompaniment segments of song_C and song_B will be mixed (mix-audio augmentation).
        Finally, mixture is created by summing vocals and accompaniment.
        Args:
            meta: dict, e.g., {
                'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
                'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}
            }
        Returns:
            data_dict: dict, e.g., {
                'vocals': (channels, segments_num),
                'accompaniment': (channels, segments_num),
                'mixture': (channels, segments_num),
            }
        """
        data_dict = {}

        for source_type in self.source_types:
            # E.g., ['vocals', 'accompaniment']

            waveforms = []  # Audio segments to be mix-audio augmented.

            for m in meta[source_type]:
                # E.g., {
                #     'hdf5_path': '.../song_A.h5',
                #     'key_in_hdf5': 'vocals',
                #     'begin_sample': '13406400',
                #     'end_sample': 13538700,
                # }

                hdf5_path = m['hdf5_path']
                key_in_hdf5 = m['key_in_hdf5']
                bgn_sample = m['begin_sample']
                end_sample = m['end_sample']

                with h5py.File(hdf5_path, 'r') as hf:

                    if source_type == 'audioset':
                        index_in_hdf5 = m['index_in_hdf5']
                        waveform = int16_to_float32(
                            hf['waveform'][index_in_hdf5][bgn_sample:end_sample]
                        )
                        waveform = waveform[None, :]
                    else:
                        waveform = int16_to_float32(
                            hf[key_in_hdf5][:, bgn_sample:end_sample]
                        )

                if self.paired_input_target_data:
                    # TODO
                    pass

                else:
                    if self.augmentor:
                        waveform = self.augmentor(waveform, source_type)

                if source_type in self.input_source_types:
                    waveform = self.match_waveform_to_input_channels(
                        waveform=waveform, input_channels=self.input_channels
                    )
                    # (input_channels, segments_num)

                waveform = librosa.util.fix_length(
                    waveform, size=self.segment_samples, axis=1
                )

                waveforms.append(waveform)
            # E.g., waveforms: [(input_channels, audio_samples), (input_channels, audio_samples)]

            # mix-audio augmentation
            data_dict[source_type] = np.sum(waveforms, axis=0)
            # data_dict[source_type]: (input_channels, audio_samples)

        # data_dict looks like: {
        #     'voclas': (input_channels, audio_samples),
        #     'accompaniment': (input_channels, audio_samples)
        # }

        return data_dict
'''