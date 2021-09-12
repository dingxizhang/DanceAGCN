import os
import torch
import random
import numpy as np

from skeleton_sequence import SkeletonSequence

class DanceDataset(torch.utils.data.Dataset):
    def __init__(self, holder, data_in='raw', bez_degree=None, for_animation=False):
        self.holder = holder  # the object created in dataset_holder.py
        self.data_in = data_in
        self.bez_degree = bez_degree
        self.for_animation = for_animation

    def __len__(self):
        return self.holder.n_samples

    def get_music_skel_seq(self, item):
        return None, SkeletonSequence(None)  # to be overridden

    def __getitem__(self, item):
        # music, skel_seq = self.get_music_skel_seq(item)
        skel_seq = self.get_music_skel_seq(item)
        metadata = skel_seq.metadata
        label = metadata['label']
        dance = self.get_dance_data(skel_seq)
        # DX: Modified _data member here just for visualization
        skel_seq._data = dance

        if self.for_animation:
            return dance, label, metadata, skel_seq
        
        return dance, label, metadata
        
    def get_dance_data(self, skel_sequence):
        if self.data_in == 'raw':
            return skel_sequence.get_raw_data(as_is=True)
        
        elif self.data_in == 'raw+gaussian':
            b = skel_sequence.get_raw_data(as_is=True)
            b = self.add_gaussian_noise(b)
            return b
        
        elif self.data_in == 'linear':
            b = skel_sequence.get_linear_interpolated_skeleton()
            return b
        
        elif self.data_in == 'raw+bcurve':
            return skel_sequence.get_raw_plus_bcurve_data(self.bez_degree, padding_size=self.holder.seq_length)
        
        elif self.data_in.split('+',1)[0] == 'bcurve':
            # frames_list_path = skel_sequence.metadata['filename'].replace('.json', '.npy')
            frames_list_path = skel_sequence.metadata['filename'].replace('.json', '.npy').replace('_gaussian', '')
            frames_list_path = self.holder.data_path.rsplit('/', 1)[0]+ '/frames_list/' + frames_list_path
            frames_list = np.load(frames_list_path).tolist()
            b, _, outliers= skel_sequence.get_bezier_skeleton(order=self.bez_degree, body=0, window=10, overlap=4, target_length=None,
                                                    frames_list=frames_list, bounds=(0, 1799))
            if self.data_in.find('gaussian') > -1:
                b = self.add_gaussian_noise(b)
            return b.astype('<f4')
        else:
            raise ValueError(f'Cannot deal with this data input: {self.data_in}')
    
    @ staticmethod
    def add_gaussian_noise(x, mu=0, sigma=0.2):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i][j] += random.gauss(mu, sigma)*x[i][j]
        return x


class DanceRevolutionDataset(DanceDataset):
    def __init__(self, holder, data_in='raw', bez_degree=None, for_animation=False):
        super().__init__(holder, data_in, bez_degree=bez_degree, for_animation=for_animation)

    def get_music_skel_seq(self, item):
        # music = self.holder.music_array[item]
        skel_seq = self.holder.skeletons[item]
        # return music, skel_seq
        return skel_seq
