import ctypes
import multiprocessing as mp
import numpy as np
import os
import sys
import json
import pickle
import random
sys.path.append('/home/dingxi/DanceRevolution')
sys.path.append('/home/dingxi/DanceRevolution/v2')
from bezier import BezierFitter
from skeleton_sequence import SkeletonSequence
from skeleton_structure import DanceRevolutionStructure, AISTplusplusStructure
# from utils.functional import load_data, load_test_data  # DM these are functions from dance revolution code


class DanceRevolutionHolder:
    def __init__(self, data_path, split, source, file_list=None, train_interval=1800, music_feat_dim=438):
        assert split in ('train', 'test'), 'Split must be either `train` or `test`'
        assert source in ('dancerevolution', 'aist++'), 'Source must be either `dancerevolution` or `aist++`'
        
        if file_list is None:
            file_list = sorted(os.listdir(data_path))

        if source == 'dancerevolution':
            music, dance, self.filenames = self.load_data(file_list, data_path, split, interval=train_interval, return_fnames=True)
            self.n_nodes = 25
            self.labels_str_to_int = {'ballet': 0, 'hiphop': 1, 'pop': 2}
            self.skeleton_structure = DanceRevolutionStructure()  # this define the edges of the skeleton
        elif source == 'aist++':
            music, dance, self.filenames = self.load_aist_data(file_list, data_path, split, interval=train_interval, return_fnames=True)
            self.n_nodes = 17
            self.labels_str_to_int = {'gBR': 0, 'gPO': 1, 'gLO': 2, 'gMH':3, 'gLH': 4, 'gHO':5, 'gWA':6, 'gKR':7, 'gJS':8, 'gJB':9}
            self.skeleton_structure = AISTplusplusStructure()  # this define the edges of the skeleton
        
        # if split == 'train':
        #     music, dance, self.filenames = load_data(file_list, data_path, 'train', interval=train_interval, return_fnames=True)
        # else:
        #     music, dance, self.filenames = load_test_data(file_list, data_path)  # TODO you should have your own train/test splits

        assert len(music) == len(dance), 'music/dance sequence mismatch'

        self.source = source
        self.data_path = data_path
        self.split = split
        self.train_interval = train_interval
        self.n_samples = len(dance)
        self.seq_length = dance[0].shape[0]
        self.skel_dim = 2  # xy coordinates
        self.n_bodies = 1  # for convenience it's best to add this extra dimension
        self.music_feat_dim = music_feat_dim
        self.skeletons = []

        self.dance_array_shape = (self.n_samples, self.skel_dim, self.seq_length, self.n_nodes, self.n_bodies)
        self.music_array_shape = (self.n_samples, self.music_feat_dim, self.seq_length)

        self.bezier_fitter = BezierFitter()

        # we create mp arrays so that these can be shared across processes safely, i.e. we can share only one copy of
        # the data across pytorch data loader workers
        dance_mp_array = mp.Array(ctypes.c_float, int(np.prod(self.dance_array_shape)))
        # music_mp_array = mp.Array(ctypes.c_float, int(np.prod(self.music_array_shape)))

        self.dance_array = np.ctypeslib.as_array(dance_mp_array.get_obj()).reshape(self.dance_array_shape)
        # self.music_array = np.ctypeslib.as_array(music_mp_array.get_obj()).reshape(self.music_array_shape)
        
        self.labels_int_to_str = {v: k for k, v in self.labels_str_to_int.items()}
        self.metadata = [self.get_metadata_from_filename(fn, i) for i, fn in enumerate(self.filenames)]

        for i, (m, d) in enumerate(zip(music, dance)):
            # DX:
            # assert d.shape[0] == self.seq_length, 'Sequence length mismatch, {} not equals to {}'.format(d.shape[0], self.seq_length)
            # assert m.shape[0] == self.seq_length and d.shape[0] == self.seq_length, 'Sequence length mismatch'
            s = self.parse_dance_sequence(d)
            self.dance_array[i] = s
            # self.music_array[i] = m.T

            # important! pass the i-th dance array element in order to correctly share the data instead of s here
            # DX: Each sequence of dance data with its label are stored as a SkeletonSequence class in self.skeletons list
            skel_seq = SkeletonSequence(data=self.dance_array[i], skel_structure=self.skeleton_structure,
                                        metadata=self.metadata[i], is_2d=True, cache=False, fitter=self.bezier_fitter)

            self.skeletons.append(skel_seq)

    def get_metadata_from_filename(self, filename, index):
        splits = filename.replace('.json', '').split('_')
        style = splits[0]
        seq_id = '_'.join(splits[2:])
        uid = style + '-' + seq_id
        metadata = dict(style=style, uid=uid, seq_id=seq_id, filename=filename, index=index,
                        label=self.labels_str_to_int[style], label_str=style)

        return metadata

    @staticmethod
    def parse_dance_sequence(x, scale_input=False, w=None, h=None, add_body_dim=True):
        if scale_input:
            assert w is not None and h is not None
        dim = int(x.shape[1]/2)

        x = x.reshape(-1, dim, 2)
        x = x.transpose((2, 0, 1))

        if add_body_dim:
            x = np.expand_dims(x, 3)

        if scale_input:
            x[0] = (x[0] + 1) * 0.5 * w
            x[1] = (x[1] + 1) * 0.5 * h
        

        return x

    def load_data(self, file_list, data_dir, split, interval=100, data_type='2D', return_fnames=False):
    # DX: Given a file_list and data_dir, output two lists music_data (contains music features) 
    # and dance_data (contains skeleton sequences with fixed interval)

        music_data, dance_data = [], []
        fnames = file_list
        # fnames = fnames[:10]  # For debug
        for fname in fnames:
            path = os.path.join(data_dir, fname)
            with open(path) as f:
                sample_dict = json.loads(f.read())
                np_music = np.array(sample_dict['music_array'])
                np_dance = np.array(sample_dict['dance_array'])
                
                if data_type == '2D':
                    # Only use 25 keypoints skeleton (basic bone) for 2D
                    # Only 'pose_keypoints_2d' is used, others including 'face_keypoints_2d', 'hand_left_keypoints_2d'
                    # and 'hand_right_keypoints_2d' are discarded
                    
                    # DX: in this step, missing frames for linear interpolation (-1 for all nodes) will be modified
                    # to be 0 for all nodes except for node 16, 17
                    np_dance = np_dance[:, :50]
                    root = np_dance[:, 2*8:2*9]
                    np_dance = np_dance - np.tile(root, (1, 25)) # this will modify -1 elements
                    np_dance[:, 2*8:2*9] = root

                if split == 'train':
                    seq_len, dim = np_music.shape
                    for i in range(0, seq_len, interval):
                        music_sub_seq = np_music[i: i + interval]
                        dance_sub_seq = np_dance[i: i + interval]
                        if len(music_sub_seq) == interval:
                            music_data.append(music_sub_seq)
                            dance_data.append(dance_sub_seq)
                elif split == 'test':
                    music_data.append(np_music)
                    dance_data.append(np_dance)

        if return_fnames:
            return music_data, dance_data, fnames
        else:
            return music_data, dance_data

    def load_aist_data(self, file_list, data_dir, split, interval=100, data_type='2D', return_fnames=False):
        music_data, dance_data = [], []
        fnames = file_list
        for fname in fnames:
            path = os.path.join(data_dir, fname)
            with open(path, 'rb') as f:
                np_dance = pickle.load(f)
            music_data.append([])
            dance_data.append(np_dance)
        
        if return_fnames:
            return music_data, dance_data, fnames
        else:
            return music_data, dance_data


if __name__ == '__main__':
    # train_holder = DanceRevolutionHolder('/home/davide/data/datasets/dance_revolution/data/train_1min', 'train')
    # test_holder = DanceRevolutionHolder('/home/davide/data/datasets/dance_revolution/data/test_1min', 'test')
    train_holder = DanceRevolutionHolder('/home/dingxi/DanceRevolution/data/train_1min', 'train')
