import torch

from skeleton_sequence import SkeletonSequence


class DanceDataset(torch.utils.data.Dataset):
    def __init__(self, holder, data_in='raw', bez_degree=None):
        self.holder = holder  # the object created in dataset_holder.py
        self.data_in = data_in
        self.bez_degree = bez_degree

    def __len__(self):
        return self.holder.n_samples

    def get_music_skel_seq(self, item):
        return None, SkeletonSequence(None)  # to be overridden

    def __getitem__(self, item):
        music, skel_seq = self.get_music_skel_seq(item)
        metadata = skel_seq.metadata
        label = metadata['label']
        dance = self.get_dance_data(skel_seq)

        return music, dance, label, metadata

    def get_dance_data(self, skel_sequence):
        if self.data_in == 'raw':
            return skel_sequence.get_raw_data(as_is=True)
        elif self.data_in == 'raw+bcurve':
            return skel_sequence.get_raw_plus_bcurve_data(self.bez_degree, padding_size=self.holder.seq_length)
        else:
            raise ValueError(f'Cannot deal with this data input: {self.data_in}')


class DanceRevolutionDataset(DanceDataset):
    def __init__(self, holder, data_in='raw', bez_degree=None):
        super().__init__(holder, data_in, bez_degree=bez_degree)

    def get_music_skel_seq(self, item):
        music = self.holder.music_array[item]
        skel_seq = self.holder.skeletons[item]
        return music, skel_seq
