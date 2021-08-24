import ctypes
import multiprocessing as mp
import numpy as np

from bezier import BezierFitter
from skeleton_sequence import SkeletonSequence
from skeleton_structure import DanceRevolutionStructure
from utils.functional import load_data, load_test_data  # DM these are functions from dance revolution code


class DanceRevolutionHolder:
    def __init__(self, data_path, split, train_interval=900, music_feat_dim=438):
        assert split in ('train', 'test'), 'Split must be either `train` or `test`'

        if split == 'train':
            music, dance, self.filenames = load_data(data_path, interval=train_interval, return_fnames=True)
        else:
            music, dance, self.filenames = load_test_data(data_path)  # TODO you should have your own train/test splits

        assert len(music) == len(dance), 'music/dance sequence mismatch'

        self.split = split
        self.train_interval = train_interval
        self.n_samples = len(music)
        self.seq_length = dance[0].shape[0]
        self.skel_dim = 2  # xy coordinates
        self.n_nodes = 25
        self.n_bodies = 1  # for convenience it's best to add this extra dimension
        self.music_feat_dim = music_feat_dim
        self.skeletons = []

        self.dance_array_shape = (self.n_samples, self.skel_dim, self.seq_length, self.n_nodes, self.n_bodies)
        self.music_array_shape = (self.n_samples, self.music_feat_dim, self.seq_length)

        self.bezier_fitter = BezierFitter()
        self.skeleton_structure = DanceRevolutionStructure()  # this define the edges of the skeleton

        # we create mp arrays so that these can be shared across processes safely, i.e. we can share only one copy of
        # the data across pytorch data loader workers
        dance_mp_array = mp.Array(ctypes.c_float, int(np.prod(self.dance_array_shape)))
        music_mp_array = mp.Array(ctypes.c_float, int(np.prod(self.music_array_shape)))

        self.dance_array = np.ctypeslib.as_array(dance_mp_array.get_obj()).reshape(self.dance_array_shape)
        self.music_array = np.ctypeslib.as_array(music_mp_array.get_obj()).reshape(self.music_array_shape)

        self.labels_str_to_int = {'ballet': 0, 'hiphop': 1, 'pop': 2}
        self.labels_int_to_str = {v: k for k, v in self.labels_str_to_int.items()}
        self.metadata = [self.get_metadata_from_filename(fn, i) for i, fn in enumerate(self.filenames)]

        for i, (m, d) in enumerate(zip(music, dance)):
            assert m.shape[0] == self.seq_length and d.shape[0] == self.seq_length, 'Sequence length mismatch'
            s = self.parse_dance_sequence(d)
            self.dance_array[i] = s
            self.music_array[i] = m.T

            # important! pass the i-th dance array element in order to correctly share the data instead of s here
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

        x = x.reshape(-1, 25, 2)
        x = x.transpose((2, 0, 1))

        if add_body_dim:
            x = np.expand_dims(x, 3)

        if scale_input:
            x[0] = (x[0] + 1) * 0.5 * w
            x[1] = (x[1] + 1) * 0.5 * h

        return x


if __name__ == '__main__':
    # train_holder = DanceRevolutionHolder('/home/davide/data/datasets/dance_revolution/data/train_1min', 'train')
    test_holder = DanceRevolutionHolder('/home/davide/data/datasets/dance_revolution/data/test_1min', 'test')
