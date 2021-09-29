import os
import random
import skimage.measure
import numpy as np
import torch

from logging import warning
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from bezier import BezierFitter, plot_bezier_curve
from plots import create_3d_figure, squeeze_subplots, remove_ticks_and_labels, scale_3d_axis, set_3d_axis_limits, \
    set_2d_axis_limits


# DM: some tiny plotting bits were adapted from
# https://github.com/XiaoCode-er/3D-Skeleton-Display and
# https://github.com/shahroudy/NTURGB-D/blob/master/Python/txt2npy.py and
# https://github.com/open-mmlab/mmskeleton/blob/master/deprecated/tools/data_processing/ntu_gendata.py


# DM taken from mmskeleton utils
def auto_padding(data_numpy, size, random_pad=False, begin=None, return_index=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        if begin is None:  # DM
            begin = random.randint(0, size - T) if random_pad else 0

        data_numpy_paded = np.zeros((C, size, V, M), dtype=data_numpy.dtype)
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy

        if return_index:
            return data_numpy_paded, begin
        else:
            return data_numpy_paded
    else:
        if return_index:
            return data_numpy, 0
        else:
            return data_numpy


class SkeletonSequence:
    def __init__(self, data, skel_structure=None, metadata=None, cache=True, filter_zeros_when_fitting=False,
                 cp_cache=None, bcurve_cache=None, cache_flags=None, empty_mask_cache=None, actual_frames_cache=None,
                 actual_frames_cached=False, empty_mask_cached=False, is_2d=False, fitter=None, ax_limits=None):
        # IMPORTANT when this object is used for training/testing, none of the below variables will be permanent, as
        # the data loader creates and destroy objects on the fly. Rely on the cache objects passed in to make what's
        # needed permanent

        # data is expected to be an array of shape C, T, V, M, corresponding to channels, frames, joints, bodies

        self._data = data  # this will be a pointer to a (row of the) shared mp.Array
        self.n_frames = None
        self.n_nodes = None
        self.n_bodies = None
        self.loaded = False
        self.skel_structure = skel_structure
        self.metadata = {} if metadata is None else metadata
        self.channels = None
        self._body_in_frame = empty_mask_cache  # pointer to a row of the whole shared mp.Array
        self._actual_frames = actual_frames_cache
        self.cache = cache
        self.filter_zeros_when_fitting = filter_zeros_when_fitting
        self.is_2d = is_2d
        self.fitter = BezierFitter() if fitter is None else fitter

        # keys are order/degree of the curve, values are numpy arrays storing bezier curve xyz points
        self._bcurve_cache = bcurve_cache  # pointer to a row of the whole shared mp.Array
        self._cp_cache = cp_cache  # pointer to a row of the whole shared mp.Array
        self._cached_degree = None if cp_cache is None else cp_cache.shape[-1] - 1  # n. of cp -1 gives B. degree

        self._cache_flags = cache_flags  # pointer to a row of the whole shared mp.Array
        self.actual_frames_cached = actual_frames_cached
        self.empty_mask_cached = empty_mask_cached
        self._data_loaded()
        self.ax_limits = ax_limits

    def new_data_filter_empty(self, node_tolerance=1):
        bof = np.array([[not SkeletonSequence.is_empty(self._data[:, f, :, b], node_tolerance=node_tolerance)
                         for b in range(self.n_bodies)] for f in range(self.n_frames)])
        xyz = np.zeros_like(self._data)
        n_frames = bof.sum(axis=0)

        for b in range(self.n_bodies):
            xyz[:, 0:n_frames[b], :, b:b + 1] = self._data[:, bof[:, b], :, b:b + 1]

        return xyz

    def filter_empty(self, node_tolerance=1):
        raise RuntimeError('Do not use this function')
        # self._data = self.new_data_filter_empty(node_tolerance=node_tolerance)
        # self.__body_in_frame = None
        # self.__actual_frames = {}

    def to_cuda(self, device):
        self._data = torch.tensor(self._data, device=device)

    def _data_loaded(self):
        self.channels, self.n_frames, self.n_nodes, self.n_bodies = self._data.shape
        self.loaded = True

    def _bof(self):
        # shape of the array below is (n_frames, n_bodies), where [i, j] == True means body j is present at frame i
        return np.array([[not SkeletonSequence.is_empty(self._data[:, f, :, b])
                          for b in range(self.n_bodies)] for f in range(self.n_frames)])

    def body_in_frame(self):
        assert self.loaded, 'Data not loaded'

        # this is an expensive search, so we do it once
        if self.cache:
            if not self.empty_mask_cached:
                self._body_in_frame = self._bof()
                self.empty_mask_cached = True

            return self._body_in_frame
        else:
            return self._bof()

    def get_actual_duration(self):
        # in case the data was pre-processed such that frames were duplicated, we need to do this to get the
        # real length of the sequence
        duration = []

        for m in range(self._data.shape[-1]):
            n = np.unique(self._data[:, :, :, m], axis=1).shape[1]
            duration.append(n)

        return duration

    def _get_frames(self):
        bif = self.body_in_frame()
        non_zero_idx = [bif[:, i].nonzero()[0] for i in range(bif.shape[1])]
        actual_frames = np.array([len(nz) for nz in non_zero_idx])
        offsets = [o[0] if len(o) > 0 else 0 for o in non_zero_idx]
        return np.column_stack([actual_frames, offsets])

    def get_actual_n_frames(self, body='all', drop_duplicate_frames=False):
        assert self.loaded, 'Data not loaded'
        assert body != 'any', 'No longer supported'

        if self.cache:
            if not self.actual_frames_cached:
                self._actual_frames = self._get_frames()
                self.actual_frames_cached = True

            a = self._actual_frames
        else:
            a = self._get_frames()

        if body == 'all':
            return a[:, 0], a[:, 1]

        else:
            return a[body, 0], a[body, 1]

    def get_raw_data(self, body_sampling='max_non_empty', as_is=False, ratio=None):
        if as_is:
            return self._data

        if body_sampling == 'all':
            actual_frames, offset = self.get_actual_n_frames(body='all')
            start = min(offset)
            end = max(actual_frames)

            if ratio is None:
                return self._data[:, start:end, :, :]
            else:
                return self._data[:, start:end:ratio, :, :]

        else:
            body, actual_frames, offset = self._sample_body(body_sampling)
            # we need this body slicing to keep the right shape of the array
            return self._data[:, offset:offset+actual_frames, :, body:body+1]

    def get_bezier_data(self, order, body_sampling='max_non_empty', normalise=False, dtype=np.float32,
                        interpolation_points=None, padding_size=None, random_pad=False, ratio=None, return_cp=False,
                        return_only_cp=False):
        n_frames, offsets = self.get_actual_n_frames('all')
        b = None

        if body_sampling == 'all':
            bodies = range(self.n_bodies)
        else:
            bodies, _, _ = self._sample_body(body_sampling)

            if not (isinstance(bodies, list) or isinstance(bodies, np.ndarray)):
                bodies = [bodies]

        if not return_only_cp:
            if ratio is None:
                assert padding_size != interpolation_points, 'Must specify either interpolation points or padding size'
            else:
                interpolation_points = max([max(1, int(n_frames[b] / ratio)) for b in bodies])

        b_list = []
        cps = []

        for body in bodies:
            b, cp = self.get_bezier_skeleton(order, body, normalise=normalise, dtype=dtype,
                                             interpolation_points=interpolation_points)
            b_list.append(b),
            cps.append(cp)

        if not return_only_cp:
            if interpolation_points is None:
                # they will have different lengths, need to pad
                # note about random_pad: if True, this will entail that different bodies will have different
                # pad-begins. I guess this is fine
                b = np.concatenate([auto_padding(b, padding_size, random_pad=random_pad, begin=offsets[i])
                                    for i, b in enumerate(b_list)], axis=-1)
            else:
                b = np.concatenate(b_list, axis=-1)  # they will have the same number of frames in this case

        cps = np.stack(cps, axis=-1)

        if return_only_cp:
            return cps
        elif return_cp:
            return b, cps
        else:
            return b

    def get_raw_plus_ip_data(self, order, body_sampling='max_non_empty', normalise=False, dtype=np.float32,
                             interpolation_points=None, padding_size=None, random_pad=False):
        actual_frames, offsets = self.get_actual_n_frames(body='all')
        assert padding_size != interpolation_points, 'Must specify either interpolation points or padding size'

        if body_sampling == 'all':
            bodies = range(self.n_bodies)
        else:
            body, _, _ = self._sample_body(body_sampling)
            bodies = [body]

        d_list = []
        pad = padding_size if interpolation_points is None else interpolation_points

        for b in bodies:
            if actual_frames[b] < order + 1:  # we don't have a bezier curve
                channels = self.channels + self.channels * self.fitter.get_n_intermediate_points(order)
                z = np.zeros((channels, pad, self.n_nodes, 1), dtype=dtype)
                d_list.append(z)
                continue

            ip_skel = self.to_b_inter_points(order, b, normalise=normalise, dtype=dtype,
                                             interpolation_points=interpolation_points)

            if interpolation_points is None:  # ip skel and raw data will have the same number of frames
                start = offsets[b]
                end = start + actual_frames[b]
                raw = self._data[:, start:end, :, b:b + 1]
            else:  # in this raw data an ip skel will have different frames. We thus sample frames uniformly from raw
                frames, _, _ = self._sample_frames(body=b, n=interpolation_points, body_sampling=None,
                                                   apply_offset=True, random=False)
                raw = self._data[:, frames, :, b:b + 1]

            data = np.concatenate([raw, ip_skel], axis=0)
            data = auto_padding(data, pad, random_pad=random_pad, begin=offsets[b])
            d_list.append(data)

        # finally we concatenate along body axis, we let do the padding to the dataset object if needed
        data = np.concatenate(d_list, axis=-1)

        return data

    def get_ip_data(self, order, body_sampling='max_non_empty', normalise=False, dtype=np.float32,
                    interpolation_points=None, padding_size=None, random_pad=False, ratio=None):
        actual_frames, offsets = self.get_actual_n_frames(body='all')

        if body_sampling == 'all':
            bodies = range(self.n_bodies)
        else:
            body, _, _ = self._sample_body(body_sampling)
            bodies = [body]

        if ratio is None:
            assert padding_size != interpolation_points, 'Must specify either interpolation points or padding size'
            pad = padding_size if interpolation_points is None else interpolation_points
        else:
            interpolation_points = max([max(1, int(actual_frames[b] / ratio)) for b in bodies])
            pad = interpolation_points

        d_list = []

        for b in bodies:
            if actual_frames[b] < order + 1:  # we don't have a bezier curve
                channels = self.fitter.get_n_intermediate_points(order) * self.channels
                z = np.zeros((channels, pad, self.n_nodes, 1), dtype=dtype)
                d_list.append(z)
                continue

            ip_skel = self.to_b_inter_points(order, b, normalise=normalise, dtype=dtype,
                                             interpolation_points=interpolation_points)
            ip_skel = auto_padding(ip_skel, pad, random_pad=random_pad, begin=offsets[b])
            d_list.append(ip_skel)

        data = np.concatenate(d_list, axis=-1)

        return data

    def get_bcurve_plus_ip_data(self, order, body_sampling='max_non_empty', normalise=False, dtype=np.float32,
                                interpolation_points=None, padding_size=None, random_pad=False, return_as_tuple=False):
        actual_frames, offsets = self.get_actual_n_frames(body='all')
        assert padding_size != interpolation_points, 'Must specify either interpolation points or padding size'

        if body_sampling == 'all':
            bodies = range(self.n_bodies)
        else:
            body, _, _ = self._sample_body(body_sampling)
            bodies = [body]

        d_list = []
        pad = padding_size if interpolation_points is None else interpolation_points

        for b in bodies:
            if actual_frames[b] < order + 1:  # we don't have a bezier curve
                channels = self.channels + self.channels * self.fitter.get_n_intermediate_points(order)
                z = np.zeros((channels, pad, self.n_nodes, 1), dtype=dtype)
                d_list.append(z)
                continue

            ip_skel = self.to_b_inter_points(order, b, normalise=normalise, dtype=dtype,
                                             interpolation_points=interpolation_points)
            bezier_skel, _ = self.get_bezier_skeleton(order, b, normalise=normalise, dtype=dtype,
                                                      interpolation_points=interpolation_points)

            if return_as_tuple:
                bezier_skel = auto_padding(bezier_skel, pad, random_pad=random_pad, begin=offsets[b])
                ip_skel = auto_padding(ip_skel, pad, random_pad=random_pad, begin=offsets[b])
                d_list.append([bezier_skel, ip_skel])
            else:
                data = np.concatenate([bezier_skel, ip_skel], axis=0)
                data = auto_padding(data, pad, random_pad=random_pad, begin=offsets[b])
                d_list.append(data)

        if return_as_tuple:
            data = (np.concatenate([dl[0] for dl in d_list], axis=-1),  # curve
                    np.concatenate([dl[1] for dl in d_list], axis=-1))  # cp
        else:
            # finally we concatenate along body axis, we let do the padding to the dataset object if needed
            data = np.concatenate(d_list, axis=-1)

        return data

    def get_raw_plus_ip_plus_bcurve_data(self, order, body_sampling='max_non_empty', normalise=False, dtype=np.float32,
                                         interpolation_points=None, padding_size=None, random_pad=False):
        actual_frames, offsets = self.get_actual_n_frames(body='all')
        assert padding_size != interpolation_points, 'Must specify either interpolation points or padding size'

        if body_sampling == 'all':
            bodies = range(self.n_bodies)
        else:
            body, _, _ = self._sample_body(body_sampling)
            bodies = [body]

        d_list = []
        pad = padding_size if interpolation_points is None else interpolation_points

        for b in bodies:
            if actual_frames[b] < order + 1:  # we don't have a bezier curve
                channels = self.channels * 2 + self.channels * self.fitter.get_n_intermediate_points(order)
                z = np.zeros((channels, pad, self.n_nodes, 1), dtype=dtype)
                d_list.append(z)
                continue

            ip_skel = self.to_b_inter_points(order, b, normalise=normalise, dtype=dtype,
                                             interpolation_points=interpolation_points)

            bezier_skel, _ = self.get_bezier_skeleton(order, b, normalise=normalise, dtype=dtype,
                                                      interpolation_points=interpolation_points)

            if interpolation_points is None:  # ip skel and raw data will have the same number of frames
                start = offsets[b]
                end = start + actual_frames[b]
                raw = self._data[:, start:end, :, b:b + 1]
            else:  # in this raw data an ip skel will have different frames. We thus sample frames uniformly from raw
                frames, _, _ = self._sample_frames(body=b, n=interpolation_points, body_sampling=None,
                                                   apply_offset=True, random=False)
                raw = self._data[:, frames, :, b:b + 1]

            data = np.concatenate([raw, ip_skel, bezier_skel], axis=0)
            data = auto_padding(data, pad, random_pad=random_pad, begin=offsets[b])
            d_list.append(data)

        # finally we concatenate along body axis, we let do the padding to the dataset object if needed
        data = np.concatenate(d_list, axis=-1)

        return data

    def get_raw_plus_bcurve_data(self, order, body_sampling='max_non_empty', normalise=False, dtype=np.float32,
                                 interpolation_points=None, padding_size=None, random_pad=False,
                                 concat_along_nodes=False):
        actual_frames, offsets = self.get_actual_n_frames(body='all')
        assert padding_size != interpolation_points, 'Must specify either interpolation points or padding size'

        if body_sampling == 'all':
            bodies = range(self.n_bodies)
        else:
            body, _, _ = self._sample_body(body_sampling)
            bodies = [body]

        d_list = []
        pad = padding_size if interpolation_points is None else interpolation_points
        concat_axis = 2 if concat_along_nodes else 0

        for b in bodies:
            if actual_frames[b] < order + 1:  # we don't have a bezier curve
                channels = self.channels if concat_along_nodes else self.channels * 2
                nodes = self.n_nodes * 2 if concat_along_nodes else self.n_nodes
                z = np.zeros((channels, pad, nodes, 1), dtype=dtype)
                d_list.append(z)
                continue

            bezier_skel, _ = self.get_bezier_skeleton(order, b, normalise=normalise, dtype=dtype,
                                                      interpolation_points=interpolation_points)

            if interpolation_points is None:  # ip skel and raw data will have the same number of frames
                start = offsets[b]
                end = start + actual_frames[b]
                raw = self._data[:, start:end, :, b:b + 1]
            else:  # in this raw data an ip skel will have different frames. We thus sample frames uniformly from raw
                frames, _, _ = self._sample_frames(body=b, n=interpolation_points, body_sampling=None,
                                                   apply_offset=True, random=False)
                raw = self._data[:, frames, :, b:b + 1]

            data = np.concatenate([raw, bezier_skel], axis=concat_axis)
            data = auto_padding(data, pad, random_pad=random_pad, begin=offsets[b])
            d_list.append(data)

        # finally we concatenate along body axis, we let do the padding to the dataset object if needed
        data = np.concatenate(d_list, axis=-1)

        return data

    def _get_bdata_from_cache(self, body, degree, interpolation_points=None):
        if self.cache and self._cache_flags[body] and degree == self._cached_degree:
            if interpolation_points is None:
                n_frames, offset = self.get_actual_n_frames(body)
            else:
                n_frames = interpolation_points
                offset = 0

            b = self._bcurve_cache[:, offset:offset+n_frames, :, body:body+1]
            cp = self._cp_cache[body, :, :, :]
            return b, cp
        else:
            return None

    def _set_bdata_in_cache(self, b, cp, body, interpolation_points=None):
        if interpolation_points is None:
            n_frames, offset = self.get_actual_n_frames(body)
        else:
            n_frames = interpolation_points
            offset = 0

        self._bcurve_cache[:, offset:offset+n_frames, :, body:body+1] = b  # will fail if there is any shape mismatch
        self._cp_cache[body] = cp
        self._cache_flags[body] = True

    def get_bezier_skeleton(self, order, body, normalise=False, dtype=np.float32, interpolation_points=None,
                            window=None, overlap=None, target_length=None, outliers_neigh=None,
                            save_idx=None, frames_list=None, bounds=None):

        if window is not None and overlap is not None:
            assert not self.cache, 'Review windows with caching'

        assert not (interpolation_points is not None and target_length is not None), \
            'Specify target_length or interpolation_points, but not both'

        assert not normalise, 'Check normalising data first'
        # Cache contains fixed-bezier-order stuff. Specifically, contains the bezier curve with interpolation points
        # equal to the action's length, as well as the corresponding control points
        cached = self._get_bdata_from_cache(body, order, interpolation_points=interpolation_points)
        outliers = None

        if self.cache and cached is not None:
            return cached[0], cached[1], outliers
        else:
            b, cp, outliers = self.to_bezier(order, body, normalise=normalise, dtype=dtype, target_length=target_length,
                                             interpolation_points=interpolation_points, window=window, overlap=overlap,
                                             outliers_neigh=outliers_neigh, save_idx=save_idx, frames_list=frames_list,
                                             bounds=bounds)

            if self.cache and order == self._cached_degree:
                self._set_bdata_in_cache(b, cp, body, interpolation_points=interpolation_points)

            return b, cp, outliers

    def to_bezier_depr(self, order, body, normalise=False, dtype=np.float32, dtype_pt=torch.float32, pytorch=False,
                       device=None, with_control_points=False):
        warning('You should not use this function')
        assert not pytorch, 'Use numpy which is faster'
        actual_frames, _ = self.get_actual_n_frames(body)

        if pytorch:
            b = torch.zeros((self.channels, actual_frames, self.n_nodes, 1), dtype=dtype_pt, device=device)
            cp = torch.zeros((self.channels, order + 1, self.n_nodes), dtype=dtype_pt, device=device) \
                if with_control_points else None
        else:
            b = np.zeros((self.channels, actual_frames, self.n_nodes, 1), dtype=dtype)
            cp = np.zeros((self.channels, order + 1, self.n_nodes), dtype=dtype) if with_control_points else None

        for node in range(self.n_nodes):
            curve, control_points = self.get_bezier_trajectory(order, node, body=body, normalise=normalise)
            b[:, :, node, 0] = curve.T

            if with_control_points:
                cp[:, :, node] = control_points.T

        return b, cp

    def to_bezier(self, order, body, normalise=False, dtype=np.float32, interpolation_points=None, window=None,
                  overlap=None, target_length=None, outliers_neigh=None, save_idx=None, frames_list=None,
                  bounds=None):
        assert not (interpolation_points is not None and target_length is not None), \
            'Specify target_length or interpolation_points, but not both'
        # DX: modified here to stop removing zero keypoints
        nodes = self.get_nodes(body, frame='all', normalise=normalise)
        outliers = None

        if self.is_2d:
            to_fit = nodes[0:2, ...]
            confidence_score = nodes[2, ...] if nodes.shape[0] == 3 else None
        else:
            to_fit = nodes
            confidence_score = None

        if outliers_neigh is not None and save_idx is not None:
            assert window is not None and overlap is not None, 'Review outlier detection for non windowed method'

        if frames_list is not None:
            assert bounds is not None, 'Need to provide sequence bounds if providing frames'
            assert target_length is None, 'Either specify frame bounds or target length'
            assert window is not None and overlap is not None, 'Review outlier detection for non windowed method'

        if window is not None and overlap is not None:
            b, _, cp, outliers = self.fitter.fit_bezier_series_with_windows(to_fit, order, window, overlap,
                                                                            target_length=target_length,
                                                                            inter_points=interpolation_points,
                                                                            outliers_neigh=outliers_neigh,
                                                                            save_idx=save_idx, frames_list=frames_list,
                                                                            bounds=bounds)
        else:
            n_frames = nodes.shape[1]
            interpolation_points = n_frames if interpolation_points is None else interpolation_points
            cp = self.fitter.fit_bezier_series(to_fit, order, dtype=dtype, zero_if_not_enough_points=True)
            b = self.fitter.get_bezier_curves(cp, order, inter_points=interpolation_points)

        b = b.transpose(1, 2, 0)[:, :, :, np.newaxis]

        if self.is_2d and confidence_score is not None:
            # this will be fed to the model, so we still need to put the confidence score as done with the raw data
            b = np.concatenate([b, np.expand_dims(confidence_score, (0, 3))], axis=0)

        return b, cp, outliers

    def to_b_inter_points(self, order, body, normalise=False, dtype=np.float32, interpolation_points=None, cython=True):
        assert not self.is_2d, 'Review this for the 2d case'
        assert cython, 'Use cython version which is way faster'
        actual_frames, _ = self.get_actual_n_frames(body)
        interpolation_points = actual_frames if interpolation_points is None else interpolation_points
        _, cp = self.get_bezier_skeleton(order, body, normalise=normalise, dtype=dtype,
                                         interpolation_points=interpolation_points)
        steps = np.linspace(0, 1, interpolation_points, dtype=dtype)

        if cython:
            ip = np.stack([self.fitter.get_intermediate_points_series(cp, t, cython=True) for t in steps])
        else:
            ip = np.stack([self.fitter.get_intermediate_points_series(cp, t, cython=False).reshape(self.n_nodes, -1)
                           for t in steps])

        ip = ip.transpose((2, 0, 1))[:, :, :, np.newaxis]
        return ip

    def to_b_inter_points_deprecated(self, order, body, normalise=False, dtype=np.float32, pad=True):
        warning('You should not use this function')
        actual_frames, offset = self.get_actual_n_frames(body='all')
        n_inter_points = self.fitter.get_n_intermediate_points(order)
        channels = self.channels * n_inter_points
        n_frames = self.n_frames if pad or body == 'all' else actual_frames[body]
        bodies = np.arange(self.n_bodies) if body == 'all' else [body]

        data = np.zeros((channels, n_frames, self.n_nodes, len(bodies)), dtype=dtype)

        for b in bodies:
            off = offset[b]
            af = actual_frames[b]

            if af < order + 1:
                continue

            for node in range(self.n_nodes):
                _, _, ip = self.get_bezier_trajectory(order, node, body=b, normalise=normalise, with_inter_points=True)
                data[:, off:off+af, node, b] = ip.reshape(af, channels).T

        return data

    def _sample_frames(self, body_sampling, n, random, apply_offset=True, body=None):
        if body is None:
            body, actual_frames, offset = self._sample_body(body_sampling)
        else:
            actual_frames, offset = self.get_actual_n_frames(body)

        if not apply_offset:
            offset = 0

        assert actual_frames > 0, f'Got no frames for body: {body}. Metadata: {self.metadata}'

        if random and n < actual_frames:
            start = np.random.randint(low=0 + offset, high=offset + actual_frames - n)
            frames = np.arange(start, start + n)
        else:
            frames = np.linspace(0 + offset, offset + actual_frames - 1, n, dtype=np.int)

        return frames, body, offset

    def _sample_body(self, body_sampling):
        actual_frames, offset = self.get_actual_n_frames(body='all')

        if body_sampling == 'any_non_empty':
            if all(actual_frames):  # both are valid, we take a random one
                body = np.random.randint(low=0, high=len(actual_frames))
            else:
                body = np.argmax(actual_frames)
        elif body_sampling == 'max_non_empty':
            body = np.argmax(actual_frames)
        elif body_sampling == 'all_non_empty':
            body = np.where(actual_frames > 0)[0]
        else:
            raise Exception('Unrecognised body sampling strategy: {}'.format(body_sampling))

        actual_frames = actual_frames[body]
        offset = offset[body]

        return body, actual_frames,  offset

    def get_node(self, node, body=0, frame='all_actual', normalise=False):
        assert self.loaded, 'Data not loaded'
        assert not normalise, 'Check normalisation'

        if frame == 'all':
            return self._data[:, :, node, body]
        elif frame == 'all_actual':
            n, offset = self.get_actual_n_frames(body=body)
            return self._data[:, offset:offset+n, node, body]
        else:
            return self._data[:, frame, node, body]

    def get_nodes(self, body=0, frame='all_actual', normalise=False):
        assert self.loaded, 'Data not loaded'
        assert not normalise, 'Check normalisation'

        if frame == 'all':
            return self._data[:, :, :, body]
        elif frame == 'all_actual':
            n, offset = self.get_actual_n_frames(body=body)

            if self.filter_zeros_when_fitting:
                non_empty_nodes = self._data[:, :, :, body].any(axis=0)

                # if we have all or no frames
                if non_empty_nodes.all() or not non_empty_nodes.any():
                    return self._data[:, offset:offset + n, :, body]

                nodes = []
                # we remove empty frames from each single trajectory, then interpolate the missing frames by simply
                # repeating indices (i.e. coordinates) and finally return the stacked data
                for node in range(self.n_nodes):
                    ne = non_empty_nodes[:, node]

                    if not ne.any():
                        nodes.append(self._data[:, offset:offset + n, node, body])
                    else:
                        non_empty_trajectory = self._data[:, ne, node, body]
                        idx = np.linspace(start=0, stop=non_empty_trajectory.shape[1] - 1, num=n, dtype=int)
                        int_trajectory = non_empty_trajectory[:, idx]
                        nodes.append(int_trajectory)

                nodes = np.stack(nodes, axis=2)
                return nodes
            else:
                return self._data[:, offset:offset + n, :, body]
        else:
            return self._data[:, frame, :, body]

    def plot(self, frame, body=0, normalise=False, init_view_h=45, init_view_v=20, ax=None, highlight_node=None,
             plot_only_highlighted=False, joint_size=2, plot_title=True, xyz=None, flip_z=False,
             frame_w_px=None, frame_h_px=None, dpi=150, ax_limits=None):
        assert self.loaded, 'Data not loaded'

        if xyz is None:
            xyz = self.get_skeleton(frame, body).T
        else:
            xyz = xyz[:, frame, :, body].T

        if ax_limits is None:
            ax_limits = self.ax_limits

        if ax is None:
            if self.is_2d:
                if frame_w_px is not None and frame_h_px is not None:
                    px = 1 / dpi
                    figsize = (frame_w_px * px, frame_h_px * px)
                    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=dpi, figsize=figsize)
                else:
                    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=dpi)

                if ax_limits is not None:
                    ax.set_xlim(left=ax_limits[0], right=ax_limits[1])
                    ax.set_ylim(top=ax_limits[2], bottom=ax_limits[3])
            else:
                fig, ax = create_3d_figure()

        self.plot_xyz(xyz, normalise=normalise, init_view_h=init_view_h, init_view_v=init_view_v, ax=ax,
                      highlight_node=highlight_node, plot_only_highlighted=plot_only_highlighted,
                      joint_size=joint_size, flip_z=flip_z)

        if plot_title and self.metadata and 'label_str' in self.metadata:
            ax.set_title(self.metadata['label_str'])

    def get_skeleton(self, frame, body=0):
        assert self.loaded, 'Data not loaded'
        return self._data[:, frame, :, body]

    def get_bezier_trajectory(self, bezier_degree, node, normalise=False, body=0, with_inter_points=False,
                              pytorch=False, device=None, curve_out=None, cp_out=None):
        assert self.loaded, 'Data not loaded'
        assert not pytorch, 'Use numpy which is faster'
        points = self.get_node(node, body=body, normalise=normalise, frame='all_actual').T

        if self.is_2d:
            points = points[:, 0:2]

        n = len(points)
        control_points = self.fitter.fit_bezier(points, degree=bezier_degree, pytorch=pytorch, device=device,
                                                out=cp_out)
        b = self.fitter.get_bezier_curve(control_points, bezier_degree, inter_points=n, pytorch=pytorch, device=device,
                                         out=curve_out)

        if with_inter_points:
            steps = np.linspace(0, 1, n, dtype=np.float32)

            if pytorch:
                inter_points = torch.stack([torch.cat(self.fitter.casteljau(control_points, t))
                                            for t in steps])
            else:
                inter_points = np.stack([np.concatenate(self.fitter.casteljau(control_points, t))
                                         for t in steps])
            return b, control_points, inter_points
        else:
            return b, control_points

    def get_frankenstein(self, nodes, mode, bezier_order, padding_size, body_sampling='all'):
        assert mode in ('raw_on_bezier', 'bezier_on_raw'), f'Unrecognised frankenstein mode: {mode}'
        raw_data = self.get_raw_data(body_sampling=body_sampling, as_is=True)

        random_padding = False
        bezier_order = bezier_order
        b_data = self.get_bezier_data(bezier_order, body_sampling=body_sampling, normalise=False,
                                      interpolation_points=None, padding_size=padding_size,
                                      random_pad=random_padding)

        # shape is channels, n_frames, n_nodes, n_bodies
        if mode == 'raw_on_bezier':
            frankenstein = np.array(b_data)

            for n in nodes:
                frankenstein[:, :, n, :] = raw_data[:, :, n, :]
        elif mode == 'bezier_on_raw':
            frankenstein = np.array(raw_data)

            for n in nodes:
                frankenstein[:, :, n, :] = b_data[:, :, n, :]
        else:
            raise RuntimeError('This should never happen (frankenstein is None')

        return frankenstein

    def get_nodes_displacement(self, bodies='all', avg_body=False, normalise=False):
        frames = self.get_actual_n_frames(body=bodies)[0]
        d = []

        for body, f in enumerate(frames):
            if f == 0:
                continue

            traj = self.get_nodes(body=body)
            framewise_disp = np.stack([np.linalg.norm(traj[:, t] - traj[:, t + 1], axis=0) for t in range(f - 1)])
            total_disp = framewise_disp.sum(axis=0)

            if normalise:
                total_disp /= total_disp.max()

            d.append(total_disp)

        if avg_body:
            d = np.mean(d, axis=0)

        return d

    def get_nodes_covariance(self, return_heatmap=True, heatmap_t=0.95, bezier_order=None):
        assert not self.is_2d, 'Write 2D version of this'
        frames = self.get_actual_n_frames(body='all')[0]
        c_list = []

        for body, f in enumerate(frames):
            if f == 0:
                continue

            # this creates a 2D matrix n_frames x (nodes x xyz)
            if bezier_order is None:
                m = self.get_nodes(body=body)
            else:
                b, cp = self.get_bezier_skeleton(bezier_order, body)
                assert cp.shape[2] == bezier_order + 1, 'Specified bezier order does not match Bezier skeleton order!' \
                                                        'This is probably because you specified a different order ' \
                                                        'than that used to obtain the cached skeleton'
                m = b.squeeze()

            m = m.transpose((1, 2, 0)).reshape((f, self.n_nodes * 3))
            c = np.corrcoef(m, rowvar=False)

            # now we apply max pooling with a kernel 3x3. This will give us the highest correlation between nodes
            # across the 3 dimensions
            cr = skimage.measure.block_reduce(c, (3, 3), np.max)

            # we set correlation on nan values and along the diagonal
            cr[np.isnan(cr)] = 0
            cr[np.diag_indices(cr.shape[0])] = 0

            if return_heatmap:
                rows, cols = np.where(np.abs(cr) >= heatmap_t)
                high_corr_nodes = np.unique(np.concatenate([rows, cols]))
                heatmap = np.zeros(self.n_nodes)
                heatmap[high_corr_nodes] = 1
            else:
                heatmap = None

            c_list.append((cr, heatmap))

        return c_list

    def animate_skeleton(self, body=0, fps=10, views=1, nrows=1, ncols=1, init_view_h=45, init_view_v=20,
                         xyz=None, comparison=False, comparison_title=None, flip_z=False,
                         duration=None, all_frames=True, dpi=150, figsize=None,
                         ax_limits=None, fig_title=None, **kwargs):
        try:
            from moviepy.video.VideoClip import VideoClip
            from moviepy.video.io.bindings import mplfig_to_npimage
        except ImportError:
            print('Need to install moviepy: `pip install moviepy`')
            return

        assert self.loaded, 'Data not loaded'
        assert not (duration is not None and fps is not None), 'Specify either fps or duration'
        fig = plt.figure(dpi=dpi) if figsize is None else plt.figure(figsize=figsize)
        axes = []

        if ax_limits is None:
            ax_limits = self.ax_limits

        def set_ax_lims(ax):
            if ax_limits is not None and self.is_2d:
                ax.set_xlim(left=ax_limits[0], right=ax_limits[1])
                ax.set_ylim(top=ax_limits[3], bottom=ax_limits[2])

        def create_ax():
            ax = fig.add_subplot(nrows, ncols, i + 1, projection=None if self.is_2d else '3d')
            set_ax_lims(ax)
            return ax

        if comparison:
            assert xyz is not None

            if not isinstance(xyz, list):
                xyz = [xyz]
                comparison_title = [comparison_title]

            for i in range(len(xyz) + 1):
                ax = create_ax()
                axes.append(ax)
        else:
            for i in range(views):
                ax = create_ax()
                axes.append(ax)

        squeeze_subplots(fig)

        if all_frames:
            # TODO: DX changed here:
            # n_frames = self._data.shape[1] if xyz is None else xyz.shape[1]
            n_frames = self._data.shape[1] if xyz is None else xyz[0].shape[1]
            offset = 0
        else:
            n_frames, offset = self.get_actual_n_frames(body=body)

        if duration is None:
            duration = int(n_frames / fps)
        else:
            fps = int(n_frames / duration)

        rotations = [init_view_h] if views == 1 else np.linspace(init_view_h, -2*init_view_h, views)

        if self.metadata and 'label_str' in self.metadata:
            fig.suptitle(self.metadata['label_str'])

        def make_frame(t):
            frame = min(int(t * fps) + offset, n_frames-1)

            for i, ax in enumerate(axes):
                ax.clear()

                if comparison:
                    data = self._data if i == 0 else xyz[i-1]
                    self.plot(frame, body=body, ax=ax, init_view_h=rotations[0], init_view_v=init_view_v,
                              plot_title=False, xyz=data, flip_z=flip_z, **kwargs)
                else:
                    self.plot(frame, body=body, ax=ax, init_view_h=rotations[i], init_view_v=init_view_v,
                              plot_title=False, xyz=xyz, flip_z=flip_z, **kwargs,)

                remove_ticks_and_labels(ax, is_3d=not self.is_2d)
                set_ax_lims(ax)

                if comparison:
                    title = 'Original' if i == 0 else comparison_title[i - 1]
                elif fig_title is not None:
                    title = fig_title
                else:
                    title = None

                if title is not None:
                    if ax_limits is None:
                        ax.set_title(title)
                    else:
                        ax.text((ax_limits[0] + ax_limits[1]) / 2,
                                (ax_limits[2] + ax_limits[3]) / 10,
                                title, fontsize=16, ha='center', va='top')

            return mplfig_to_npimage(fig)

        animation = VideoClip(make_frame, duration=duration)
        plt.close(fig)

        return animation

    def plot_as_space_time_graph(self, body=0, normalise=False, init_view_h=45, init_view_v=20,
                                 highlight_node=None, plot_only_highlighted=False, frames='all_actual', plot_n=5,
                                 flip_z=False, ax=None, fig=None, **kwargs):
        assert self.loaded, 'Data not loaded'
        assert not self.is_2d, 'Review this for 2d case'

        if ax is None and fig is None:
            fig, ax = create_3d_figure()

        if frames in ['all', 'all_actual']:
            if frames == 'all':
                n = self.n_frames
                offset = 0
            else:
                n, offset = self.get_actual_n_frames(body=body)

            n -= 1
            frames = np.arange(0, n) if plot_n is None else np.linspace(0, n, plot_n, dtype=np.int)
            frames = frames + offset

        for i, frame in enumerate(frames):
            xyz = np.array(self.get_skeleton(frame, body=body).T)

            if normalise:
                xyz = SkeletonSequence.normalise_skeleton(xyz, channels_first_axis=False)

            xyz[:, 2] += i
            self.plot_xyz(xyz, normalise=False, init_view_h=init_view_h, init_view_v=init_view_v, ax=ax,
                          highlight_node=highlight_node, plot_only_highlighted=plot_only_highlighted,
                          set_axis_lim=False, **kwargs)
            fig.tight_layout()

        remove_ticks_and_labels(ax)
        scale_3d_axis(ax, y_scale=len(frames) * 0.5)
        ax.autoscale_view()

        if self.metadata:
            fig.suptitle(self.metadata['label_str'])

        if flip_z:
            ax.invert_zaxis()

        ax.set_ylabel('Time')

        return ax

    def plot_bezier_trajectory(self, node, bezier_degree, body=0, init_view_h=45, init_view_v=20, ax=None,
                               add_trajectory=True, normalise=False, plot_control_points=True):
        assert self.loaded, 'Data not loaded'
        b, control_points = self.get_bezier_trajectory(bezier_degree, node, body=body, normalise=normalise)
        control_points = control_points if plot_control_points else None

        if ax is None:
            if self.is_2d:
                fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150)
            else:
                fig, ax = create_3d_figure()

        if add_trajectory:
            trajectory = self.get_node(node, body=body, normalise=normalise, frame='all_actual').T

            if self.is_2d:
                trajectory = trajectory[:, 0:2]
        else:
            trajectory = None

        if not self.is_2d:
            ax.view_init(init_view_v, init_view_h)

        plot_bezier_curve(b, control_points, fitted_points=trajectory, ax=ax)

    def animate_bezier_trajectory(self, node, bezier_degree=3, views=3, init_view_h=45, init_view_v=20, fps=10,
                                  body=0, joint_size=3, add_node_trajectory=True, flip_z=False, zdir='z'):
        try:
            from moviepy.video.VideoClip import VideoClip
            from moviepy.video.io.bindings import mplfig_to_npimage
        except ImportError:
            print('Need to install moviepy: `pip install moviepy`')
            return

        assert self.loaded, 'Data not loaded'
        b, control_points = self.get_bezier_trajectory(bezier_degree, node, body=body, normalise=False)
        node_trajectory = self.get_node(node, body=body, normalise=False, frame='all_actual').T

        fig = plt.figure(dpi=150)
        axes = []
        squeeze_subplots(fig)
        n_frames, offset = self.get_actual_n_frames(body=body)
        duration = int(n_frames / fps)
        rotations = [init_view_h] if views == 1 else np.linspace(init_view_h, -2 * init_view_h, views)

        n_rows = 3 if add_node_trajectory else 2

        if self.is_2d:
            node_trajectory = node_trajectory[:, 0:2]
            n_cols = n_rows
            n_rows = 1
            views = 1
        else:
            n_cols = views

        for i in range(n_cols*n_rows):
            ax = fig.add_subplot(n_rows, n_cols, i+1, projection=None if self.is_2d else '3d')
            axes.append(ax)

        if self.metadata:
            fig.suptitle(self.metadata['label_str'])

        b_min = control_points.min(axis=0)
        b_max = control_points.max(axis=0)

        if add_node_trajectory:
            t_min = node_trajectory.min(axis=0)
            t_max = node_trajectory.max(axis=0)
            ax_min = [min(b_min[i], t_min[i]) for i in range(len(b_min))]
            ax_max = [max(b_max[i], t_max[i]) for i in range(len(b_max))]
        else:
            ax_min = b_min
            ax_max = b_max

        def make_frame(t):
            frame = int(t * fps) + offset

            # plot skeleton first
            for i in range(0, views):
                ax = axes[i]
                ax.clear()
                self.plot(frame, body=body, ax=ax, init_view_h=rotations[i], init_view_v=init_view_v,
                          highlight_node=node, joint_size=joint_size, plot_title=False, flip_z=flip_z)
                remove_ticks_and_labels(ax, is_3d=not self.is_2d)

            tt = t / float(duration)
            r = self.fitter.casteljau(control_points, tt)
            limit_t = int(b.shape[0] * tt)

            # plot bezier below skeleton
            for ir, i in enumerate(range(views, views * 2)):
                ax = axes[i]
                ax.clear()

                if self.is_2d:
                    set_2d_axis_limits(ax, x_lim=(ax_min[0], ax_max[0]), y_lim=(ax_min[1], ax_max[1]))
                else:
                    ax.view_init(init_view_v, rotations[ir])

                    if zdir == 'y':
                        set_3d_axis_limits(ax,
                                           x_lim=(ax_min[0], ax_max[0]),
                                           y_lim=(ax_min[2], ax_max[2]),  # sic, because zdir='y'
                                           z_lim=(ax_min[1], ax_max[1]))
                    else:
                        set_3d_axis_limits(ax,
                                           x_lim=(ax_min[0], ax_max[0]),
                                           y_lim=(ax_min[1], ax_max[1]),
                                           z_lim=(ax_min[2], ax_max[2]))

                plot_bezier_curve(b, control_points, fitted_points=None, ax=ax, with_intermediate_points=r,
                                  limit_t=limit_t, add_legend=False)

                if self.is_2d:
                    ax.invert_yaxis()
                elif flip_z:
                    ax.invert_zaxis()

                remove_ticks_and_labels(ax, is_3d=not self.is_2d)

            # plot node trajectory
            if add_node_trajectory:
                for ir, i in enumerate(range(views*2, views * 3)):
                    ax = axes[i]
                    ax.clear()

                    if self.is_2d:
                        set_2d_axis_limits(ax, x_lim=(ax_min[0], ax_max[0]), y_lim=(ax_min[1], ax_max[1]))
                    else:
                        if zdir == 'y':
                            ax.view_init(init_view_v, rotations[ir])
                            set_3d_axis_limits(ax,
                                               x_lim=(ax_min[0], ax_max[0]),
                                               y_lim=(ax_min[2], ax_max[2]),  # sic, because zdir='y'
                                               z_lim=(ax_min[1], ax_max[1]))
                        else:
                            ax.view_init(init_view_v, rotations[ir])
                            set_3d_axis_limits(ax,
                                               x_lim=(ax_min[0], ax_max[0]),
                                               y_lim=(ax_min[1], ax_max[1]),
                                               z_lim=(ax_min[2], ax_max[2]))

                    if self.is_2d:
                        ax.plot(node_trajectory[offset:frame, 0], node_trajectory[offset:frame, 1])
                    else:
                        ax.plot(node_trajectory[offset:frame, 0], node_trajectory[offset:frame, 1],
                                node_trajectory[offset:frame, 2], zdir=zdir)

                    if self.is_2d:
                        ax.invert_yaxis()
                    elif flip_z:
                        ax.invert_zaxis()

                    remove_ticks_and_labels(ax, is_3d=not self.is_2d)

            return mplfig_to_npimage(fig)

        animation = VideoClip(make_frame, duration=duration)
        plt.close(fig)

        return animation

    def animate_bezier_trajectory_all_nodes(self, bezier_degree=3, views=3, init_view_h=45, init_view_v=20, fps=20,
                                            body=0, joint_size=3, add_node_trajectory=True, nodes_set='all',
                                            save_path=None, vid_ext='mp4', flip_z=False):
        assert self.loaded, 'Data not loaded'
        assert not self.is_2d, 'Review for 2d case'
        animations = []

        if save_path is not None:
            uid = self.metadata['uid']
            save_path = os.path.join(save_path, uid)
            Path(save_path).mkdir(parents=True, exist_ok=True)

        if nodes_set == 'trunk':
            nodes = self.skel_structure.trunk_joints
        elif nodes_set == 'legs':
            nodes = self.skel_structure.leg_joints
        elif nodes_set == 'arms':
            nodes = self.skel_structure.arm_joints
        else:
            nodes = list(range(self.n_nodes))

        for node in tqdm(nodes, desc='Producing Bezier animations'):
            animation = self.animate_bezier_trajectory(node, bezier_degree=bezier_degree, views=views,
                                                       init_view_v=init_view_v, init_view_h=init_view_h, fps=fps,
                                                       body=body, joint_size=joint_size,
                                                       add_node_trajectory=add_node_trajectory, flip_z=flip_z)
            animations.append(animation)

            if save_path is not None:
                p = os.path.join(save_path, 'node={}_body={}_degree={}.{}'.format(node, body, bezier_degree, vid_ext))
                animation.write_videofile(p, fps=fps, audio=False, verbose=False, logger=None)

        return animations

    @staticmethod
    def normalise_skeleton(xyz, channels_first_axis=True):
        center_joint = xyz[:, 0] if channels_first_axis else xyz[0, :]

        center_joint_x = np.mean(center_joint[0])
        center_joint_y = np.mean(center_joint[1])
        center_joint_z = np.mean(center_joint[2])

        center = np.array([center_joint_x, center_joint_y, center_joint_z])
        center = center[:, np.newaxis] if channels_first_axis else center
        norm_xyz = xyz - center

        return norm_xyz

    def plot_xyz(self, xyz, normalise=False, init_view_h=45, init_view_v=20, ax=None, highlight_node=None,
                 plot_only_highlighted=False, set_axis_lim=True, joint_size=2, flip_z=False, zdir='z',
                 invert_2d_ax=True):
        if ax is None:
            if self.is_2d:
                fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150)
            else:
                fig, ax = create_3d_figure()

        if not self.is_2d:
            ax.view_init(init_view_v, init_view_h)

        if normalise:
            xyz = SkeletonSequence.normalise_skeleton(xyz, channels_first_axis=False)

        if normalise and set_axis_lim and not self.is_2d:
            set_3d_axis_limits(ax)

        ax.set_xlabel('X')
        # assert not flip_z, 'Why is flip_z=True?'

        if self.is_2d:
            ax.set_ylabel('Y')
        else:
            if zdir == 'y':
                ax.set_ylabel('Z')
                ax.set_zlabel('Y')
            else:
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

        ax.set_facecolor('none')

        if not plot_only_highlighted:
            for i, part in enumerate(self.skel_structure.body):
                if not part:
                    continue

                if hasattr(self.skel_structure, 'colours'):
                    colour = self.skel_structure.colours[i]
                else:
                    colour = 'b'

                x_plot = xyz[part, 0]
                y_plot = xyz[part, 1]

                if self.is_2d:
                    ax.plot(x_plot, y_plot, color=colour, marker='o', markerfacecolor=colour, markersize=joint_size)
                else:
                    z_plot = xyz[part, 2]
                    ax.plot3D(x_plot, y_plot, z_plot, color=colour, marker='o', markerfacecolor=colour, zdir=zdir,
                              markersize=joint_size)

        if highlight_node is not None:
            if not isinstance(highlight_node, list) and not isinstance(highlight_node, tuple):
                highlight_node = [highlight_node]

            for n in highlight_node:
                hp = xyz[n]

                if self.is_2d:
                    ax.scatter(hp[0], hp[1], facecolor='gold', s=joint_size*2, linewidths=joint_size*2,
                               edgecolors='face', alpha=1)
                else:
                    if zdir == 'y':
                        ax.scatter3D(hp[0], hp[2], hp[1], color='gold', zdir=zdir, s=joint_size, linewidths=joint_size,
                                     edgecolors='gold', alpha=1)
                    else:
                        ax.scatter3D(hp[0], hp[1], hp[2], color='gold', zdir=zdir, s=joint_size, linewidths=joint_size,
                                     edgecolors='gold', alpha=1)

        if flip_z and not self.is_2d:
            ax.invert_zaxis()

        if self.is_2d and invert_2d_ax:
            ax.invert_yaxis()

    @staticmethod
    def is_empty(xyz, node_tolerance=None):
        if node_tolerance is None:
            return not xyz.any()
        else:
            n_nodes = xyz.shape[1]
            non_empty_nodes = SkeletonSequence.non_empty_nodes(xyz)
            return n_nodes - non_empty_nodes > node_tolerance

    @staticmethod
    def non_empty_nodes(xyz):
        assert xyz.ndim == 2, 'expected 2d array'
        return xyz.any(axis=0).sum()