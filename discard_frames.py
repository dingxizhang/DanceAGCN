import os
import json
import random
import argparse
import numpy as np
from tqdm.auto import tqdm

# input_dir is the folder containing preprocessed original json files, which is used to generate frames discarded json files for bcurve
# json_dir is the folder containing raw json files which is used to generate frames discarded raw json files for linear interpolation
# output_dir is the output folder
# For bcurve, run discard_frames.py -> Done
# For linear, run discard_frames.py -> interpolate_missing_keypoints.py -> myprepro.py -> Done
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='/home/dingxi/DanceRevolution/data/all_notwins_01sigma')
# parser.add_argument('--json_dir', type=str, default='/home/dingxi/DanceRevolution/data/json')
parser.add_argument('--output_dir', type=str, default='/home/dingxi/DanceRevolution/data/all_notwins_01sigma_03discard')
parser.add_argument('--discard_ratio', type=float, default=0.3)
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

if not os.path.exists(os.path.join(args.output_dir, 'bcurve')):
    os.mkdir(os.path.join(args.output_dir, 'bcurve'))

if not os.path.exists(os.path.join(args.output_dir, 'linear')):
    os.mkdir(os.path.join(args.output_dir, 'linear'))

if not os.path.exists(os.path.join(args.output_dir, 'frames_list')):
    os.mkdir(os.path.join(args.output_dir, 'frames_list'))

def get_missing_frames_idx(sequence, args):
    # load dance array of the sequence
    with open(os.path.join(args.input_dir, sequence)) as f:
        raw_dict = json.loads(f.read())
        dance = raw_dict['dance_array'] # dance is a list with shape 900(frames)*274(keypoints)
    
    seq_len = len(dance)
    missing_idx = sorted(random.sample(list(range(0, seq_len)), int(seq_len*args.discard_ratio)))
    
    # output missing frames list to npy files
    kept_frames = [item for item in range(0, seq_len) if item not in missing_idx]
    np.save(os.path.join(args.output_dir, 'frames_list', sequence.replace('.json', '.npy')), np.array(kept_frames))

    return missing_idx

def parser_seq_name(sequence):
    info = sequence.strip('.json').split('_')
    dir_name = info[0] + '_1min'
    pose_name = info[2] + '_' + info[3]
    return dir_name, pose_name

def discard_frames(sequence, frames_list, method, args):
    # This function manipulates preprocessed json files
    assert method in ['bcurve', 'linear'], 'Choose what method is it for'

    with open(os.path.join(args.input_dir, sequence)) as f:
        raw_dict = json.loads(f.read())
        dance = raw_dict['dance_array'] # dance is a list with shape 1800(frames)*274(keypoints)
        if method == 'bcurve':
            new_dance = drop_frames(dance, frames_list, args)
        elif method == 'linear':
            new_dance = modify_frames(dance, frames_list, args)
        elif method == 'interpolate':
            new_dance = interpolate(dance)
    
    # dump results into json files
    if method == 'bcurve':
        output_path = os.path.join(args.output_dir, 'bcurve', sequence)
    elif method == 'linear':
        output_path = os.path.join(args.output_dir, 'linear', sequence)

    with open(output_path, 'w') as f:
        new_dict = raw_dict.copy()
        new_dict['dance_array'] = new_dance
        json.dump(new_dict, f)
    
    
def drop_frames(dance, frames_list, args):
    bcurve = dance.copy()
    # Discard target frames listed in missing_idx
    for idx in sorted(frames_list, reverse=True):
        # For bezier curve, just drop the frame, note that this needs to be done in a reverse order
        del bcurve[idx]
    
    return bcurve

def modify_frames(dance, frames_list, args):
    linear = dance.copy()
    # Modify values of target frames listed in missing_idx to -1
    for idx in sorted(frames_list):
        linear[idx] = [-1]*len(linear[idx])
    
    # Use linear method to interpolate
    result = interpolate(linear)
    
    return result

def interpolate(frames, stride=10):
    
    frames = np.array(frames)
    frames = frames[:, :50]
    shape = frames.shape # shape should be (1800, 50)
    frames = frames.reshape(shape[0], 25, 2)
    for i in range(len(frames)):
        # DX: the shape of pose_points is (25, 2) 25 is the number of nodes
        # and for each node we have (x, y)
        pose_points = frames[i]

        for j, point in enumerate(pose_points):
            if point[0] == -1 and point[1] == -1:
                k1 = i
                # DX: k1 is used to locate left (on a time axis) most boundary given a stride
                while k1 > i - stride and k1 >= 0:
                    tmp_point = frames[k1][j]
                    if tmp_point[0] != -1 or tmp_point[1] != -1:
                        break # DX: stop moving if hits frame 0 or limited by stride
                    k1 -= 1

                k2 = i
                # DX: is the right most boundary
                while k2 < i + stride and k2 <= len(frames) - 1:
                    tmp_point = frames[k2][j]
                    if tmp_point[0] != -1 or tmp_point[1] != -1:
                        break
                    k2 += 1

                # DX: The strategy of linear interpolation is like this:
                # if both left and right points exist, use the average of them
                # if one of left (right) point doesn't exist, use right (left) point
                if k1 == -1 and k2 < i + stride:
                    target_right_point = frames[k2][j]
                    point[0] = target_right_point[0]
                    point[1] = target_right_point[1]
                if k1 > i - stride and k2 == len(frames):
                    target_left_point = frames[k1][j]
                    point[0] = target_left_point[0]
                    point[1] = target_left_point[1]

                if (k1 > i - stride and k1 >= 0) and (k2 < i + stride and k2 <= len(frames) - 1):
                    target_left_point = frames[k1][j]
                    target_right_point = frames[k2][j]
                    point[0] = (target_left_point[0] + target_right_point[0]) / 2
                    point[1] = (target_left_point[1] + target_right_point[1]) / 2

                if (k1 > i - stride and k1 >= 0) and (k2 == i + stride and k2 <= len(frames) - 1):
                    target_left_point = frames[k1][j]
                    point[0] = target_left_point[0]
                    point[1] = target_left_point[1]

                if (k1 == i - stride and k1 >= 0) and (k2 < i + stride and k2 <= len(frames) - 1):
                    target_right_point = frames[k2][j]
                    point[0] = target_right_point[0]
                    point[1] = target_right_point[1]
    
    result = frames.reshape(shape[0], 50).tolist()
    return result

def discard_frames_linear_raw_json(sequence, frames_list, args):
    # This functions works on modifying raw json files
    # dir_name (ballet_1min), pose_name(0000_00), json_file(frame000000_keypoints.json)
    dir_name, pose_name = parser_seq_name(sequence)
    pose_path = os.path.join(args.json_dir, dir_name, pose_name)
    json_files = sorted(os.listdir(pose_path)) # 0000_00
    for json_file in json_files:
        frame_idx = int(json_file.strip('_keypoints.json').strip('frame'))
        with open(os.path.join(pose_path, json_file)) as f:
            raw_dict = json.loads(f.read())
        
        linear_dict = raw_dict.copy()
        if frame_idx in frames_list:
            # prepro.py just dropped the confidence score so it should be fine to set everything to -1
            linear_dict['people'][0]['pose_keypoints_2d'] = [-1 for item in raw_dict['people'][0]['pose_keypoints_2d']]
        
        dump_path = os.path.join(args.output_dir, 'linear', dir_name, pose_name)
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)

        with open(os.path.join(dump_path, json_file), 'w') as f:
            json.dump(linear_dict, f)


        
if __name__ == '__main__':
    sequences = os.listdir(args.input_dir)
    for sequence in tqdm(sequences):
        frames_list = get_missing_frames_idx(sequence, args)
        discard_frames(sequence, frames_list, 'bcurve', args)
        discard_frames(sequence, frames_list, 'linear', args)
        # discard_frames_linear_raw_json(sequence, frames_list, args)