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
parser.add_argument('--input_dir', type=str, default='/home/dingxi/DanceRevolution/data/all_1min_notwins')
parser.add_argument('--json_dir', type=str, default='/home/dingxi/DanceRevolution/data/json')
parser.add_argument('--output_dir', type=str, default='/home/dingxi/DanceRevolution/data/all_1min_05discard_notwins')
parser.add_argument('--discard_ratio', type=float, default=0.5)
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
    missing_idx = sorted(random.sample(list(range(0, seq_len-1)), int(seq_len*args.discard_ratio)))
    
    # output missing frames list to npy files
    np.save(os.path.join(args.output_dir, 'frames_list', sequence.replace('.json', '.npy')), np.array(missing_idx))

    return missing_idx

def parser_seq_name(sequence):
    info = sequence.strip('.json').split('_')
    dir_name = info[0] + '_1min'
    pose_name = info[2] + '_' + info[3]
    return dir_name, pose_name

def discard_frames_bcurve(sequence, frames_list, args):
    with open(os.path.join(args.input_dir, sequence)) as f:
        raw_dict = json.loads(f.read())
        dance = raw_dict['dance_array'] # dance is a list with shape 900(frames)*274(keypoints)
        bcurve_dance = drop_frames(dance, frames_list, args)
    
    # dump results into json files
    with open(os.path.join(args.output_dir, 'bcurve', sequence), 'w') as f:
        bcurve_dict = raw_dict.copy()
        bcurve_dict['dance_array'] = bcurve_dance
        json.dump(bcurve_dict, f)
    
    
def drop_frames(dance, frames_list, args):
    bcurve = dance.copy()
    # Discard target frames listed in missing_idx
    for idx in sorted(frames_list, reverse=True):
        # For bezier curve, just drop the frame, note that this needs to be done in a reverse order
        del bcurve[idx]
    
    return bcurve

def discard_frames_linear(sequence, frames_list, args):
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
        discard_frames_bcurve(sequence, frames_list, args)
        discard_frames_linear(sequence, frames_list, args)