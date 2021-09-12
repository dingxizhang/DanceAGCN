import os
import json
import random
import numpy as np
from tqdm.auto import tqdm

def add_gaussian_noise(x, mu=0, sigma=0.1):
    for i in range(len(x)):
        for j in range(len(x[0])):
            x[i][j] += random.gauss(mu, sigma)*x[i][j]
    return x

def add_gaussian_for_bcurve(input_dir, output_dir, mu=0, sigma=0.1):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    file_list = os.listdir(input_dir)
    for file in tqdm(file_list):
        file_path = os.path.join(input_dir, file)
        
        # read raw data
        with open(file_path, 'r') as f:
            raw = json.loads(f.read())
        dance = raw['dance_array']
        raw['dance_array'] = add_gaussian_noise(dance, mu, sigma)
        output_path = os.path.join(output_dir, file)
        
        # dump jittered data
        with open(output_path, 'w') as f:
            json.dump(raw, f)

def add_gaussian_for_linear(input_dir, output_dir, mu=0, sigma=0.1):
    for music_dir in ['ballet_1min', 'hiphop_1min', 'pop_1min']:
        print(music_dir)
        music_path = os.path.join(input_dir, music_dir)
        dir_names = sorted(os.listdir(music_path))
        for dir_name in tqdm(dir_names):
            pose_path = os.path.join(music_path, dir_name)
            
            json_files = sorted(os.listdir(pose_path))
            for i, json_file in enumerate(json_files):
                with open(os.path.join(pose_path, json_file)) as f:
                    raw_dict = json.loads(f.read())
                
                linear_dict = raw_dict.copy()

                if len(linear_dict['people']) > 0:
                    pose_points = np.array(linear_dict['people'][0]['pose_keypoints_2d']).reshape(25, 3)
                    for point in pose_points:
                        if point[0] == -1 and point[1] == -1:
                            continue
                        else:
                            point[0] += random.gauss(mu, sigma)*point[0]
                            point[1] += random.gauss(mu, sigma)*point[1]
                    pose_points = pose_points.reshape(-1).tolist()
                    linear_dict['people'][0]['pose_keypoints_2d'] = pose_points
                    dump_path = os.path.join(output_dir, music_dir, dir_name)
                    if not os.path.exists(dump_path):
                        os.makedirs(dump_path)
                    with open(os.path.join(dump_path, json_file), 'w') as f:
                        json.dump(linear_dict, f)
                    

if __name__ == '__main__':
    input_dir = '/home/dingxi/DanceRevolution/data/all_1min_05discard_notwins/linear'
    output_dir = '/home/dingxi/DanceRevolution/data/all_1min_05discard_notwins/linear_gaussian'
    # add_gaussian_for_bcurve(input_dir, output_dir)
    add_gaussian_for_linear(input_dir, output_dir)