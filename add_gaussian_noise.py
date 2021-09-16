import os
import json
import random
import numpy as np
from tqdm.auto import tqdm

def add_gaussian_noise(x, mu=0, sigma=0.1):
    x += random.gauss(mu, sigma)*x
    return x

def add_gaussian_for_bcurve(input_dir, output_dir, mu=0, sigma=0.1):
    # DX: this function is to add Gaussian noise to preprocessed json files   
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    file_list = os.listdir(input_dir)
    for file in tqdm(file_list):
        file_path = os.path.join(input_dir, file)
        
        # read raw data
        with open(file_path, 'r') as f:
            raw = json.loads(f.read())
        
        dance = np.array(raw['dance_array'])[:, :50].reshape(-1, 25, 2)
        for i in range(len(dance)):
            for j in range(len(dance[0])):
                if dance[i][j][0] == -1 and dance[i][j][1] == -1:
                    continue
                else:
                    dance[i][j][0] = add_gaussian_noise(dance[i][j][0], mu, sigma)
                    dance[i][j][1] = add_gaussian_noise(dance[i][j][1], mu, sigma)
        
        raw['dance_array'] = dance.reshape(-1, 50).tolist()
        output_path = os.path.join(output_dir, file)
        
        # dump jittered data
        with open(output_path, 'w') as f:
            json.dump(raw, f)

def add_gaussian_for_linear(input_dir, output_dir, mu=0, sigma=0.1):
    # DX: this function is to add Gaussian noise to raw json files
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
    input_dir = '/home/dingxi/DanceRevolution/data/all_1min_notwins'
    output_dir = '/home/dingxi/DanceRevolution/data/all_01sigma'
    add_gaussian_for_bcurve(input_dir, output_dir, sigma=0.1)