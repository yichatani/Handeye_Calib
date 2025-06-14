from datasampler import DataSampler
import json

if __name__ == '__main__':
    # Create a DataSampler object
    config_file_path = "/home/glab/Hardware_WS/src/handeye_calib/src/config_files/trial2.json"
    with open(config_file_path, 'r', encoding='utf-8') as file:
        configs = json.load(file)
    sampler = DataSampler(configs)
    sampler.sample_pose_and_image() 
    sampler.detect_chessboard_and_save()
    print("Sampling complete.")