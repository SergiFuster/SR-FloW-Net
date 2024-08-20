from src.utils.utils import *
from src.models.models import *
from src.models.codes.losses import *
from pprint import pprint
import  os, random
from time import time

sr_state_dict_path = 'src/experiments/runetv2/model-2024-08-15-14-45-55-9680919d-772b-42a9-847e-2d1cf71de850.pth'
save_folder = 'src/experiments/fullnet'
folders = ['data/images/S2', 'data/images/S3']

files = os.listdir(folders[0])
for file in files:
    master, slave = [os.path.join(folder, file) for folder in folders]
    master, slave = u.extract_s2mat(master), u.extract_s3mat(slave)
    master = u.downsample(master, 1024)
    model = FullNet()
    sr_state_dict, _ = u.load_model(sr_state_dict_path)
    model.train(file, master, slave, 10000, 0.0001)
    model.save(save_folder)