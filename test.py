from src.utils.utils import *
from src.models.models import *
from src.models.codes.losses import *
from pprint import pprint
import  os, random
from time import time

folders = ['data/images/S2', 'data/images/S3']
files = os.listdir(folders[0])
for file in files[0:2]:
    master, slave = [os.path.join(folder, file) for folder in folders]
    master, slave = u.extract_s2mat(master), u.extract_s3mat(slave)
    master = u.downsample(master, 1024)
    model = FullNet()
    model.train()