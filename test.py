from src.utils.utils import *
from src.models.models import *
from src.models.codes.losses import *
from pprint import pprint
import  os, random
from time import time

master_path = 'data/images/S2/18.mat'
slave_path = 'data/images/S3/18.mat'

master = u.extract_s2mat(master_path)
slave = u.extract_s3mat(slave_path)

master, *(_) = u.extract_central_patch(master, 1024)
slave, *(_) = u.extract_central_patch(slave, 1024//15)

folder = 'src/experiments/fullnet'
files = os.listdir(folder)

results = {}
for file in files:
    model = FullNet(os.path.join(folder, file))
    
    registered, _, _ = model.evaluate(master, slave)
    pprint(u.take_metrics_between_imgs(master, registered))
