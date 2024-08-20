from src.utils.utils import *
from src.models.models import *
from src.models.codes.losses import *
from pprint import pprint
import  os, random
from time import time

folder = 'src/experiments/regnet'

files = os.listdir(folder)
for file in files:
    model = RegNetWrapper(os.path.join(folder, file))
    image = model.history['training'][0]['image']
    master = u.extract_s2mat(f'data/images/S2/{image}')
    slave = u.extract_s3mat(f'data/images/S3/{image}')
    master, *(_) = u.extract_central_patch(master, 5280)
    slave, *(_) = u.extract_central_patch(slave, 352)
    master, slave = master[:, :, -1:], slave[:, :, -1:]
    model.evaluate(master, slave)