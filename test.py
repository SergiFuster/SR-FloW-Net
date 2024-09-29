from src.utils.utils import *
from src.models.models import *
from src.models.codes.losses import *
from pprint import pprint
import  os, random
from time import time

def make_tests():
    s2_path = 'data/images/S2/18.mat'
    s3_path = 'data/images/S3/18.mat'
    s2_image = u.extract_s2mat(s2_path)
    s3_image = u.extract_s3mat(s3_path)

    s3_patch, *(_) = u.extract_central_patch(s3_image, 1024 // 15)
    s2_patch, *(_) = u.extract_central_patch(s2_image, 1024)

    model = SR_FloW_Net('src/experiments/sr_flow_net/SR-FloW-Net-2024-09-29-00-10-57-b1e11ae1-20f2-46bc-950d-bfab5486e6ed.pth')
    reg, _, sr = model.evaluate(s2_patch, s3_patch)
    u.show_channels_img(s2_patch)
    u.show_channels_img(s3_patch)
    u.show_channels_img(reg)
    u.show_channels_img(sr)
if __name__ == '__main__':
    make_tests()