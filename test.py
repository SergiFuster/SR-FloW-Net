from src.utils.utils import *
from src.models.models import *
from src.models.codes.losses import *

# Dummy function to try the model
# Note: be sure to add images in the project and adapt the s2_path and s3_path to the images paths
# The model will be saved in experiments/, if no folder exists, it will be created
def make_tests():
    s2_path = 'data/images/S2/18.mat'
    s3_path = 'data/images/S3/18.mat'
    s2_image = u.extract_s2mat(s2_path)
    s3_image = u.extract_s3mat(s3_path)

    s3_patch, *(_) = u.extract_central_patch(s3_image, 1024 // 15)
    s2_patch, *(_) = u.extract_central_patch(s2_image, 1024)

    u.show_channels_img(s2_patch, 'Master Input')
    u.show_channels_img(s3_patch, 'Slave Input')

    model = SR_FloW_Net()
    model.train('model_name', s2_patch, s3_patch, 1000, 0.001)
    model.save('experiments/')
    reg, _, s3sr = model.evaluate(s2_patch, s3_patch)

    u.show_channels_img(reg, 'Registered Output')
    u.show_channels_img(s3sr, 'Super-Resolved Output')

if __name__ == '__main__':
    make_tests()