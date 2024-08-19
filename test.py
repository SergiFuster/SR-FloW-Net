from src.utils.utils import *
from src.models.models import *
from src.models.codes.losses import *
import  os, random
loss_functions = [
    NCC(),
    CC3D(kernel_size=[9, 9, 2]),
    LNCC3D()
    
]

folder_save = 'src/experiments/fullnet'
models_folder = 'src/experiments/runetv2'
file = os.listdir(models_folder)[-1]
for loss_function in loss_functions:
    model = FullNet()
    sr_state_dict, _ = u.load_model(os.path.join(models_folder, file))
    model.train('18', 1000, 0.001, 1024, super_resolution_state_dict=sr_state_dict, loss_function=loss_function)
    model.save(folder_save)
