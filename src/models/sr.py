import json, torch, numpy as np
from .codes.model_templates import SRUNetv2
from ..utils import utils as u
import torch.nn.functional as F

def evaluate_from_fullnet(model_path : str, img : np.ndarray, upsamplings=4):

    checkpoint = torch.load(model_path)
    parameters = checkpoint['parameters']

    channels, *(_) = parameters['input_size']

    full_model_state_dict = torch.load(checkpoint['state_dict'])
    
    super_resolution_state_dict = {k.replace('super_resolution.', ''): v 
                               for k, v in full_model_state_dict.items() 
                               if k.startswith('super_resolution.')}

    sr_model_fine_tuned = SRUNetv2(channels)
    sr_model_fine_tuned.load_state_dict(super_resolution_state_dict)


    device = u.get_device()
    xtra = torch.tensor(u.prepare_img_dimensions(img)).to(device)

    # Fine tunned evaluation
    sr_model_fine_tuned.to(device)
    sr_model_fine_tuned.eval()
    with torch.inference_mode():
        s2sr_fine_tuned = sr_model_fine_tuned(xtra, upsamplings)
        s2sr_fine_tuned = s2sr_fine_tuned.cpu().detach().numpy().astype(np.float32)

    
    s2sr_fine_tuned= np.array(s2sr_fine_tuned)[0]
    s2sr_fine_tuned = np.moveaxis(s2sr_fine_tuned, 0, -1)
    return s2sr_fine_tuned

def evaluate(model_path : str, img : np.ndarray, upsamplings=4):

    checkpoint = torch.load(model_path)

    parameters = checkpoint['parameters']

    channels = parameters['channels_used']

    sr_model = SRUNetv2(channels)
    sr_model.load_state_dict(torch.load(checkpoint['state_dict']))

    device = u.get_device()
    xtra = torch.tensor(u.prepare_img_dimensions(img)).to(device)
    
    # Original evaluation
    sr_model.to(device)
    sr_model.eval()
    with torch.inference_mode():
        s2sr = sr_model(xtra, upsamplings)
        s2sr = s2sr.cpu().detach().numpy().astype(np.float32)
    
    s2sr = np.array(s2sr)[0]
    s2sr =  np.moveaxis(s2sr, 0, -1)
    return s2sr