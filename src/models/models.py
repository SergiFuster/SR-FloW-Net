
from collections import defaultdict
import torch.backends.cudnn as cudnn
from ..data.data import *
from .codes.model_templates import *
from .codes.losses import *
from .codes.unets import *
from torch.utils.data import TensorDataset, DataLoader
from ..utils import utils as u
import torch
import json
import time

class FullNetWrapper():
    def __init__(self, model_path=None):
        """
        Parameters
        ----------
        hyperparameters : dict
            Parameters for the model.
            {
                'epochs' : int,
                'learning_rate' : float
            }
        """
        if not model_path:
            self.model = None 
            self.history = {'training' : []}
        else:
            self.model, self.history = u.load_model(model_path)
    
    def save(self, path, name=None):
        """
        Save the model and its history to a path.
        """
        u.save_model(self.model, self.history, path, name)

    def train(self, image_index, epochs=100, learning_rate=0.001, verbose=True, patch_size=1024, kernel=[9,9,2], PCA=False, n_components=1, normalize_pca=False,
                super_resolution_state_dict=None, one_channel=False):

        s2_image = u.extract_s2mat(f'data/images/mat/S2/{image_index}.mat')
        s3_image = u.extract_s3mat(f'data/images/mat/S3/{image_index}.mat')

        s2_patch, *(_) = u.extract_central_patch(s2_image, patch_size) # s2
        s3_patch, *(_) = u.extract_central_patch(s3_image, patch_size // 15) # s3

        if verbose: 
            u.show_channels_img(s2_patch, 'S2 image patch')
            u.show_channels_img(s3_patch, 'S3 image patch')

        if PCA:
            s2_patch_pca = u.get_PCA(s2_patch, n_components, normalize=normalize_pca)
            s3_patch_pca = u.get_PCA(s3_patch, n_components, normalize=normalize_pca)
            
            if verbose:
                u.show_channels_img(s2_patch, 'S2 PCA patch size')
                u.show_channels_img(s3_patch, 'S3 PCA patch size')

        inputs = (s2_patch_pca, s3_patch_pca) if PCA else (s2_patch, s3_patch)
        xtra, ytra = u.prepare_img_dimensions(inputs[0]), u.prepare_img_dimensions(inputs[1])

        if one_channel: xtra, ytra = xtra[:, -1:, :, :], ytra[:, -1:, :, :]

        patches, channels, heigth, width = xtra.shape

        model = FullRegNet((channels, heigth, width), super_resolution_state_dict)

        if verbose: print('-- Model training')

        cudnn.benchmark = True

        device = u.get_device()

        model.to(device)

        model.train()

        learning_rates = [{'epoch' : 0, 'lr' : learning_rate}]

        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, maximize=False)
        print('-- Using Adam optimizer')
        
        loss_functions = [CC3D(kernel_size=kernel).loss, Grad('l2', loss_mult=1).loss, L2().loss]

        weights = [1, 0.5, 0.5]

        xtra, ytra = torch.tensor(xtra).to(device), torch.tensor(ytra).to(device)
        losses = []
        initial_time = time.time()

        best_model = None
        best_loss = None

        print(f'-- Training for {epochs} epochs')
        for epoch in range(epochs):
            
            if epoch == epochs // 2:
                print(f'-- Halfway through training (epoch {epoch})')
                print('-- Lowering learning rate')
                opt = torch.optim.Adam(model.parameters(), lr=learning_rate/10, maximize=False)

            reg, flow, s3sr = model(xtra, ytra)   

            loss = 0
            losses_dict = {}

            curr_loss = loss_functions[0](xtra, reg) * weights[0]
            losses_dict['CC3D'] = f'{curr_loss.item():.6f}'
            loss += curr_loss

            curr_loss = loss_functions[1](xtra, flow) * weights[1]
            losses_dict['GRAD'] = f'{curr_loss.item():.6f}'
            loss += curr_loss

            curr_loss = loss_functions[2](s3sr, ytra) * weights[2]
            losses_dict['L2'] = f'{curr_loss.item():.6f}'
            loss += curr_loss
            losses_dict['TOTAL'] = f'{loss.item():.6f}'

            lr_actual = learning_rate

            loss_info = f'loss: {loss.item()})'
            opt.zero_grad()
            loss.backward()
            opt.step()
            learning_rates.append({'epoch': epoch, 'lr' : lr_actual})

            # Visualization purposes
            losses.append(losses_dict)
            # ------------------------

            epoch_info = f'Epoch: {epoch + 1}'
            print('  '.join((epoch_info, loss_info)), flush=True)

            if not best_model or loss < best_loss: best_model, best_loss = model.state_dict(), loss

        history_item = {
            'epochs' : epochs,
            'learning_rate' : learning_rate,
            'image' : image_index,
            'losses' : losses,
            'weights' : weights,
            'time' : time.time() - initial_time,
            'PCA' : PCA
        }

        self.history['training'].append(history_item)
        self.model = model

    def evaluate(self, master_image : np.ndarray, slave_image : np.ndarray):

        xtra, ytra = u.prepare_img_dimensions(master_image), u.prepare_img_dimensions(slave_image)

        device = u.get_device()

        self.model.to(device)
        self.model.eval()

        xtra, ytra =  torch.tensor(xtra).to(device), torch.tensor(ytra).to(device)

        with torch.inference_mode():
                    
            registered, flow, s3sr = self.model(xtra, ytra)
            registered = registered.cpu().detach().numpy().astype(np.float32)
            flow = flow.cpu().detach().numpy().astype(np.float32)
            xtra = xtra.cpu().detach().numpy().astype(np.float32)
            ytra = ytra.cpu().detach().numpy().astype(np.float32)
            s3sr = s3sr.cpu().detach().numpy().astype(np.float32)

        u.show_results(xtra, ytra, s3sr, registered, flow)

        return registered, xtra, s3sr

    def sweet_evaluate(model_idx : str):
        full_model_state_dict_path = f'./Sergi/models/fullnet/{model_idx}/model.pth'

        with open(f'./Sergi/models/fullnet/{model_idx}/parameters.json', 'r') as archivo:
            parameters = json.load(archivo)

        channels, heigth, width = parameters['input_size']

        full_model_state_dict = torch.load(full_model_state_dict_path)
        
        super_resolution_state_dict = {k.replace('super_resolution.', ''): v 
                                for k, v in full_model_state_dict.items() 
                                if k.startswith('super_resolution.')}

        sr_model_fine_tuned = SRUNetv2(channels)
        sr_model_fine_tuned.load_state_dict(super_resolution_state_dict)

        sr_model = SRUNetv2(channels)
        sr_model.load_state_dict(torch.load('Sergi/models/srunet/9/model.pth'))

        s2_image = u.extract_s2mat(f'images/mat/S2/{parameters["image_index"]}.mat')
        s2_patch, *(_) = u.extract_central_patch(s2_image, 1024)

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        xtra = torch.tensor(u.prepare_img_dimensions(s2_patch)).to(device)
        xtra = F.interpolate(xtra, scale_factor=1/16, mode='bilinear')

        # Fine tunned evaluation
        sr_model_fine_tuned.to(device)
        sr_model_fine_tuned.eval()
        with torch.inference_mode():
            s2sr_fine_tuned = sr_model_fine_tuned(xtra)
            s2sr_fine_tuned = s2sr_fine_tuned.cpu().detach().numpy().astype(np.float32)
        
        # Original evaluation
        sr_model.to(device)
        sr_model.eval()
        with torch.inference_mode():
            s2sr = sr_model(xtra)
            s2sr = s2sr.cpu().detach().numpy().astype(np.float32)

        xtra = xtra.cpu().detach().numpy().astype(np.float32)
        
        s2_down, s2sr_fine_tuned, s2sr = np.array(xtra)[0], np.array(s2sr_fine_tuned)[0], np.array(s2sr)[0]
        s2_down, s2sr_fine_tuned, s2sr =  np.moveaxis(s2_down, 0, -1), np.moveaxis(s2sr_fine_tuned, 0, -1), np.moveaxis(s2sr, 0, -1)
        return s2_down, s2_patch, s2sr_fine_tuned, s2sr
    

class RuNet():
    def __init__(self, model_path=None):
        """
        Parameters
        ----------
        hyperparameters : dict
            Parameters for the model.
            {
                'epochs' : int,
                'learning_rate' : float
            }
        """
        if not model_path:
            self.model = None 
            self.history = {'training' : []}
        else:
            self.model, self.history = u.load_model(model_path)

    def save(self, path, name=None):
        """
        Save the model and its history to a path.
        """
        u.save_model(self.model, self.history, path, name)

    def train(self, data : np.ndarray, epochs : int, learning_rate : float, resolution : int=None, ratio : int=None):
        # region DOCSTRING
        """
        Train the model with the given data for the given resolution or ratio with the given hyperparameters.
        
        Parameters
        ----------
        :param data: expected shape (images, channels, heigth, width)
        :type data: np.ndarray
        :param epochs: number of epochs to train the model
        :type epochs: int
        :param learning_rate: optimizer learning rate
        :type learning_rate: float
        :param resolution: output resolution
        :type resolution: int
        :param ratio: original * ratio = output resolution
        :type ratio: int
        """
        # endregion
        
        # region Assertions
        assert resolution or ratio, "Either resolution or ratio must be provided"
        resolution = resolution if resolution else int(data[2] * ratio)
        assert resolution > data[2], "Output resolution must be greater than the original image resolution"
        assert resolution % 16 == 0, "Output resolution must be a multiple of 16 because of the model architecture"
        # endregion

        batches, channels, heigth, width = data.shape

        model = SRUNet(channels)

        cudnn.benchmark = True

        device = u.get_device()

        model.to(device)

        print('-- Model training')

        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, maximize=False)
        
        loss_function = L2().loss

        initial_time = time.time()

        best_model = None
        best_loss = None

        xtra = torch.tensor(data).to(device)
        tradata = TensorDataset(xtra)
        traloader = DataLoader(dataset=tradata, batch_size=1, shuffle=False)

        print(f'-- Training for {epochs} epochs')
        for epoch in range(epochs):
            
            for iteration, batch in enumerate(traloader):

                input = batch[0].to(device)
                sr = model(input, resolution)

                loss = loss_function(sr, input)

                loss_info = f'loss: {loss.item()})'
                opt.zero_grad()
                loss.backward()
                opt.step()

            epoch_info = f'Epoch: {epoch + 1}'
            print('  '.join((epoch_info, loss_info)), flush=True)

            if not best_model or loss < best_loss: best_model, best_loss = model.state_dict(), loss

        history_item = {
            'epochs' : epochs,
            'data' : data.size,
            'learning_rate' : learning_rate,
            'loss' : best_loss.item(),
            'time' : time.time() - initial_time,
        }

        self.history['training'].append(history_item)
        self.model = model

    def evaluate(self, master_image : np.ndarray, slave_image : np.ndarray):

        xtra, ytra = u.prepare_img_dimensions(master_image), u.prepare_img_dimensions(slave_image)

        device = u.get_device()

        self.model.to(device)
        self.model.eval()

        xtra, ytra =  torch.tensor(xtra).to(device), torch.tensor(ytra).to(device)

        with torch.inference_mode():
                    
            registered, flow, s3sr = self.model(xtra, ytra)
            registered = registered.cpu().detach().numpy().astype(np.float32)
            flow = flow.cpu().detach().numpy().astype(np.float32)
            xtra = xtra.cpu().detach().numpy().astype(np.float32)
            ytra = ytra.cpu().detach().numpy().astype(np.float32)
            s3sr = s3sr.cpu().detach().numpy().astype(np.float32)

        u.show_results(xtra, ytra, s3sr, registered, flow)

        return registered, xtra, s3sr