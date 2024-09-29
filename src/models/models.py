import torch.nn as nn
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

class SR_FloW_Net():
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
        u.save_model(self.model, self.history, path, 'SR-FloW-Net', model_name=name)

    def train(self, image : str, master : np.ndarray, slave : np.ndarray, epochs=100, learning_rate=0.001, PCA=False, n_components=1, super_resolution_state_dict=None, loss_function=CC3D([9, 9, 2])):
        # region DOCSTRING
        """
        Trains a deep learning model for image registration using a 3D Convolutional Neural Network (CNN) with optional Principal Component Analysis (PCA) preprocessing.

        Parameters
        ----------
        image : str
            The identifier for the input image or the corresponding dataset.
        master : np.ndarray
            The master image or fixed image in the registration process, represented as a NumPy array.
        slave : np.ndarray
            The slave image or moving image in the registration process, represented as a NumPy array.
        epochs : int, optional, default=100
            The number of epochs to train the model.
        learning_rate : float, optional, default=0.001
            The initial learning rate for the optimizer.
        PCA : bool, optional, default=False
            If True, apply Principal Component Analysis (PCA) to reduce dimensionality of the input images.
        n_components : int, optional, default=1
            The number of principal components to keep if PCA is applied.
        super_resolution_state_dict : dict or None, optional, default=None
            A dictionary containing pre-trained weights for super-resolution, if any.
        loss_function : callable, optional, default=LNCC3D
            The primary loss function for training, typically used for measuring the similarity between the registered images.

        Returns
        -------
        None
            This method trains the model in place and does not return anything. The trained model's state dictionary is stored in `self.model`, and training history is updated in `self.history['training']`.

        Notes:
        -----
        - If multiple GPUs are available, the model will be trained using `DataParallel` to distribute the workload.
        - The training process includes calculating three different losses: the primary similarity loss (e.g., LNCC3D), a gradient loss, and an L2 loss for the super-resolution output.
        - The training process monitors and saves the best model (based on loss) during the training loop.
        - Training history, including losses and learning rates across epochs, is recorded and stored in `self.history`.
        - This method assumes that the utility functions such as `get_PCA`, `prepare_img_dimensions`, and `get_device` are implemented elsewhere in the codebase and are accessible via the `u` namespace.
        - The optimizer used for training is Adam, with a learning rate that remains constant throughout the training process, unless modified externally.

        Example:
        -------
        >>> model = FullNet()
        >>> trainer.train(image="01", master=np_array_master, slave=np_array_slave, epochs=50, learning_rate=0.0005)

        """
        # endregion

        xtra, ytra = map(u.prepare_img_dimensions, [master, slave])

        _, channels, heigth, width = xtra.shape

        model = FullRegNet((channels, heigth, width), super_resolution_state_dict)

        if u.multi_gpu():
            print('Multiple GPUs detected, using DataParallel')
            model = nn.DataParallel(model)

        print('-- Model training')

        cudnn.benchmark = True

        device = u.get_device()

        model.to(device)

        model.train()

        learning_rates = [{'epoch' : 0, 'lr' : learning_rate}]

        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, maximize=False)
        print('-- Using Adam optimizer')
        
        loss_functions = [loss_function.loss, Grad('l2', loss_mult=1).loss, L2().loss]

        weights = [1, 0.5, 0.5]

        xtra, ytra = torch.tensor(xtra).to(device), torch.tensor(ytra).to(device)
        losses = []
        initial_time = time.time()

        best_model = None
        best_loss = None

        print(f'-- Training for {epochs} epochs')
        for epoch in range(epochs):

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
            'image' : image,
            'losses' : losses,
            'weights' : weights,
            'loss_function' : loss_function.__str__(),
            'loss' : best_loss.item(),
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

        # u.show_results(xtra, ytra, s3sr, registered, flow)
        registered, xtra, s3sr = list(map(u.undo_tensor_format, [registered, xtra, s3sr]))
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

class DUnet():
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
        u.save_model(self.model, self.history, path=path, model_name=name, prefix='DUnet')

    def train(self, image : str, master : np.ndarray, slave : np.ndarray, epochs=100, learning_rate=0.001, loss_function=CC3D([9, 9, 2])):
        # region DOCSTRING
        """
        Trains a deep learning model for image registration using a 3D Convolutional Neural Network (CNN) with optional Principal Component Analysis (PCA) preprocessing.

        Parameters
        ----------
        image : str
            The identifier for the input image or the corresponding dataset.
        master : np.ndarray
            The master image or fixed image in the registration process, represented as a NumPy array.
        slave : np.ndarray
            The slave image or moving image in the registration process, represented as a NumPy array.
        epochs : int, optional, default=100
            The number of epochs to train the model.
        learning_rate : float, optional, default=0.001
            The initial learning rate for the optimizer.
        PCA : bool, optional, default=False
            If True, apply Principal Component Analysis (PCA) to reduce dimensionality of the input images.
        n_components : int, optional, default=1
            The number of principal components to keep if PCA is applied.
        super_resolution_state_dict : dict or None, optional, default=None
            A dictionary containing pre-trained weights for super-resolution, if any.
        loss_function : callable, optional, default=LNCC3D
            The primary loss function for training, typically used for measuring the similarity between the registered images.

        Returns
        -------
        None
            This method trains the model in place and does not return anything. The trained model's state dictionary is stored in `self.model`, and training history is updated in `self.history['training']`.

        Notes:
        -----
        - If multiple GPUs are available, the model will be trained using `DataParallel` to distribute the workload.
        - The training process includes calculating three different losses: the primary similarity loss (e.g., LNCC3D), a gradient loss, and an L2 loss for the super-resolution output.
        - The training process monitors and saves the best model (based on loss) during the training loop.
        - Training history, including losses and learning rates across epochs, is recorded and stored in `self.history`.
        - This method assumes that the utility functions such as `get_PCA`, `prepare_img_dimensions`, and `get_device` are implemented elsewhere in the codebase and are accessible via the `u` namespace.
        - The optimizer used for training is Adam, with a learning rate that remains constant throughout the training process, unless modified externally.

        Example:
        -------
        >>> model = FullNet()
        >>> trainer.train(image="01", master=np_array_master, slave=np_array_slave, epochs=50, learning_rate=0.0005)

        """
        # endregion

        xtra, ytra = map(u.prepare_img_dimensions, [master, slave])

        _, channels, heigth, width = xtra.shape

        model = SRegNet((channels, heigth, width))

        print('-- Model training')

        cudnn.benchmark = True

        device = u.get_device()

        model.to(device)

        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, maximize=False)
        print('-- Using Adam optimizer')
        
        loss_functions = [loss_function.loss, Grad('l2', loss_mult=1).loss]

        weights = [1, 0.5, 0.5]

        xtra, ytra = torch.tensor(xtra).to(device), torch.tensor(ytra).to(device)
        initial_time = time.time()

        best_model = None
        best_loss = None

        print(f'-- Training for {epochs} epochs')
        for epoch in range(epochs):

            reg, flow = model(xtra, ytra)   

            loss = 0

            loss += loss_functions[0](xtra, reg) * weights[0]

            loss += loss_functions[1](xtra, flow) * weights[1]

            loss_info = f'loss: {loss.item()})'
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_info = f'Epoch: {epoch + 1}'
            print('  '.join((epoch_info, loss_info)), flush=True)

            if not best_model or loss < best_loss: best_model, best_loss = model.state_dict(), loss

        history_item = {
            'epochs' : epochs,
            'learning_rate' : learning_rate,
            'image' : image,
            'weights' : weights,
            'loss_function' : loss_function.__str__(),
            'loss' : best_loss.item(),
            'time' : time.time() - initial_time
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
                    
            registered, flow = self.model(xtra, ytra)
            registered = registered.cpu().detach().numpy().astype(np.float32)

        # u.show_results(registered)
        registered = u.undo_tensor_format(registered)
        return registered


class RuNet():
    def __init__(self, model_path=None):
        """
        Parameters
        ----------
        model_path : str
            Path to the model to load.
            If None, a new model will be created.
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
        u.save_model(self.model, self.history, path, name, prefix='RUNet1')

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
        resolution = resolution if resolution else int(data.shape[2] * ratio)
        assert resolution > data.shape[2], "Output resolution must be greater than the original image resolution"
        assert resolution % 16 == 0, "Output resolution must be a multiple of 16 because of the model architecture"
        # endregion

        batches, channels, heigth, width = data.shape

        model = SRUNet(channels)

        cudnn.benchmark = True

        device = u.get_device(1)

        """ if u.multi_gpu():
            print('Multiple GPUs detected, using DataParallel')
            model = nn.DataParallel(model) """

        model.to(device)

        print('-- Model training')

        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, maximize=False)
        
        loss_function = L2().loss

        initial_time = time.time()

        best_model = None
        best_loss = None

        xtra = torch.tensor(data).float().to(device)
        tradata = TensorDataset(xtra)
        traloader = DataLoader(dataset=tradata, batch_size=1, shuffle=True)

        print(f'-- Training for {epochs} epochs')
        for epoch in range(epochs):
            
            loss = 0

            for iteration, batch in enumerate(traloader):

                input = batch[0].to(device)
                output = model(input, resolution)

                loss = loss_function(output, input)

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

    def evaluate(self, image : np.ndarray, resolution : int, ratio : int) -> np.ndarray:

        assert resolution or ratio, 'Both resolution and ratio are None, at least one must be a valid value'
        
        resolution = resolution if resolution else ratio

        device = u.get_device()

        self.model.to(device)
        self.model.eval()

        input = u.prepare_img_dimensions
        input = torch.tensor(input).to(device)

        with torch.inference_mode():
                    
            sr = self.model(input, resolution)

            sr = sr.cpu().detach().numpy().astype(np.float32)

        sr = u.undo_tensor_format(sr)
        return sr
    
class RuNetv2():
    def __init__(self, model_path=None):
        """
        Parameters
        ----------
        model_path : str
            Path to the model to load.
            If None, a new model will be created.
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
        u.save_model(self.model, self.history, path, name, prefix='RUNet2')

    def train(self, data : np.ndarray, epochs : int, learning_rate : float, upsamplings : int):
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
        :param upsampling: [1, 4] output resolution = 2^upsampling * original image resolution
        :type upsampling: int
        """
        # endregion

        batches, channels, heigth, width = data.shape

        model = SRUNetv2(channels)

        cudnn.benchmark = True

        device = u.get_device()

        if u.multi_gpu():
            print('Multiple GPUs detected, using DataParallel')
            model = nn.DataParallel(model)

        model.to(device)

        print('-- Model training')

        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, maximize=False)
        
        loss_function = L2().loss

        initial_time = time.time()

        best_model = None
        best_loss = None

        xtra = torch.tensor(data).float().to(device)
        tradata = TensorDataset(xtra)
        traloader = DataLoader(dataset=tradata, batch_size=1, shuffle=True)

        print(f'-- Training for {epochs} epochs')
        for epoch in range(epochs):
            
            loss = 0

            for iteration, batch in enumerate(traloader):

                input = batch[0].to(device)
                output = model(input, upsamplings)

                loss = loss_function(output, input)

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

    def evaluate(self, image : np.ndarray, resolution : int, ratio : int) -> np.ndarray:

        assert resolution or ratio, 'Both resolution and ratio are None, at least one must be a valid value'
        
        resolution = resolution if resolution else ratio

        device = u.get_device()

        self.model.to(device)
        self.model.eval()

        input = u.prepare_img_dimensions
        input = torch.tensor(input).to(device)

        with torch.inference_mode():
                    
            sr = self.model(input, resolution)

            sr = sr.cpu().detach().numpy().astype(np.float32)

        sr = u.undo_tensor_format(sr)
        return sr

class FloU_Net():
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

        u.save_model(self.model, self.history, path=path, model_name=name, prefix='FloU-Net')

    def train(self, image : str, master : np.ndarray, slave : np.ndarray, epochs=100, learning_rate=0.001):
        """
        Trains a deep learning model for image registration using a 3D Convolutional Neural Network (CNN) with optional Principal Component Analysis (PCA) preprocessing.

        Parameters
        ----------
        image : str
            The identifier for the input image or the corresponding dataset.
        master : np.ndarray
            The master image or fixed image in the registration process, represented as a NumPy array.
        slave : np.ndarray
            The slave image or moving image in the registration process, represented as a NumPy array.
        epochs : int, optional, default=100
            The number of epochs to train the model.
        learning_rate : float, optional, default=0.001
            The initial learning rate for the optimizer.

        Returns
        -------
        None
            This method trains the model in place and does not return anything. The trained model's state dictionary is stored in `self.model`, and training history is updated in `self.history['training']`.

        Notes:
        -----
        - If multiple GPUs are available, the model will be trained using `DataParallel` to distribute the workload.
        - The training process includes calculating three different losses: the primary similarity loss (e.g., LNCC3D), a gradient loss, and an L2 loss for the super-resolution output.
        - The training process monitors and saves the best model (based on loss) during the training loop.
        - Training history, including losses and learning rates across epochs, is recorded and stored in `self.history`.
        - This method assumes that the utility functions such as `get_PCA`, `prepare_img_dimensions`, and `get_device` are implemented elsewhere in the codebase and are accessible via the `u` namespace.
        - The optimizer used for training is Adam, with a learning rate that remains constant throughout the training process, unless modified externally.

        Example:
        -------
        >>> model = RegNet()
        >>> model.train(image="01", master=np_array_master, slave=np_array_slave, epochs=50, learning_rate=0.0005)

        """

        xtra, ytra = map(u.prepare_img_dimensions, [master, slave])


        _, channels, heigth, width = ytra.shape

        if channels > 1: xtra, ytra = xtra[:, -1:, :, :], ytra[:, -1:, :, :]

        ratio = xtra.shape[2]//ytra.shape[2]
        model = RegNet((heigth, width), RATIO=ratio)

        if u.multi_gpu():
            print('Multiple GPUs detected, using DataParallel')
            model = nn.DataParallel(model)

        print('-- Model training')

        cudnn.benchmark = True

        device = u.get_device()

        model.to(device)

        model.train()

        learning_rates = [{'epoch' : 0, 'lr' : learning_rate}]

        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, maximize=False)
        print('-- Using Adam optimizer')
        
        loss_functions = [NCC(scale=ratio).loss, Grad('l2', loss_mult=1).loss]

        weights = [1, 0.5]

        xtra, ytra = torch.tensor(xtra).to(device), torch.tensor(ytra).to(device)
        losses = []
        initial_time = time.time()

        best_model = None
        best_loss = None

        print(f'-- Training for {epochs} epochs')
        for epoch in range(epochs):

            reg, flow = model(xtra, ytra)
            loss = 0
            losses_dict = {}

            curr_loss = loss_functions[0](xtra, reg) * weights[0]
            losses_dict['SIM'] = f'{curr_loss.item():.6f}'
            loss += curr_loss

            curr_loss = loss_functions[1](xtra, flow) * weights[1]
            losses_dict['SMOOTH'] = f'{curr_loss.item():.6f}'
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
            'image' : image,
            'losses' : losses,
            'weights' : weights,
            'loss_function' : NCC().__str__(),
            'loss' : best_loss.item(),
            'time' : time.time() - initial_time
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
                    
            registered, flow = self.model(xtra, ytra, registration=True)
            registered = registered.cpu().detach().numpy().astype(np.float32)

        # u.show_results(xtra, ytra, registered)
        registered = u.undo_tensor_format(registered)
        return registered