import numpy as np, h5py, cv2, torch, shutil, os, sys, json, uuid, src.metrics.metrics as m
from datetime import datetime
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn as nn
from time import time
from matplotlib import colors
# from sklearn.decomposition import PCA

# Obtén la ruta del directorio actual del script
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Agrega el directorio padre al sys.path
directorio_padre = os.path.join(directorio_actual, '..')
sys.path.append(directorio_padre)

PLT_COLORS = plt.get_cmap('tab10')

def get_files_path(folder, endswith="", startswith=""):
    """
    Return the relative routes of files in folder filtered by endswith and startswith
    """
    return [f'{folder}/{file}' for file in os.listdir(folder) if os.path.isfile(f'{folder}/{file}') and file.endswith(endswith) and file.startswith(startswith)]

def undo_tensor_format(tensor : np.ndarray):
    return np.moveaxis(tensor[0], 0, -1)

def get_colored_optic_flow(optical_flow):

    # Calcula la magnitud y dirección del flujo óptico
    magnitude = np.sqrt(optical_flow[0]**2 + optical_flow[1]**2)
    angle = np.arctan2(optical_flow[1], optical_flow[0])

    # Normaliza la magnitud para usarla como intensidad en la representación de color
    magnitude_normalized = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))

    # Convierte la dirección del ángulo a valores de color HSV
    hue = (angle + np.pi) / (2 * np.pi)
    saturation = np.ones_like(hue)
    value = magnitude_normalized

    # Combina los valores HSV en una matriz de color RGB
    rgb_image = colors.hsv_to_rgb(np.stack([hue, saturation, value], axis=-1))

    return rgb_image

def normalize_image(image):
    x_min = np.min(image)
    x_max = np.max(image)

    return (image - x_min) / (x_max - x_min)

def normalize_image_by_channel(image):
    for i in range(image.shape[2]):
        image[:, :, i] = normalize_image(image[:, :, i])

    return image


def split_in_quarters(img: np.ndarray) -> np.ndarray:
    """
    Split an image into four equal parts.

    This function takes an image with dimensions (width, height, channels)
    and divides it into four equal parts, returning an array with the
    four sub-images.

    Parameters
    ----------
    img : np.ndarray
        The input image with shape (height, width, channels).

    Returns
    -------
    np.ndarray
        An array of four sub-images with shape (4, height//2, width//2, channels).

    Examples
    --------
    >>> img = np.random.rand(100, 100, 3)
    >>> quarters = split_in_quarters(img)
    >>> quarters.shape
    (4, 50, 50, 3)
    """
    h, w, _ = img.shape
    return np.array([img[:h//2, :w//2, :], img[:h//2, w//2:, :], img[h//2:, :w//2, :], img[h//2:, w//2:, :]])

def load_history(path):
    return torch.load(path, map_location=torch.device('cpu'))['history']

def extract_history_data(history):
    return history['training'][0]['data']

def extract_history_loss(history):
    return history['training'][0]['loss']

def extract_history_time(history):
    return history['training'][0]['time']

def show_image(img, title):
        """
        Displays an 2D image
        """
        fig = plt.figure()
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(title)
        plt.show(block=False)

def multi_gpu():
    return torch.cuda.device_count() > 1

def get_device(gpu=0):
    # Verifica si CUDA (GPU) está disponible
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        # Imprime información sobre la GPU
        print(torch.cuda.get_device_name(0))
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("No se detectó una GPU. PyTorch se ejecutará en la CPU.")
    
    return device

def from_tensor_to_image(tensor):
    tensor = tensor[0, :, :, :]
    tensor = np.moveaxis(tensor, 0, -1)
    return tensor

def extract_s3mat(name):
    f = h5py.File(name,'r')
    s3 = f.get('RGB_S3') # *RGBA FORMAT*
    s3 = np.array(s3)
    s3 = s3.astype(np.float32)
    s3 = np.moveaxis(s3, 0, -1) 
    return s3

def extract_s2mat(name):
    f = h5py.File(name,'r')
    s2 = f.get('RGB_S2')
    s2 = np.array(s2)
    s2 = s2.astype(np.float32)
    s2 = np.moveaxis(s2, 0, -1) 
    return s2

def extract_s3gtmat(name):
    f = h5py.File(name,'r')
    s3gt = f.get('S3_SIM')
    s3gt = np.array(s3gt)
    s3gt = s3gt.astype(np.float32)
    s3gt = np.moveaxis(s3gt, 0, -1)
    return s3gt

def save_jpg(arr, output_path):

    array_uint8 = (arr * 255).astype(np.uint8)[:, :, :3]

    # Crear una imagen a partir del array
    imagen = Image.fromarray(array_uint8, 'RGB')

    # Guardar la imagen como un archivo JPEG
    imagen.save(output_path)

def extract_central_patch(img, size, pytorch=False) -> tuple[np.ndarray, int, int, int, int]:
    """
    if pytorch: input format -> (batch, bands, heigth, width)
    else: input format -> (heigth, width, channels)
    """

    # Get image dimensions
    if pytorch: batches, channels, heigth, width = img.shape
    else:
        heigth, width, *(_) = img.shape

    # Central image point coords
    cx = width // 2
    cy = heigth // 2

    # Calculate the area of interest
    ix = max(0, cx - size // 2)
    fx = min(width, cx + size // 2)
    iy = max(0, cy - size // 2)
    fy = min(heigth, cy + size // 2)

    # Extract central patch
    patch = img[:, :, iy:fy, ix:fx] if pytorch else img[iy:fy, ix:fx, ...]

    return patch, ix, fx, iy, fy

def patchetize(tensor, patches):
    """Divide a tensor in (patches*patches) patches"""
    h, w = tensor.shape[2], tensor.shape[3]
    assert h % patches == 0 and w % patches == 0, "The image dimensions must be divisible by the number of patches"

    patch_h, patch_w = h // patches, w // patches

    patches = tensor.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)

    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

    return patches.view(-1, tensor.size(1), patch_h, patch_w)

def extract_random_sized_patch(s2, s3, size, pytorch=False):

    """We assume that size is in relation to s3"""

    # Get image dimensions
    if pytorch: batches, channels, heigth, width = s3.shape
    else: heigth, width, _ = s3.shape

    # Max coords
    maxx = width - size
    maxy = heigth - size

    # Calculate the area of interest
    ix = np.random.randint(0, maxx)
    fx = ix + size
    iy = np.random.randint(0, maxy)
    fy = iy + size

    # Extract patch
    patch_s3 = s3[:, :, iy:fy, ix:fx] if pytorch else s3[iy:fy, ix:fx, :]

    ratio = s2.shape[2] // s3.shape[2] if pytorch else s2.shape[0] // s3.shape[0]

    ix, fx, iy, fy = ix*ratio, fx*ratio, iy*ratio, fy*ratio

    patch_s2 = s2[:, :, iy:fy, ix:fx] if pytorch else s2[iy:fy, ix:fx, :]
    
    return patch_s2, patch_s3, ix/ratio, fx/ratio, iy/ratio, fy/ratio

def visualize_tensor(tensor, title=''):
    for i in range(tensor.shape[0]):
        patch = tensor[i, :, :, :]
        patch = (np.moveaxis(patch, 0, -1)* 255).astype(np.uint8)
        plt.imshow(patch)
        plt.title(title)
        plt.show()

def extract_patch(img, patch_coords, pytorch=False):
    """
    Extract personalized patch from an image represented as a numpy array 
    or as a tensor
    """

    # Get image dimensions
    if pytorch: batches, channels, heigth, width = img.shape
    else: 
        if img.shape == 3: heigth, width, _ = img.shape
        else: heigth, width = img.shape

    iy, fy, ix, fx = patch_coords
    patch = img[:, :, iy:fy, ix:fx] if pytorch else img[iy:fy, ix:fx, ...]
    
    return patch, ix, fx, iy, fy

def prepare_img_dimensions(img):

    """Image with dimensions (height, width, channels) is converted to (patch, channels, height, width)"""
    img = img[np.newaxis, :, :, :] # (patches, heigth, width, channels)

    img = np.moveaxis(img, -1, 1) # Move channels to second position for pytorch

    return img

def show_channels_img(img, title=''):
    """For show channels from an image"""
    channels = img.shape[-1]
    # Figure with all channels
    fig, ax = plt.subplots(1, channels)
    fig.suptitle(title)
    for i in range(channels):
        if channels > 1:
            ax[i].imshow(img[:,:,i], cmap='gray', vmin=0, vmax=1)
            ax[i].set_title(f'Channel {i}')
        else:
            ax.imshow(img[:,:,i], cmap='gray')
            ax.set_title(f'Channel {i}')
    plt.show()

def downsample(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

def get_metrics():
    metrics =[
        m.rmse,
        m.nrmse,
        m.psnr,
        m.ssim_index,
        m.ergas,
        m.q_index,
        m.mi
    ]

    titles = [
    "RMSE (Error Cuadrático Medio de la Raíz)",
    "NRMSE (Error Cuadrático Medio Normalizado)",
    "PSNR (Relación Señal a Ruido de Pico)",
    "SSIM (Índice de Similitud Estructural)",
    "ERGAS (Error Relativo Global de la Abundancia Espectral)",
    "Índice de calidad (Q)",
    "Información mutua (MI)"
    ]
    return metrics, titles

def take_metrics_between_imgs(img1, img2, verbose=True):
    """Take all metrics from 2 images"""
    metrics, titles = get_metrics()

    pairs = zip(metrics, titles)
    results = []
    for pair in pairs:
        metric, title = pair
        if title == 'SSIM (Índice de Similitud Estructural)':
            res = metric(img1, img2, multichannel=True)
        else:
            res = metric(img1, img2)
        if verbose: print(title)
        if verbose: print(res)
        results.append(res)
    return results

def move_files(_from, _to):
    if not os.path.exists(_to):
        os.makedirs(_to)
    
    files = os.listdir(_from)

    for file in files:
        origin = os.path.join(_from, file)
        destiny = os.path.join(_to, file)
        shutil.move(origin, destiny)
        print(f'File {file} moved to {destiny}')

def unique_name():
    """Generates a unique name constitued by date and uuid"""
    return f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-{uuid.uuid4()}'

def remove_dataparallel_wrapper(state_dict):
    """
    Remove the DataParallel wrapper from the model state dictionary.

    Parameters
    ----------
    state_dict : dict
        The model state dictionary.

    Returns
    -------
    dict
        The model state dictionary without the DataParallel wrapper.
    """
    # Check if sr_state_dict is actually a DataParallel object
    if isinstance(state_dict, torch.nn.DataParallel):
        # Unwrap the model from DataParallel to access the actual state_dict
        state_dict = state_dict.module.state_dict()
    return state_dict


def load_model(path):
    """
    Load the model checkpoint and return the model and the history.

    Parameters
    ----------
    path : str
        The path to the model checkpoint.

    Returns
    -------
    dict
        A dictionary with the model and the log.
    """
    try:
        checkpoint = torch.load(path)
        print(f'Model loaded from {path}')
        return checkpoint['state_dict'], checkpoint['history']
    except Exception as e:
        print(f'Error loading model from {path}')
        print(e)

def save_model(state_dict, history, path, prefix, model_name=None):
    """
    Save the model checkpoint with the log and parameters in the given path.

    checkpoing = {
        'state_dict': state_dict,
        'history': history
    }

    Parameters
    ----------
    state_dict : dict
        The model state dictionary.
    history : dict
        A dictionar with every training information as epochs, learning rate, etc.
    path : str
        The path to save the model checkpoint.
    model_name : str, optional
        The model name to save the checkpoint. If None, a unique name will be given.

    Returns
    -------
    None
    """
    if not model_name: model_name = f'{prefix}-{unique_name()}.pth'

    path = os.path.join(path, model_name)
    
    checkpoint = {
        'state_dict': state_dict,
        'history': history
    }

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True) # Create the directory if it doesn't exist
        torch.save(checkpoint, path)
        print(f'Model saved in {path}')
    except Exception as e:
        print(f'Error saving model in {path}')
        print(e)

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight, a=-0.01, b=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def is_img_shape(volume):
    """
    Usually input images for model has the shape (batch, channels, heigth, width)
    
    But we need (heigth, width, channels) for visualization
    """
    return len(volume.shape) <= 3

def transform_to_img_shape(volume):
    """
    Usually input tensors has the shape (1, channels, heigth, width) and we want (heigth, width, channels)
    """
    return np.moveaxis(volume[0], 0, -1)

def mix_images_chessboard(image1: np.ndarray, image2: np.ndarray, n: int) -> np.ndarray:
    # Ensure images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same dimensions.")
    
    # Image dimensions
    size = image1.shape
    
    height = size[0]
    width = size[1]

    # Calculate the size of each square
    square_height = height // n
    square_width = width // n
    
    # Initialize the result image
    mixed_image = np.zeros_like(image1)
    
    # Create the checkerboard pattern
    for i in range(n):
        for j in range(n):
            # Determine the start and end points for the current square
            y_start = i * square_height
            y_end = (i + 1) * square_height if i < n - 1 else height
            x_start = j * square_width
            x_end = (j + 1) * square_width if j < n - 1 else width
            
            # Alternate between image1 and image2
            if (i + j) % 2 == 0:
                mixed_image[y_start:y_end, x_start:x_end] = image1[y_start:y_end, x_start:x_end]
            else:
                mixed_image[y_start:y_end, x_start:x_end] = image2[y_start:y_end, x_start:x_end]
    
    return mixed_image

def show_in_rows(*args):
    """Show images in rows"""
    white_box = np.ones(args[0].shape[:2])
    cols = max([img.shape[2] for img in args])
    fig, axes = plt.subplots(len(args), cols, sharex=True, sharey=True)
    for row, img in enumerate(args):
        for col in range(cols):
            if col < img.shape[2]:
                axes[row, col].imshow(img[:, :, col], cmap='gray', vmin=0, vmax=1)
            else:
                axes[row, col].imshow(white_box, cmap='gray', vmin=0, vmax=1)
    plt.show()

def show_results(*args):
    """Show the results of the registration"""

    fig, axes = plt.subplots(1, len(args), sharex=True, sharey=True)
    fig.suptitle('Results')

    for i, img in enumerate(args):
        if not is_img_shape(img): img = transform_to_img_shape(img)
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.show()

def get_PCA(img, n : int, normalize=True):
    """ Receiven an image with the number of components desired (n) and returns the components"""
    if len(img.shape) > 3:
        # We take it for granted that the image is ready for go in the network
        img = np.squeeze(img, axis=0)
        img = np.moveaxis(img, 0, -1)

    reshaped = img.reshape(-1, img.shape[-1])
    pca = PCA(n_components=n)
    result = pca.fit_transform(reshaped)

    result = result.reshape(list(img.shape[:-1]) + [n])
    if normalize: return normalize_image_by_channel(result)
    else: return result

def get_all_s2_mat_images_as_np_array(directory_path : str):
    files = os.listdir(directory_path)
    result = []
    for file in files:
        path = os.path.join(directory_path, file)
        result.append(extract_s2mat(path))
        print(f'-- {file} stacked')
    return np.array(result)

def get_all_png_images_as_np_array(directory_path : str): 
    files = os.listdir(directory_path)
    result = []
    for file in files:
        path = os.path.join(directory_path, file)
        if not os.path.isfile(path): continue
        result.append(cv2.imread(path))
    return np.array(result)

def show_tensor(tensor, title=''):
    tensor = np.squeeze(tensor, 0)
    tensor = np.moveaxis(tensor, 0, -1)
    # Show all channels in the same figure
    fig, ax = plt.subplots(1, tensor.shape[-1])
    fig.suptitle(title)
    for i in range(tensor.shape[-1]):
        ax[i].imshow(tensor[:,:,i], cmap='gray', vmin=0, vmax=1)
    plt.show(block=False)

def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_weights(model):
    for param in model.parameters():
        param.requires_grad = True






        