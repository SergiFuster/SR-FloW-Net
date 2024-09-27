from src.utils.utils import *
from src.models.models import *
from src.models.codes.losses import *
from pprint import pprint
import  os, random
from time import time

def wavelength_to_rgb(wavelength):
    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    R = G = B = 0
    
    if 380 <= wavelength <= 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 < wavelength <= 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 < wavelength <= 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 < wavelength <= 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 < wavelength <= 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 < wavelength <= 750:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0

    # Ajustar intensidad
    if 380 <= wavelength <= 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 645 <= wavelength <= 750:
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
    else:
        factor = 1.0

    rgb = [int(intensity_max * ((R * factor) ** gamma)),
           int(intensity_max * ((G * factor) ** gamma)),
           int(intensity_max * ((B * factor) ** gamma))]

    return tuple(rgb)

def save_layer_as_png(layer, filename, color):
    # Crear una imagen RGB a partir de la capa
    colored_layer = np.zeros((layer.shape[0], layer.shape[1], 3), dtype=np.uint8)
    
    # Asignar el color especificado a las regiones no nulas de la capa
    for i in range(3):  # Para cada canal R, G, B
        colored_layer[:, :, i] = layer * color[i]
    
    # Guardar la imagen
    img = Image.fromarray(colored_layer)
    img.save(filename)


def extract_mat(path, key=None):
    if not key: key = 'TIF'
    file = h5py.File(path, 'r')
    data = file.get(key)
    data = np.array(data)
    data = data.astype(np.float32)
    data = np.moveaxis(data, 0, -1)
    return data

folders = ['data/images/S2', 'data/images/S3']
files = ['S2_16.mat', 'S3_16.mat']
paths = [os.path.join(folders[i], files[i]) for i in range(2)]

images = list(map(extract_mat, paths))
for idx, image in enumerate(images):

    layers = image
    wavelengths = np.linspace(380, 750, layers.shape[0])
    for i, layer in enumerate(layers):
        color = wavelength_to_rgb(wavelengths[i])
        folder = './results'
        filename = f"{files[idx]}_layer_{i+1}.png"
        img = Image.fromarray(layer)
        img.save(filename)
        print(f"Capa {i+1} guardada como {filename}")