import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import correlate
from scipy.signal import convolve2d
from math import log10

def rmse(img1 : np.ndarray, img2 : np.ndarray):
    """Computates RMSE between img1 and img2"""
    diff = np.subtract(img1, img2, dtype=np.float64)
    rmse = np.sqrt(np.mean(np.square(diff)))
    return rmse

def nrmse(img1 : np.ndarray, img2 : np.ndarray):
    """Uses rmse function to calculate nrmse, we assume that imgs in range [0, 255]"""
    return rmse(img1, img2) / 255

def psnr(img1 : np.ndarray, img2 : np.ndarray):

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / np.sqrt(mse))
    return psnr

def ssim_index(img1 : np.ndarray, img2 : np.ndarray, multichannel=False, data_range=1):
    if multichannel: index = ssim(img1, img2, channel_axis=2, data_range=data_range)
    else: index = ssim(img1, img2, data_range=data_range)
    return index


def sam(image1, image2):
    """
    Calcula la métrica SAM entre dos imágenes hiperespectrales.
    
    :param image1: Primera imagen hiperespectral, de dimensiones (M, N, L)
    :param image2: Segunda imagen hiperespectral, de dimensiones (M, N, L)
    :return: Matriz de ángulos espectrales, de dimensiones (M, N)
    """
    if image1.shape != image2.shape:
        raise ValueError("Las imágenes deben tener las mismas dimensiones")
    
    # Convertir las imágenes a matrices de vectores espectrales (M*N, L)
    vec1 = image1.reshape(-1, image1.shape[-1])
    vec2 = image2.reshape(-1, image2.shape[-1])
    
    # Calcular productos punto
    dot_product = np.sum(vec1 * vec2, axis=1)
    
    # Calcular normas
    norm1 = np.linalg.norm(vec1, axis=1)
    norm2 = np.linalg.norm(vec2, axis=1)
    
    # Calcular ángulos espectrales
    cos_theta = dot_product / (norm1 * norm2)
    
    # Asegurarse de que los valores estén en el rango [-1, 1] para evitar errores numéricos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calcular el ángulo en radianes
    theta = np.arccos(cos_theta)
    
    # Calcular la media de los ángulos espectrales
    mean_sam = np.mean(theta)

    
    return mean_sam

def correlation_coefficient(img1 : np.ndarray, img2 : np.ndarray):
    mean1, mean2 = img1.mean(), img2.mean()
    std1, std2 = img1.std(), img2.std()
    n = img1.size
    if n < 2:
        return 0
    cov = correlate(img1, img2, mode='constant')
    c = cov.mean()
    return (c - mean1 * mean2) / (std1 * std2)

def q_index(img1 : np.ndarray, img2 : np.ndarray):
    n = img1.size
    if n < 2:
        return 0
    covariance = np.cov(img1.flatten(), img2.flatten())[0, 1]
    R1, R2 = img1.max() - img1.min(), img2.max() - img2.min()
    q = 4 * covariance * img1.mean() * img2.mean()
    q /= (R1 ** 2 + R2 ** 2) * (img1.std() ** 2 + img2.std() ** 2)
    return q

def mi(img1 : np.ndarray, img2 : np.ndarray):
    histogram_2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=256)
    pxy = histogram_2d / float(histogram_2d.sum())
    px = np.histogram(img1, bins=256)[0] / float(img1.size)
    py = np.histogram(img2, bins=256)[0] / float(img2.size)
    px_py = px[:, np.newaxis] * py[np.newaxis, :]
    nonzero_indices = pxy > 0
    return np.sum(pxy[nonzero_indices] * np.log(pxy[nonzero_indices] / px_py[nonzero_indices]))

def ergas(img1 : np.ndarray, img2 : np.ndarray, scale_factor=1):
    m = img1.shape[0]
    n = img1.shape[1]
    mse = np.mean((img1 - img2) ** 2)
    rmse = np.sqrt(mse)
    ergas = 100 * (rmse / img2.mean()) / scale_factor
    return ergas
    
def get_metrics():
    return[
        rmse,
        nrmse,
        psnr,
        ssim_index,
        ergas,
        q_index,
        mi
    ]

def get_labels():
    return [
        "RMSE",
        "NRMSE",
        "PSNR",
        "SSIM",
        "ERGAS",
        "Q",
        "MI"
    ]

def get_titles():
    return [
        "RMSE (Error Cuadrático Medio de la Raíz)",
        "NRMSE (Error Cuadrático Medio Normalizado)",
        "PSNR (Relación Señal a Ruido de Pico)",
        "SSIM (Índice de Similitud Estructural)",
        "ERGAS (Error Relativo Global de la Abundancia Espectral)",
        "Q (Índice de calidad)",
        "MI (Información mutua)"
    ]

def get_labeled_metrics():
    return zip(get_labels(), get_metrics())

def get_titled_metrics():
    return zip(get_titles(), get_metrics())

