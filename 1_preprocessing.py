import cv2
import os
import glob
import numpy as np
import json

# KONFIGURASI FOLDER
input_dir = 'input'
output_dir = 'output'

os.makedirs(output_dir, exist_ok=True)

# GENERATE KERNEL GAUSSIAN
def get_cv2_gaussian_kernel(ksize, sigma=0.0):
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    
    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    kernel_1d = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel_1d /= np.sum(kernel_1d) # Normalisasi agar total nilai = 1
    return kernel_1d

# KONVOLUSI GAUSSIAN TERPISAH (SEPARABLE CONVOLUTION)
def gaussian_blur_2d(img, ksize, sigma=0.0):
    kernel_1d = get_cv2_gaussian_kernel(ksize, sigma)
    pad = ksize // 2
    
    # NumPy 'reflect' mode untuk padding di tepi gambar
    padded = np.pad(img, pad, mode='reflect')
    h, w = img.shape
    
    # Tahap 1: Konvolusi Horizontal
    h_blurred = np.zeros((h + 2 * pad, w), dtype=np.float64)
    for i in range(ksize):
        h_blurred += padded[:, i : i + w] * kernel_1d[i]
        
    # Tahap 2: Konvolusi Vertikal
    v_blurred = np.zeros((h, w), dtype=np.float64)
    for i in range(ksize):
        v_blurred += h_blurred[i : i + h, :] * kernel_1d[i]
        
    return v_blurred

# ADAPTIVE THRESHOLDING (GAUSSIAN C)
def manual_adaptive_threshold(blur_img, block_size=31, c=1):
    # Hitung nilai rata-rata lokal menggunakan kernel Gaussian berskala 31x31
    local_mean = gaussian_blur_2d(blur_img, ksize=block_size, sigma=0.0)
    
    # Thresholding biner: jika piksel > (rata-rata lokal - C), nilai = 255, else 0
    thresh = np.where(blur_img > (local_mean - c), 255, 0)
    return thresh.astype(np.uint8)

# GENERAL 2D CONVOLUTION (Untuk Sharpening)
def convolve_2d(img, kernel):
    ksize = kernel.shape[0]
    pad = ksize // 2
    padded = np.pad(img, pad, mode='reflect')
    
    h, w = img.shape
    output = np.zeros((h, w), dtype=np.float64)
    
    # Geser kernel dan kalikan secara vektorisasi
    for ky in range(ksize):
        for kx in range(ksize):
            output += padded[ky : ky + h, kx : kx + w] * kernel[ky, kx]
            
    # Clamp pixel ke rentang [0, 255]
    return np.clip(output, 0, 255).astype(np.uint8)

# SIMPAN JSON
def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

# =========================================
# MAIN PIPELINE EXECUTION
# =========================================
for f in glob.glob(f'{input_dir}/*.*'):

    print(f'Processing {f}...')

    # Langsung load sebagai grayscale (1 channel matriks 2D)
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f'Gagal membaca file: {f}')
        continue

    # Tahap 1: Gaussian Blur (Kernel 5x5, sigma otomatis)
    blur = gaussian_blur_2d(img, ksize=5, sigma=0.0)

    # Tahap 2: Adaptive Threshold (Block Size=31, C=1)
    thresh = manual_adaptive_threshold(blur, block_size=31, c=1)

    # Tahap 3: Invert
    inv = 255 - thresh

    # Tahap 4: Sharpening (Custom Kernel 3x3)
    sharpen_kernel = np.array([[-1, -1, -1], 
                               [-1,  9, -1], 
                               [-1, -1, -1]], dtype=np.float64)
    sharp = convolve_2d(inv, sharpen_kernel)

    # Tahap 5: Encoding ke biner 0 dan 1 (Format list untuk JSON)
    encoded = np.where(sharp >= 128, 1, 0).tolist()

    # Simpan hasil matriks ekstraksi fitur ke JSON
    base_name = os.path.splitext(os.path.basename(f))[0]
    json_path = os.path.join(output_dir, f'{base_name}.json')
    save_json(encoded, json_path)

print('Seluruh citra tanda tangan berhasil diproses!')