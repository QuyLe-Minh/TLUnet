import torch
from preprocessing.dicom.filter import *
import torch.nn.functional as f
import pydicom as dcm
from math import *
import numpy as np
import os

def custom_sort(item):
    return int(item.split('_')[1])

def segmentation(folder_path, cube):
    files = os.listdir(folder_path)
    files.sort(key = custom_sort)
    _, h, w, d = cube.shape
    seg = torch.empty((1, 1, h, w, len(files)))
    for i, file in enumerate(files):
        path = os.path.join(folder_path, file)
        arr = read_xray(path)
        arr = torch.tensor((arr > 0).astype(np.uint8)).reshape(1,1, 512, 512)
        arr = f.interpolate(arr, size = (h, w))
        seg[:, :, :, :, i] = arr

    seg = f.interpolate(seg, size = (h, w, d))
    seg = seg.squeeze(0).to(torch.uint8)
    return seg

def voxelization(folder_path):
    """Create isometric cube

    Returns:
        cube: shape (D, H, W)
    """
    files = os.listdir(folder_path)
    files.sort(key = custom_sort)
    data = dcm.dcmread(os.path.join(folder_path, files[0]))
    row_space, col_space = data.PixelSpacing
    thickness = data.SliceThickness
    
    anisotropic_diffusion = apply_anisotropic_diffusion()
    
    cube = torch.empty((1, 1, floor(row_space * 512), floor(col_space * 512), floor(thickness * len(files))))
    for i, file in enumerate(files):
        path = os.path.join(folder_path, file)
        data = dcm.dcmread(path)
        arr = data.pixel_array
        arr = window_image(arr)
        
        arr = filter(anisotropic_diffusion, arr)

        arr = arr.reshape(1, 1, 512, 512)
        arr = f.interpolate(arr, scale_factor = (row_space, col_space))

        cube[:, :, :, :, i] = arr

    cube = f.interpolate(cube, scale_factor = (1, 1, thickness))
    cube = cube.squeeze(0)
        
    return cube