import torch
from preprocessing.dicom.filter import *
import pydicom as dcm
from math import *
import numpy as np
import os
from monai.transforms import Resized

def custom_sort(item):
    return int(item.split('_')[1])

def segmentation(folder_path):
    """Create non-uniform cube

    Returns:
        seg: torch tensor shape (1,D,H,W)
    """
    files = os.listdir(folder_path)
    files.sort(key = custom_sort)
    seg = torch.empty((1, len(files), 512, 512))
    for i, file in enumerate(files):
        path = os.path.join(folder_path, file)
        arr = read_xray(path)
        arr = torch.from_numpy((arr > 0).astype(np.uint8))
        seg[0, i, :, :] = arr

    return seg

def voxelization(folder_path):
    """Create non-uniform cube

    Returns:
        cube: torch tensor shape (1,D,H,W)
    """
    files = os.listdir(folder_path)
    files.sort(key = custom_sort)
    data = dcm.dcmread(os.path.join(folder_path, files[0]))
    row_space, col_space = data.PixelSpacing
    thickness = data.SliceThickness
    
    cube = torch.empty((1, len(files), 512, 512))
    for i, file in enumerate(files):
        path = os.path.join(folder_path, file)
        data = dcm.dcmread(path)
        arr = data.pixel_array

        cube[0, i, :, :] = torch.from_numpy(arr)
        
    return cube, row_space, col_space, thickness

def run(cube_path, seg_path):
    cube, scale_x, scale_y, scale_z = voxelization(cube_path)
    seg = segmentation(seg_path)
    
    _, d, h, w = cube.shape
    
    transform = Resized(["image", "label"], spatial_size=(floor(d * scale_z), floor(h * scale_x), floor(w * scale_y)), mode = "trilinear")
    transformed = transform({"image":cube, "label":seg})   
    cube, seg = transformed["image"], transformed["label"]

    _, d, h, w = cube.shape

    anisotropic_diffusion = apply_anisotropic_diffusion()

    cube = window_image(cube[0])
    cube = filter(anisotropic_diffusion, cube)
    cube = cube.unsqueeze(0)
    
    return cube, seg 
    
    