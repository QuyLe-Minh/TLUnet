import torch
from preprocessing.filter import *
import torch.nn.functional as f
import pydicom as dcm
from math import *
import numpy as np
import os

def custom_sort(item):
    return int(item.split('_')[1])

def make_mask(folder_path, cube):
    files = os.listdir(folder_path)
    files.sort(key = custom_sort)
    _, h, w, d = cube.shape
    seg = torch.empty((len(files), h, w))
    for i, file in enumerate(files):
        path = os.path.join(folder_path, file)
        arr = read_xray(path)
        arr = torch.tensor((arr > 0).astype(np.uint8)).reshape(1,1, 512, 512)
        arr = f.interpolate(arr, scale_factor = (h/512, w/512), mode = "nearest")
        seg[i] = arr.reshape(h, w)

    return seg

def segmentation(folder_path, cube):
    """Create 3D segmentation

    Args:
        folder_path;
        cube (3D tensor): Volume

    Returns:
        3D segmentation
    """
    seg = make_mask(folder_path, cube)
    _, h, w, d = cube.shape
    seg = seg.unsqueeze(1)
    seg = seg.permute(2, 3, 0, 1)

    #make grid
    x = torch.linspace(-1, 1, d)
    y = torch.linspace(-1, 1, 1)
    mx, my = torch.meshgrid((y, x))
    grid = torch.stack((mx, my), 2).permute(1, 0, 2).unsqueeze(0)
    grid =  grid.repeat(h, 1, 1, 1)

    seg = f.grid_sample(seg, grid, mode = "nearest")
    seg = seg.permute(3, 0, 1, 2)
    seg = seg.to(torch.uint8)
    return seg

def make_cube(folder_path):
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
    
    cube = torch.empty((len(files), floor(row_space * 512), floor(col_space * 512)))
    for i, file in enumerate(files):
        path = os.path.join(folder_path, file)
        data = dcm.dcmread(path)
        arr = data.pixel_array
        arr = window_image(arr)
        
        arr = filter(anisotropic_diffusion, arr)

        arr = arr.reshape(1, 1, 512, 512)
        arr = f.interpolate(arr, scale_factor = (row_space, col_space))
        arr = arr.reshape(arr.shape[-2:])

        cube[i] = torch.tensor(arr)
        
    return cube, thickness

def voxelization(folder_path):
    """Voxelization

    Returns:
        cube: shape (1, H, W, D)
    """
    #make cube
    cube, thickness = make_cube(folder_path)
    d, h, w = cube.shape
    cube = cube.unsqueeze(1)
    cube = cube.permute(2, 3, 0, 1)

    #make grid
    x = torch.linspace(-1, 1, round(d*thickness))
    y = torch.linspace(-1, 1, 1)
    mx, my = torch.meshgrid((y, x))
    grid = torch.stack((mx, my), 2).permute(1, 0, 2).unsqueeze(0)
    grid =  grid.repeat(h, 1, 1, 1)

    cube = f.grid_sample(cube, grid)
    
    cube = cube.permute(3, 0, 1, 2)    #1,h,w,d
    return cube