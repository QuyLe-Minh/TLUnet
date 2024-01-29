import torch
from preprocessing.nii.filter import *
import torch.nn.functional as f
import nibabel as nib
from math import *
import numpy as np
import os

def make_mask(path, cube):
    segmentation = nib.load(path)
    header = segmentation.header
    h_seg, w_seg, d_seg = header.get_data_shape()
    segmentation = segmentation.get_fdata()
    _, h, w, d = cube.shape
    seg = torch.empty((d_seg, h, w))
    for i in range(d_seg):
        arr = torch.tensor(segmentation[:, :, i]).reshape(1, 1, 512, 512)
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

def make_cube(path):
    """Create isometric cube

    Returns:
        cube: shape (D, H, W)
    """
    loaded = nib.load(path)
    header = loaded.header
    row_space, col_space, thickness = header.get_zooms()
    h, w, d = header.get_data_shape()
    original_cube = loaded.get_fdata()  #numpy
    
    anisotropic_diffusion = apply_anisotropic_diffusion()
    
    cube = torch.empty((d, floor(row_space * h), floor(col_space * w)))
    for i in range(d):
        arr = original_cube[:, :, i]
        arr = window_image(arr)
        
        arr = filter(anisotropic_diffusion, arr)

        arr = arr.reshape(1, 1, h, w)
        arr = f.interpolate(arr, scale_factor = (row_space, col_space))
        arr = arr.reshape(arr.shape[-2:])

        cube[i] = torch.tensor(arr)
        
    return cube, thickness

def voxelization(path):
    """Voxelization

    Returns:
        cube: shape (1, H, W, D)
    """
    #make cube
    cube, thickness = make_cube(path)
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