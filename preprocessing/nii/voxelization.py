import torch
from preprocessing.nii.filter import *
import torch.nn.functional as f
import nibabel as nib
from math import *

def segmentation(path):
    seg = nib.load(path)
    header = seg.header
    h_seg, w_seg, d_seg = header.get_data_shape()
    scale_x, scale_y, scale_z = header.get_zooms()
    seg = torch.tensor(seg.get_fdata())
    seg = seg.reshape(1, 1, h_seg, w_seg, d_seg)
    seg = f.interpolate(seg, scale_factor = (scale_x, scale_y, scale_z))
    seg = seg.to(torch.uint8).squeeze(0)
    
    return seg


def voxelization(path):
    """Create isometric cube

    Returns:
        cube: shape (1, H, W, D)
    """
    loaded = nib.load(path)
    header = loaded.header
    scale_x, scale_y, scale_z = header.get_zooms()
    h, w, d = header.get_data_shape()
    original_cube = loaded.get_fdata()  #numpy
    
    anisotropic_diffusion = apply_anisotropic_diffusion()
    cube = torch.empty((1, 1, floor(scale_x * h), floor(scale_y * w), d))
    for i in range(d):
        arr = original_cube[:, :, i]
        arr = window_image(arr)
        
        arr = filter(anisotropic_diffusion, arr)

        arr = arr.reshape(1, 1, h, w)
        arr = f.interpolate(arr, scale_factor = (scale_x, scale_y))
        arr = arr.reshape(arr.shape[-2:])

        cube[:, :, :, :, i] = arr
        
    cube = f.interpolate(cube, scale_factor = (1, 1, scale_z))
    cube = cube.squeeze(0)
    return cube
