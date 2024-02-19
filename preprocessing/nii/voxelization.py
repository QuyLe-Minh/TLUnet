import torch
from preprocessing.nii.filter import *
import nibabel as nib
from math import *
from monai.transforms import Resized

def segmentation(path):
    """create segmentation


    Returns:
        seg: 3D label 1,H,W,D
    """
    seg = nib.load(path)
    seg = torch.tensor(seg.get_fdata())
    seg = seg.unsqueeze(0)
    
    return seg


def voxelization(path):
    """Create isometric cube

    Returns:
        cube: shape (1, H, W, D)
        seg: shape (1,H,W,D)
    """
    loaded = nib.load(path)
    header = loaded.header
    scale_x, scale_y, scale_z = header.get_zooms()
    h, w, d = header.get_data_shape()
    original_cube = loaded.get_fdata()  #numpy

    return torch.tensor(original_cube).unsqueeze(0), scale_x, scale_y, scale_z


def run(cube_path, seg_path):
    cube, scale_x, scale_y, scale_z = voxelization(cube_path)
    seg = segmentation(seg_path)
    
    _, h, w, d = cube.shape
    
    transform = Resized(["image", "label"], spatial_size=(floor(h * scale_x), floor(w * scale_y), floor(d * scale_z)), mode = ["trilinear", "nearest"])
    transformed = transform({"image":cube, "label":seg})   
    cube, seg = transformed["image"], transformed["label"]

    _, h, w, d = cube.shape

    anisotropic_diffusion = apply_anisotropic_diffusion()

    cube = window_image(cube[0])
    cube = filter(anisotropic_diffusion, cube)
    cube = cube.unsqueeze(0)
    
    return cube, seg 
    