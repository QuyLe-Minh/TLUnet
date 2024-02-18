import torch
from preprocessing.nii.filter import *
import nibabel as nib
from math import *
from monai.transforms import Compose, Rotate90d, Resized

def segmentation(path):
    """create segmentation


    Returns:
        seg: 3D label 1,D,H,W
    """
    seg = nib.load(path)
    seg = torch.tensor(seg.get_fdata())
    seg = seg.permute(2, 0, 1)
    seg = seg.unsqueeze(0)
    
    return seg


def voxelization(path):
    """Create isometric cube

    Returns:
        cube: shape (1, D, H, W)
        seg: shape (1, D, H, W)
    """
    loaded = nib.load(path)
    header = loaded.header
    scale_x, scale_y, scale_z = header.get_zooms()
    original_cube = torch.from_numpy(loaded.get_fdata())  #numpy
    original_cube = original_cube.permute(2, 0, 1)

    return original_cube.unsqueeze(0), scale_x, scale_y, scale_z


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
    
