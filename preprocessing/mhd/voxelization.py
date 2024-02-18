import torch
from preprocessing.mhd.filter import *
import SimpleITK as sitk
from math import *
from monai.transforms import Resized

def segmentation(path):
    """create segmentation


    Returns:
        seg: 3D label 1,D,H,W
    """
    seg = sitk.ReadImage(path)
    seg = torch.tensor(sitk.GetArrayFromImage(seg))
    seg = seg.unsqueeze(0)
    
    return seg


def voxelization(path):
    header = sitk.ReadImage(path)
    scale_x, scale_y, scale_z = header.GetSpacing()
    original_cube = torch.tensor(sitk.GetArrayFromImage(header))
    
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
    
