import torch
from preprocessing.mhd.filter import *
import SimpleITK as sitk
from math import *
from monai.transforms import Resized

def segmentation(path):
    """create segmentation


    Returns:
        seg: 3D label 1,H,W,D
    """
    seg = sitk.ReadImage(path)
    seg = torch.tensor(sitk.GetArrayFromImage(seg))
    seg = seg.permute(1, 2, 0)
    seg = seg.unsqueeze(0)
    
    return seg


def voxelization(path):
    """Create isometric cube

    Returns:
        cube: shape (1, H, W, D)
        seg: shape (1,H,W,D)
    """
    header = sitk.ReadImage(path)
    scale_x, scale_y, scale_z = header.GetSpacing()
    original_cube = torch.tensor(sitk.GetArrayFromImage(header)).permute(1, 2, 0)
    h, w, d = original_cube.shape
     
    
    anisotropic_diffusion = apply_anisotropic_diffusion()
    cube = torch.empty((1, h, w, d))
    for i in range(d):
        arr = original_cube[:, :, i]
        arr = window_image(arr)
        
        arr = filter(anisotropic_diffusion, arr)
        
        cube[:, :, :, i] = arr.unsqueeze(0)
    
   
    return cube, scale_x, scale_y, scale_z

def run(cube_path, seg_path):
    cube, scale_x, scale_y, scale_z = voxelization(cube_path)
    seg = segmentation(seg_path)
    
    _, h, w, d = cube.shape
    
    transform = Resized(["image", "label"], spatial_size=(floor(h * scale_x), floor(w * scale_y), floor(d * scale_z)), mode = "trilinear")

    transformed = transform({"image":cube, "label":seg})   
    return transformed["image"], transformed["label"] 
    
