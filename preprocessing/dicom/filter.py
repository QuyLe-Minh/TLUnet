import SimpleITK as sitk
import torch
import pydicom as dcm
import numpy as np

def standardize(dcm):
    px_arr = dcm.pixel_array
    if dcm.PixelRepresentation == 1: #signed = 1, unsigned = 0
        bit_shift = dcm.BitsAllocated - dcm.BitsStored #handled signed pixel data
        dtype = px_arr.dtype
        new_arr = (px_arr << bit_shift).astype(dtype) >> bit_shift #??
        px_arr = dcm.pixel_data_handlers.util.apply_modality_lut(new_arr, dcm)
    return px_arr

def read_xray(path, fix_monochrome = True):
  try:
    dicom = dcm.dcmread(path)
    data = standardize(dicom)
    data = data - np.min(data)
    data = data / (np.max(data) + 1e-5)
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = 1.0 - data
    return data

  except dcm.errors.InvalidDicomError:
    print("Failed")
    return

def window_image(img, window_center = -300, window_width = 1400, intercept = 0, slope = 1, rescale=True):
    img = (img*slope +intercept) 
    img_min = window_center - window_width//2 
    img_max = window_center + window_width//2
    img[img<img_min] = img_min 
    img[img>img_max] = img_max 
    img = img - img_min
    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0 
    return img
    
def apply_anisotropic_diffusion(num_iterations=20, time_step=0.125, conductance=1.5):
    anisotropic_diffusion_filter = sitk.GradientAnisotropicDiffusionImageFilter()
    anisotropic_diffusion_filter.SetNumberOfIterations(num_iterations)
    anisotropic_diffusion_filter.SetTimeStep(time_step)
    anisotropic_diffusion_filter.SetConductanceParameter(conductance)
    return anisotropic_diffusion_filter

def filter(anisotropic_diffusion_filter, arr):
    image = sitk.GetImageFromArray(arr)
    diffused_image = anisotropic_diffusion_filter.Execute(image)
    diffused_arr = sitk.GetArrayFromImage(diffused_image)
    return torch.tensor(diffused_arr)
