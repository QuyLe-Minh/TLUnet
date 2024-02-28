import SimpleITK as sitk
import torch

def window_image(img, window_center = -300, window_width = 1400, intercept = 0, slope = 1, rescale=True):
    img = (img*slope +intercept) 
    img_min = window_center - window_width//2 
    img_max = window_center + window_width//2
    img[img<img_min] = img_min 
    img[img>img_max] = img_max 
    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0 
    return img
    
def apply_anisotropic_diffusion(num_iterations=10, time_step=0.0625, conductance=1.5):
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
