import torch
import nibabel as nib
from math import *
from monai.transforms import Resize
from metrics.metrics import *

def inference(gt, val):
    """
    gt, val: path
    """
    data = nib.load(gt)
    header = data.header
    h, w, d = header.get_data_shape()
    gt = torch.tensor(data.get_fdata())

    seg = torch.load(val)

    transform = Resize((h, w, d), mode="nearest")
    seg = transform(seg)
    return gt, seg

validation = [1, 2, 3, 4, 8, 22, 25, 29, 32, 35, 36, 38]
s = 0
for i, val in enumerate(validation):
    code = str(val)
    if val < 10:
        code = "0" + val
    gt, pred = inference(f"labelstr/label00{code}.nii.gz", f"results/val_{i}.pth")
    d = dice(pred, gt)
    s += d
    with open("result.txt", "w") as file:
        file.write(f"{d:>2f} \t")

s /= len(val)
with open("result.txt", "w") as file:
    file.write(f"mean: {s:>2f} \t")
