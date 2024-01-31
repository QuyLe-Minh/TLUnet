from preprocessing.nii.augmentation import *
from preprocessing.nii.voxelization import *
import os

def read_lits(folder_path):
    sample_dataset = 850
    sample_val = 3
    files = os.listdir(folder_path)
    num_samples = len(files)//2
    for i in range(num_samples):
        if i in [23, 26, 53, 56, 62, 64, 100, 108, 120, 121, 126, 127, 128]:
            mode = "val.pth"
        else:
            mode = "dataset.pth"
        data_path = os.path.join(folder_path, f"volume-{i}.nii")
        seg_path = os.path.join(folder_path, f"segmentation-{i}.nii")
        
        cube, seg = run(data_path, seg_path)

        if mode == "dataset.pth":
            transformed = augmented(cube, seg)
            for j in range(len(transformed)):
                dataset = {}
    
                dataset["data"] = transformed[j]["image"]
                dataset["value"] = torch.ceil(transformed[j]["label"]).to(torch.uint8)
                dataset["value"] = torch.clamp(dataset["value"], min = 0, max = 1)
                torch.save(dataset, f"dataset/train/train_{sample_dataset}.pth")
                sample_dataset += 1
        else:
            val = {}
            val["data"] = cube
            val["value"] = torch.ceil(seg).to(torch.uint8)
            val["value"] = torch.clamp(val["value"], min = 0, max = 1)
            torch.save(val, f"dataset/val/val_{sample_val}.pth")
            sample_val +=1
            
        print(f"LITS: Successful saving patient: {i}. Current sample: {sample_dataset}. Current val: {sample_val}")
