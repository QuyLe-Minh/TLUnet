from preprocessing.mhd.augmentation import *
from preprocessing.mhd.voxelization import *
import os

def read_sliver(folder_path, sample_dataset, sample_val):
    for i in range(20):
        if i in [1, 2]:
            mode = "val.pth"
        else:
            mode = "dataset.pth"
        
        code = str(i)
        if i < 10:
            code = "0" + code
        data_path = os.path.join(folder_path, "scan", f"liver-orig0{code}.mhd")
        seg_path = os.path.join(folder_path, "label", f"liver-seg0{code}.mhd")
        
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
    return sample_dataset, sample_val
