from preprocessing.mhd.augmentation import *
from preprocessing.mhd.voxelization import *
import os

def read_sliver(folder_path, sample_dataset, sample_val):
    for i in range(1,21):
        if i in [2, 20]:
            mode = "val"
        else:
            mode = "dataset"
        
        code = str(i)
        if i < 10:
            code = "0" + code
        data_path = os.path.join(folder_path, "scan", f"liver-orig0{code}.mhd")
        seg_path = os.path.join(folder_path, "label", f"liver-seg0{code}.mhd")
        
        cube, seg = run(data_path, seg_path)

        if mode == "dataset":
            transformed = augmented(cube, seg)
            for j in range(len(transformed)):
                dataset = {}
    
                dataset["data"] = transformed[j]["image"]
                dataset["value"] = torch.clamp(transformed[j]["label"], min = 0, max = 1)
                torch.save(dataset, f"dataset/train_tlu/train_{sample_dataset}.pth")
                sample_dataset += 1
        else:
            val = {}
            val["data"] = cube
            val["value"] = torch.clamp(seg, min = 0, max = 1)
            torch.save(val, f"dataset/val_tlu/val_{sample_val}.pth")
            sample_val +=1
            
        print(f"SLIVER07: Successful saving patient: {i}. Current sample: {sample_dataset}. Current val: {sample_val}")
    return sample_dataset, sample_val
