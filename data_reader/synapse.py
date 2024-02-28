from preprocessing.synapse.augmentation import *
from preprocessing.synapse.voxelization import *
import os

def read_synapse(folder_path, sample_dataset, sample_val):
    images = os.path.join(folder_path, "imagesTr")
    labels = os.path.join(folder_path, "labelsTr")
    train = [range(1, 11), range(21, 41)]
    train = [n for i in train for n in i]
    for sample in train:
        if sample in [1, 2, 3, 4, 8, 22, 25, 29, 32, 35, 36, 38]:
            mode = "val"
        else:
            mode = "dataset"
        
        code = str(sample)
        if sample < 10:
            code = "0" + code

        data_path = os.path.join(images, f"img00{code}.nii.gz")
        seg_path = os.path.join(labels, f"label00{code}.nii.gz")
        
        cube, seg = run(data_path, seg_path)

        if mode == "dataset":
            transformed = augmented(cube, seg)
            for j in range(len(transformed)):
                dataset = {}
    
                dataset["data"] = transformed[j]["image"]
                dataset["value"] = torch.clamp(transformed[j]["label"], min = 0, max = 1)
                torch.save(dataset, f"dataset/train_synapse/train_{sample_dataset}.pth")
                sample_dataset += 1
        else:
            val = {}
            val["data"] = cube
            val["value"] = torch.clamp(seg, min = 0, max = 1)
            torch.save(val, f"dataset/val_synapse/val_{sample_val}.pth")
            sample_val +=1
            
        print(f"LITS: Successful saving patient: {sample}. Current sample: {sample_dataset}. Current val: {sample_val}")
    return sample_dataset, sample_val
