from preprocessing.nii.augmentation import *
from preprocessing.nii.voxelization import *

def read_lits(folder_path):
    sample_dataset = 850
    sample_val = 3
    dataset = {}
    val = {}
    files = os.listdir(folder_path)
    num_samples = len(files)//2
    for i in range(num_samples):
        if i <= 50:
            mode = "val.pth"
        else:
            mode = "dataset.pth"
        data_path = os.path.join(folder_path, f"volume-{i}.nii")
        seg_path = os.path.join(folder_path, f"segmentation-{i}.nii")
        
        cube = voxelization(data_path)
        seg = segmentation(seg_path, cube)
        seg = torch.clamp(seg, min = 0, max = 1)

        if mode == "dataset.pth":
            transformed = augmented(cube, seg)
            for j in range(len(transformed)):
                key_data = f"data_{sample_dataset}"
                key_value = f"value_{sample_dataset}"
    
                dataset[key_data] = transformed[j]["image"]
                dataset[key_value] = transformed[j]["label"].to(torch.uint8)
                torch.save(dataset, 'dataset/' + mode)
                sample_dataset += 1
        else:
            key_data = f"data_{sample_val}"
            key_val = f"value_{sample_val}"
            val[key_data] = cube
            val[key_val] = seg.to(torch.uint8)
            torch.save(val, "dataset/" + mode)
            sample_val +=1
            
        print(f"LITS: Successful saving patient: {i}. Current sample: {sample_dataset}. Current val: {sample_val}")
    return sample_dataset, sample_val
