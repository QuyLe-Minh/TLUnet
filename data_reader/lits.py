from preprocessing.nii.augmentation import *
from preprocessing.nii.voxelization import *

def read_lits(folder_path):
    sample = 900
    dataset = {}
    files = os.listdir(folder_path)
    num_samples = len(files)//2
    for i in range(num_samples):
        data_path = os.path.join(folder_path, f"volume-{i}.nii")
        seg_path = os.path.join(folder_path, f"segmentation-{i}.nii")
        
        cube = voxelization(data_path)
        seg = segmentation(seg_path, cube)
        seg = torch.clamp(seg, min = 0, max = 1)
        
        transformed = augmented(cube, seg)
        for i in range(len(transformed)):
            key_data = f"data_{sample}"
            key_value = f"value_{sample}"

            dataset[key_data] = transformed[i]["image"]
            dataset[key_value] = transformed[i]["label"].to(torch.uint8)
            torch.save(dataset, 'dataset/dataset.pth')
            sample += 1
            
        print(f"LITS: Successful saving patient: {i}. Current sample: {sample}")
    return sample
