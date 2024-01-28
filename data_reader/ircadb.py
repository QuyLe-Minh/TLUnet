from preprocessing.augmentation import *
from preprocessing.voxelization import *

def custom_sort(item):
    return int(item.split('_')[1])

def read(folder_path):
    sample = 0
    dataset = {}
    data_path = os.path.join(folder_path, "data")
    seg_path = os.path.join(folder_path, "seg")
    for patient in os.listdir(data_path):
        data_patient = os.path.join(data_path, patient)
        seg_patient = os.path.join(seg_path, patient)
        
        cube = voxelization(data_patient)
        seg = segmentation(seg_patient, cube)
        
        transformed = augmented(cube, seg)
        for i in range(len(transformed)):
            key_data = f"data_{sample}"
            key_value = f"value_{sample}"

            dataset[key_data] = transformed[i]["image"]
            dataset[key_value] = transformed[i]["label"].to(torch.uint8)
            torch.save(dataset, 'dataset.pth')
            sample += 1 
        