from preprocessing.dicom.augmentation import *
from preprocessing.dicom.voxelization import *

def read_ircadb(folder_path, sample_dataset, sample_val):

    data_path = os.path.join(folder_path, "data")
    seg_path = os.path.join(folder_path, "seg")
    for patient in os.listdir(data_path):
        if patient in ["3Dircadb1.2", "3Dircadb1.6"]:
            mode = "val"
        else:
            mode = "dataset"
            
        data_patient = os.path.join(data_path, patient)
        seg_patient = os.path.join(seg_path, patient)
        
        cube, seg = run(data_patient, seg_patient)

        if mode == "dataset":
            transformed = augmented(cube, seg)
            for i in range(len(transformed)):
                dataset = {}
    
                dataset["data"] = transformed[i]["image"]
                dataset["value"] = torch.clamp(transformed[i]["label"], min = 0, max = 1)
                torch.save(dataset, f"dataset/train_tlu/train_{sample_dataset}.pth")
                sample_dataset += 1 
        else:
            val = {}
            val["data"] = cube
            val["value"] = torch.clamp(seg, min = 0, max = 1)
            torch.save(val, f"dataset/val_tlu/val_{sample_val}.pth")
            sample_val += 1

        print(f"IRCADB: Successful saving patient: {patient}. Current sample: {sample_dataset}. Current val: {sample_val}")
    return sample_dataset, sample_val
        
        