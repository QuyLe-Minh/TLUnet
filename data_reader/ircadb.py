from preprocessing.dicom.augmentation import *
from preprocessing.dicom.voxelization import *
# import numpy as np
# import os
# import shutil

### Uncomment to extract liver from the original dataset

# origin_data = "C:\\Users\\Admin\\OneDrive - hcmut.edu.vn\\Giáo trình\\Digital Image Processing & Computer Vision\\Ass 1\\3Dircadb1"
# data = "C:\\Users\\Admin\\OneDrive - hcmut.edu.vn\\Giáo trình\\Digital Image Processing & Computer Vision\\Ass 3\\ircadb\\data"
# seg = "C:\\Users\\Admin\\OneDrive - hcmut.edu.vn\\Giáo trình\\Digital Image Processing & Computer Vision\\Ass 3\\ircadb\\seg"

# os.makedirs(data, exist_ok=True)
# os.makedirs(seg, exist_ok=True)
  
# def make_mask():
#     for folder in os.listdir(origin_data):

#         path = os.path.join(origin_data, folder, "MASKS_DICOM\\liver")
#         dest = os.path.join(seg, folder)
#         os.makedirs(dest, exist_ok=True)

#         for dir1, _, files in os.walk(path):
#             for file in files:
#                 dir = os.path.join(path, file)
#                 print(dir)
#                 shutil.copy2(dir, dest)

# def make_training():
#   for folder in os.listdir(origin_data):
#     path = os.path.join(origin_data, folder)
#     dest = os.path.join(data, folder)
#     os.makedirs(dest, exist_ok=True)
    
#     for sub_folder in os.listdir(path):
#       if sub_folder == "PATIENT_DICOM":
#         dir = os.path.join(path, sub_folder)
#         for file in os.listdir(dir):
#           shutil.copy2(os.path.join(dir, file), dest)
#         break
       
# def make_dataset():
#   make_mask()
#   make_training()
  
# make_dataset()
# print("-----------------------------SUCCESS----------------------")

def read_ircadb(folder_path, sample_dataset, sample_val):

    data_path = os.path.join(folder_path, "data")
    seg_path = os.path.join(folder_path, "seg")
    for patient in os.listdir(data_path):
        if patient in ["3Dircadb1.1", "3Dircadb1.8"]:
            mode = "val.pth"
        else:
            mode = "dataset.pth"
            
        data_patient = os.path.join(data_path, patient)
        seg_patient = os.path.join(seg_path, patient)
        
        cube, seg = run(data_patient, seg_patient)

        if mode == "dataset.pth":
            transformed = augmented(cube, seg)
            for i in range(len(transformed)):
                dataset = {}
    
                dataset["data"] = transformed[i]["image"]
                dataset["value"] = torch.ceil(transformed[i]["label"]).to(torch.uint8)
                dataset["value"] = torch.clamp(dataset["value"], min = 0, max = 1)
                torch.save(dataset, f"dataset/train/train_{sample_dataset}.pth")
                sample_dataset += 1 
        else:
            val = {}
            val["data"] = cube
            val["value"] = torch.ceil(seg).to(torch.uint8)
            val["value"] = torch.clamp(val["value"], min = 0, max = 1)
            torch.save(val, f"dataset/val/val_{sample_val}.pth")
            sample_val += 1

        print(f"Successful saving patient: {patient}. Current sample: {sample_dataset}. Current val: {sample_val}")
    return sample_dataset, sample_val
        
        