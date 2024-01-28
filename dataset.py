import pydicom 
import numpy as np
import os
import re
import shutil
import cv2 as cv
from PIL import Image

origin_data = "C:\\Users\\Admin\\OneDrive - hcmut.edu.vn\\Giáo trình\\Digital Image Processing & Computer Vision\\Ass 1\\3Dircadb1"
data = "C:\\Users\\Admin\\OneDrive - hcmut.edu.vn\\Giáo trình\\Digital Image Processing & Computer Vision\\Ass 3\\ircadb\\data"
seg = "C:\\Users\\Admin\\OneDrive - hcmut.edu.vn\\Giáo trình\\Digital Image Processing & Computer Vision\\Ass 3\\ircadb\\seg"

os.makedirs(data, exist_ok=True)
os.makedirs(seg, exist_ok=True)
  
def make_mask():
    for folder in os.listdir(origin_data):

        path = os.path.join(origin_data, folder, "MASKS_DICOM\\liver")
        dest = os.path.join(seg, folder)
        os.makedirs(dest, exist_ok=True)

        for dir1, _, files in os.walk(path):
            for file in files:
                dir = os.path.join(path, file)
                print(dir)
                shutil.copy2(dir, dest)

def make_training():
  for folder in os.listdir(origin_data):
    path = os.path.join(origin_data, folder)
    dest = os.path.join(data, folder)
    os.makedirs(dest, exist_ok=True)
    
    for sub_folder in os.listdir(path):
      if sub_folder == "PATIENT_DICOM":
        dir = os.path.join(path, sub_folder)
        for file in os.listdir(dir):
          shutil.copy2(os.path.join(dir, file), dest)
        break
       
def make_dataset():
  make_mask()
  make_training()
  
make_dataset()
print("-----------------------------SUCCESS----------------------")