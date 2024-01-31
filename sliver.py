import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def read_mhd(file_path):
    image = sitk.ReadImage(file_path)
    return image

# Replace 'your_file.mhd' with the path to your MHD file
file_path = "liver-seg020.mhd"
image = read_mhd(file_path)

# Extract pixel spacing
pixel_spacing = image.GetSpacing()

# Print the pixel spacing
print("Pixel Spacing:", pixel_spacing)
arr = sitk.GetArrayFromImage(image)
print(arr.shape)
print(np.unique(arr[120]))
plt.imshow(arr[120] * 100, cmap = "gray")
plt.show()
