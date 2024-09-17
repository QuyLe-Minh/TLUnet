import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/Admin/Downloads/img0002.npz"
data = np.load(path)
for array_name in data.files:
    print(array_name)
    img, label = data[array_name]
    print(img.shape)
    plt.imshow(img[50], cmap='gray')
    plt.show()
    
