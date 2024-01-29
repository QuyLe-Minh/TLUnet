from monai.transforms import *
from preprocessing.nii.voxelization import *

def augmented(image, label):
    transform = Compose([
        Rotate90d(keys = ["image", "label"], k = 1),
        RandRotated(keys=["image", "label"], range_x=10, range_y=10, range_z=45, prob=0.3),
        Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 7), magnitude_range=(50, 150), prob=0.8),
        RandSpatialCropSamplesd(keys=["image", "label"], num_samples=50, roi_size=(192, 192, 64)),
        ToTensor(),
    ])
    transformed = transform({"image": image, "label":label})
    return transformed