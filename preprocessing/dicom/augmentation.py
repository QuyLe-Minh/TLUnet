from monai.transforms import *
from preprocessing.dicom.voxelization import *

def augmented(image, label):
    transform = Compose([
        RandRotated(keys=["image", "label"], range_x=10, range_y=10, range_z=45, prob=0.5),
        Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 7), magnitude_range=(50, 150), prob=0.5),
        RandSpatialCropSamplesd(keys=["image", "label"], num_samples=60, roi_size=(64, 192, 192)),
        ToTensor()
    ])
    transformed = transform({"image": image, "label":label})
    return transformed