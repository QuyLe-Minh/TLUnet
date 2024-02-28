from monai.transforms import *
from preprocessing.dicom.voxelization import *

def augmented(image, label):
    transform = Compose([
        RandRotated(keys=["image", "label"], range_x=np.pi/18, range_y=np.pi/18, range_z=np.pi/4, prob=0.3),
        Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 7), magnitude_range=(50, 150), prob=0.3),
        RandSpatialCropSamplesd(keys=["image", "label"], num_samples=120, roi_size=(192, 192, 64)),
        ToTensor()
    ])
    transformed = transform({"image": image, "label":label})
    return transformed