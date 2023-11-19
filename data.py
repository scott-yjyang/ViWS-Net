from os.path import exists, join, basename
from os import makedirs, remove
#from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, ToTensor

from dataset import MultiDatasetFromFolder, DeRainDatasetFromFolderTest, DeSnowDatasetFromFolderTest, DeHazeDatasetFromFolderTest


def transform():
    return Compose([
        ToTensor(),
    ])


def get_training_set_derain(data_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,
                            future_frame):
    return MultiDatasetFromFolder(data_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset,
                                   patch_size,
                                   future_frame,
                                   transform=transform())

def get_test_set_derain(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame):
    return DeRainDatasetFromFolderTest(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame,
                                       transform=transform())

def get_test_set_desnow(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame):
    return DeSnowDatasetFromFolderTest(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame,
                                       transform=transform())

def get_test_set_dehaze(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame):
    return DeHazeDatasetFromFolderTest(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame,
                                       transform=transform())