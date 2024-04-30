from dataclasses import dataclass, field
import numpy as np
import cv2
import glob
from typing import List

@dataclass 
class PredictionOutput:
    input_prompt: dict
    mask: np.ndarray = None
    masked_image: np.ndarray = None
    score: float = 0.0
    area_ratio: float = 0.0

@dataclass
class ImageData:
    # Loading
    image: np.ndarray
    name: str

    current_mask_idx: int = -1
    prediction_outputs: List[PredictionOutput] = field(default_factory=lambda: [])
    
@dataclass
class StereoImageData:
    left: List[ImageData] = None
    right: List[ImageData] = None
    frame_ids: List[str] = None

def load_image_data(path, test_num = -1):
    image_datas = []
    counter = 0
    for file in sorted(glob.glob(path)):
        img = cv2.imread(file)
        img_data = ImageData(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), file.split("/")[-1])
        image_datas.append(img_data)
        counter += 1
        if test_num > 0 and counter >= test_num:
            break
    
    return image_datas

def print_stereo_names(stereo_image_data, print_range):
    for i in print_range:
        print("left: {}, right: {}".format(stereo_image_data.left[i].name, stereo_image_data.right[i].name))

def load_stereo_image_data(left_path, right_path, test_num = None, already_processed: List[str] = []):
    # Test here for alot of images

    test_large_multiplier = 1 # default is 1
    left_image_datas = []
    right_image_datas = []
    for _ in range(test_large_multiplier):
        left_image_datas = left_image_datas + load_image_data(left_path, test_num)
        right_image_datas = right_image_datas + load_image_data(right_path, test_num)

    assert len(left_image_datas) == len(right_image_datas)
    # TODO temp
    frame_ids = [i for i in range(len(left_image_datas))]

    # TODO, @Paul, how do I check if left image correlates to right? unless we agree on naming the image
    # Or if there is a csv or json file, for the data information.
    # TODO, check for already processed images

    stereo_image_data = StereoImageData()
    stereo_image_data.left = left_image_datas
    stereo_image_data.right = right_image_datas
    stereo_image_data.frame_ids = frame_ids
    return stereo_image_data

def convert_mask_torch_to_opencv(sam_mask):
    h, w = sam_mask.shape[-2:]
    return sam_mask.reshape(h, w, 1) * 255
