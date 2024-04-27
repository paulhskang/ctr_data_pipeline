import cv2
import numpy as np
import torch
import sys
from dataclasses import dataclass
from typing import List, Dict

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from ctr_labeller.types import ImageData, PredictionOutput

@dataclass
class InputPrompt:
    bounding_box: np.ndarray = None
    point_coords: np.ndarray = None
    point_labels: np.ndarray = None

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

def apply_mask(image, mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1.0])], axis=0)
    else:
        color = np.array([30, 144, 255],dtype=np.uint8)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return cv2.addWeighted(image, 1.0, mask_image, 0.6, 0)

class SAMBatchedPredictor:
    def __init__(self):
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.input_prompts = {}
        self.resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    def predict(self, image_datas: List[ImageData], input_prompts: List[dict]):
        for image_data in image_datas:
            self.predictor.set_image(image_data.image)
            for input_prompt in input_prompts:
                mask, score, _ = self.predictor.predict(
                    point_coords=input_prompt["point_coords"],
                    point_labels=input_prompt["point_labels"],
                    box=input_prompt["box"],
                    multimask_output=False)
                image_data.prediction_outputs.append(PredictionOutput(input_prompt["name"], mask, apply_mask(image_data.image, mask), score))
            image_data.current_mask_idx = 0
