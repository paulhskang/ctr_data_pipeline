import cv2
import numpy as np
import operator
import torch
import sys

from typing import List

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from ctr_labeller.types import ImageData, PredictionOutput, StereoImageData2

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


from segment_anything.utils.transforms import ResizeLongestSide

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

def prepare_box(box, transform, device):
    torch_box = torch.as_tensor(box, device=device)
    transform.apply_boxes_torch(torch_box, )

class SAMBatchedPredictor:
    def __init__(self, data_saver, sort_based_on = "None"):
        """
        sort_based_on: Valid options are None, highest_score, lowest_area_ratio
        """
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.input_prompts = {}
        self.resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
        self.sort_based_on = sort_based_on
        self.data_saver = data_saver
        self.sam = sam


    def append_to_batch(self, batched_input, image, input_prompts: List[dict]):
        prepared_image = prepare_image(image, self.resize_transform, self.sam)
        for input_prompt in input_prompts:
            to_append_dict = {'image': prepared_image, 'original_size': image.shape[:2]}
            assert ((input_prompt["box"] is not None) or (input_prompt["point_coords"] is not None))
            if input_prompt["box"] is not None:
                torch_box = torch.as_tensor(input_prompt["box"], device=self.sam.device)
                to_append_dict["box"] = self.resize_transform.apply_boxes_torch(torch_box, image.shape[:2])
            if input_prompt["point_coords"] is not None:
                to_append_dict["point_labels"] = torch.as_tensor(input_prompt["point_labels"], device=self.sam.device)
                torch_point_coords = torch.as_tensor(input_prompt["point_coords"], device=self.sam.device)
                to_append_dict["point_coords"] = self.resize_transform.apply_coords_torch(torch_point_coords, image.shape[:2])
            batched_input.append(to_append_dict)

    def predict_stereo(self, batch_data,
                       left_input_prompts: List[dict], right_input_prompts: List[dict]):
        assert len(left_input_prompts) == len(right_input_prompts)
        prompt_size = len(left_input_prompts)

        for i in range(len(batch_data)):
            stereo_image_data = stereo_image_datas[i]
            batched_input = []
            self.append_to_batch(batched_input, stereo_image_data.left.image , left_input_prompts )
            self.append_to_batch(batched_input, stereo_image_data.right.image, right_input_prompts)
            batched_output = self.sam(batched_input, multimask_output=False)


    def predict(self, image_datas: List[ImageData], frame_ids: List[str], input_prompts: List[dict]):
        image_pixel_num = image_datas[0].image.shape[0] * image_datas[0].image.shape[1]
        for i  in range(len(image_datas)):
            image_data = image_datas[i]
            if self.data_saver.check_is_mask_processed(frame_ids[i]):
                continue
            self.predictor.set_image(image_data.image)
            for input_prompt in input_prompts:
                mask, score, _ = self.predictor.predict(
                    point_coords=input_prompt["point_coords"],
                    point_labels=input_prompt["point_labels"],
                    box=input_prompt["box"],
                    multimask_output=False)
                area_ratio = len(np.column_stack(np.where(mask > 0))) / image_pixel_num
                prediction_output = PredictionOutput(
                        input_prompt,
                        mask=mask,
                        masked_image=apply_mask(image_data.image, mask),
                        score=score[0],
                        area_ratio=area_ratio)
                image_data.prediction_outputs.append(prediction_output)
            # Sorting
            if self.sort_based_on == "highest_score":
                image_data.prediction_outputs = sorted(image_data.prediction_outputs, key=operator.attrgetter('score'), reverse=True)
            elif self.sort_based_on == "lowest_area_ratio":
                image_data.prediction_outputs = sorted(image_data.prediction_outputs, key=operator.attrgetter('area_ratio'))
            image_data.current_mask_idx = 0
