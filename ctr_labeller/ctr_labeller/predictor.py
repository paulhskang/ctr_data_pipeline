# Copyright (c) 2025, Wilfred and Joyce Posluns Centre for Guided Innovation and Therapeutic Intervention (PCIGITI), University of Toronto
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
 
import cv2
import numpy as np
import operator
import torch
from typing import List
import sys

sys.path.append("..")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ctr_labeller.types import ImageData, PredictionOutput, StereoImageData

def apply_mask(image, mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1.0])], axis=0)
    else:
        color = np.array([30, 144, 255],dtype=np.uint8)
    h, w = mask.shape[-2:]
    mask_image = (mask.reshape(h, w, 1) * color.reshape(1, 1, -1)).astype(np.uint8)
    return cv2.addWeighted(image, 1.0, mask_image, 0.6, 0)

def prepare_image(image, transform, device):
    np_image = image.cpu().detach().numpy()
    new_image = transform.apply_image(np_image)
    new_image = torch.as_tensor(new_image, device=device.device)
    return new_image.permute(2, 0, 1).contiguous()

def collect_prediction_outputs(image_data, batched_output, input_prompts):
    image_pixel_num = image_data.image.shape[0] * image_data.image.shape[1]
    for j in range(len(batched_output)):
        output = batched_output[j]
        print(j)
        mask = output["masks"][0].cpu().detach().numpy()
        score = output["iou_predictions"][0].cpu().detach().numpy()[0]
        area_ratio = mask.sum() / image_pixel_num
        prediction_output = PredictionOutput(
                input_prompts[j],
                mask=mask,
                masked_image=apply_mask(image_data.image, mask),
                score=score,
                area_ratio=area_ratio)
        image_data.prediction_outputs.append(prediction_output)

class SAMBatchedPredictor:
    def __init__(self, data_saver, sort_based_on = "None"):
        """
        sort_based_on: Valid options are None, highest_score, lowest_area_ratio
        """
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("Current device is: ", device)
        
        sam = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt")
        sam.to(device=device)
        self.predictor = SAM2ImagePredictor(sam)
        self.predictor.model = torch.compile(self.predictor.model)

        self.input_prompts = {}
        self.sort_based_on = sort_based_on
        self.data_saver = data_saver
        self.sam = sam

    def predict_one(self, image_data, input_prompts):
        image_pixel_num =image_data.image.shape[0] * image_data.image.shape[1]
        self.predictor.set_image(image_data.image)
        point_coords = np.array(input_prompts["point_coords"]) if input_prompts["point_coords"] != [] else None
        point_labels = np.array(input_prompts["point_labels"]) if input_prompts["point_labels"] != [] else None
        box = np.array(input_prompts["box"]) if input_prompts["box"] is not None else None
        mask, score, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False)
        area_ratio = mask.sum() / image_pixel_num
        prediction_output = PredictionOutput(
                input_prompts,
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

    def predict_stereo(self, batch_data, left_input_prompts: dict, right_input_prompts: dict):
        batch_size = len(batch_data["frame_id"])

        indices = [i for i in range(batch_size)
                   if not self.data_saver.check_is_mask_processed(batch_data["frame_id"][i].item())]
        if not indices:
            return []

        left_image_datas, right_image_datas, frame_ids, all_images = [], [], [], []
        for i in indices:
            left_np  = batch_data["left_image"][i].cpu().detach().numpy()
            right_np = batch_data["right_image"][i].cpu().detach().numpy()
            left_image_datas.append(ImageData(left_np,  batch_data["left_image_name"][i],
                                              batch_data["left_image_path"][i], batch_data["frame_id"][i]))
            right_image_datas.append(ImageData(right_np, batch_data["right_image_name"][i],
                                               batch_data["right_image_path"][i], batch_data["frame_id"][i]))
            frame_ids.append(batch_data["frame_id"][i].item())
            all_images.append(left_np)
            all_images.append(right_np)

        n = len(indices)
        point_coords_batch, point_labels_batch, box_batch = [], [], []
        for _ in range(n):
            for prompts in [left_input_prompts, right_input_prompts]:
                pc = np.array(prompts["point_coords"]) if prompts["point_coords"] != [] else None
                pl = np.array(prompts["point_labels"]) if prompts["point_labels"] != [] else None
                bx = np.array(prompts["box"])           if prompts["box"] is not None    else None
                point_coords_batch.append(pc)
                point_labels_batch.append(pl)
                box_batch.append(bx)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            self.predictor.set_image_batch(all_images)
            masks_batch, scores_batch, _ = self.predictor.predict_batch(
                point_coords_batch=point_coords_batch,
                point_labels_batch=point_labels_batch,
                box_batch=box_batch,
                multimask_output=False)

        stereo_image_datas = []
        for i in range(n):
            for image_data, prompts, flat_idx in [
                (left_image_datas[i],  left_input_prompts,  i * 2),
                (right_image_datas[i], right_input_prompts, i * 2 + 1),
            ]:
                mask  = masks_batch[flat_idx][0]
                score = scores_batch[flat_idx][0]
                image_pixel_num = image_data.image.shape[0] * image_data.image.shape[1]
                area_ratio = mask.sum() / image_pixel_num
                image_data.prediction_outputs.append(PredictionOutput(
                    prompts,
                    mask=mask,
                    masked_image=apply_mask(image_data.image, mask),
                    score=score,
                    area_ratio=area_ratio))
                if self.sort_based_on == "highest_score":
                    image_data.prediction_outputs = sorted(image_data.prediction_outputs,
                                                           key=operator.attrgetter('score'), reverse=True)
                elif self.sort_based_on == "lowest_area_ratio":
                    image_data.prediction_outputs = sorted(image_data.prediction_outputs,
                                                           key=operator.attrgetter('area_ratio'))
                image_data.current_mask_idx = 0
            stereo_image_datas.append(
                StereoImageData(frame_id=frame_ids[i], left=left_image_datas[i], right=right_image_datas[i]))
        return stereo_image_datas

    # Might still want these, batched prediction

    # def predict(self, image_datas: List[ImageData], frame_ids: List[str], input_prompts: List[dict]):
    #     image_pixel_num = image_datas[0].image.shape[0] * image_datas[0].image.shape[1]
    #     for i  in range(len(image_datas)):
    #         image_data = image_datas[i]
    #         if self.data_saver.check_is_mask_processed(frame_ids[i]):
    #             continue
    #         self.predictor.set_image(image_data.image)
    #         for input_prompt in input_prompts:
    #             mask, score, _ = self.predictor.predict(
    #                 point_coords=input_prompt["point_coords"],
    #                 point_labels=input_prompt["point_labels"],
    #                 box=input_prompt["box"],
    #                 multimask_output=False)
    #             area_ratio = mask.sum() / image_pixel_num
    #             prediction_output = PredictionOutput(
    #                     input_prompt,
    #                     mask=mask,
    #                     masked_image=apply_mask(image_data.image, mask),
    #                     score=score[0],
    #                     area_ratio=area_ratio)
    #             image_data.prediction_outputs.append(prediction_output)
    #         # Sorting
    #         if self.sort_based_on == "highest_score":
    #             image_data.prediction_outputs = sorted(image_data.prediction_outputs, key=operator.attrgetter('score'), reverse=True)
    #         elif self.sort_based_on == "lowest_area_ratio":
    #             image_data.prediction_outputs = sorted(image_data.prediction_outputs, key=operator.attrgetter('area_ratio'))
    #         image_data.current_mask_idx = 0

   # def __append_to_batch(self, batched_input, image, input_prompts: List[dict]):
    #     prepared_image = prepare_image(image, self.resize_transform, self.sam)
    #     for input_prompt in input_prompts:
    #         to_append_dict = {'image': prepared_image, 'original_size': image.shape[:2]}
    #         assert ((input_prompt["box"] is not None) or (input_prompt["point_coords"] is not None))
    #         if input_prompt["box"] is not None:
    #             box = self.resize_transform.apply_boxes(input_prompt["box"], image.shape[:2])
    #             box_torch = torch.as_tensor(box, dtype=torch.float, device=self.sam.device)
    #             to_append_dict["box"] = box_torch[None, :]
    #         else:
    #             to_append_dict["box"] = None
    #         if input_prompt["point_coords"] is not None:
    #             point_coords = self.resize_transform.apply_coords(input_prompt["point_coords"], image.shape[:2])
    #             coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.sam.device)
    #             labels_torch = torch.as_tensor(input_prompt["point_labels"], dtype=torch.int, device=self.sam.device)
    #             to_append_dict["point_coords"] = coords_torch[None, :, :]
    #             to_append_dict["point_labels"] = labels_torch[None, :]
    #         else:
    #             to_append_dict["point_coords"] = None
    #             to_append_dict["point_labels"] = None
    #         batched_input.append(to_append_dict)

    # Doesn't work for some reason
    # def predict_stereo_batched(self, batch_data,
    #                    left_input_prompts: List[dict], right_input_prompts: List[dict]):
    #     assert len(left_input_prompts) == len(right_input_prompts)
    #     batch_size = len(batch_data["frame_id"])

    #     # Memory runs out this way for 12gb GPU
    #     # prompt_size = len(left_input_prompts)
    #     # for i in range(batch_size):
    #     #     batched_input = []
    #     #     self.__append_to_batch(batched_input, batch_data["left_image"][i] , left_input_prompts )
    #     #     self.__append_to_batch(batched_input, batch_data["right_image"][i], right_input_prompts)
    #     #     batched_output = self.sam(batched_input, multimask_output=False)

    #     stereo_image_datas = []
    #     for i in range(batch_size):
    #         batched_input = []
    #         self.__append_to_batch(batched_input, batch_data["left_image"][i] , left_input_prompts)
    #         batched_output = self.sam(batched_input, multimask_output=False)
    #         left_image_data = ImageData(batch_data["left_image"][i].cpu().detach().numpy(), batch_data["left_image_name"][i].item())
    #         collect_prediction_outputs(left_image_data, batched_output, left_input_prompts)
    #         input('hehe')

    #         batched_input = []
    #         self.__append_to_batch(batched_input, batch_data["right_image"][i] , right_input_prompts)
    #         batched_output = self.sam(batched_input, multimask_output=False)
    #         right_image_data = ImageData(batch_data["right_image"][i].cpu().detach().numpy(), batch_data["right_image_name"][i].item())
    #         collect_prediction_outputs(right_image_data, batched_output, right_input_prompts)
    #         stereo_image_datas.append(StereoImageData(frame_id=batch_data["frame_id"][i], left=left_image_data, right=right_image_data))
    #     return stereo_image_datas