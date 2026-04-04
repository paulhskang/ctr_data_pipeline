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
import os
import torch
from PIL import Image as PILImage
import sys

sys.path.append("..")
from sam2.build_sam import build_sam2, build_sam2_video_predictor
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

        self.video_predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt", device=device)

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

    def predict_stereo_video(self, stereo_dataset, left_input_prompts, right_input_prompts,
                             chunk_size=200):
        """Process frames in chunks using SAM2 video predictor for temporal tracking.
        Prompt only the first frame of each chunk; SAM2 propagates across chunk_size frames.
        Requires JPEG images with integer filenames (e.g. 0001.jpg).
        """
        import tempfile, shutil
        from collections import defaultdict

        lpc = np.array(left_input_prompts["point_coords"])  if left_input_prompts["point_coords"]  != [] else None
        lpl = np.array(left_input_prompts["point_labels"])  if left_input_prompts["point_labels"]  != [] else None
        lbx = np.array(left_input_prompts["box"])           if left_input_prompts["box"] is not None else None
        rpc = np.array(right_input_prompts["point_coords"]) if right_input_prompts["point_coords"] != [] else None
        rpl = np.array(right_input_prompts["point_labels"]) if right_input_prompts["point_labels"] != [] else None
        rbx = np.array(right_input_prompts["box"])          if right_input_prompts["box"] is not None else None

        # Group frame_infos by left directory — each folder is a separate video sequence
        dir_to_frame_infos = defaultdict(list)
        for fi in stereo_dataset.frame_infos:
            dir_to_frame_infos[os.path.dirname(fi["left_image_path"])].append(fi)

        for left_dir, frame_infos_in_dir in sorted(dir_to_frame_infos.items()):
            # Preserve original order from reference.csv
            frame_infos_sorted = frame_infos_in_dir
            all_vidxs = list(range(len(frame_infos_sorted)))

            for chunk_start in range(0, len(all_vidxs), chunk_size):
                chunk_vidxs = all_vidxs[chunk_start:chunk_start + chunk_size]
                chunk_frame_infos = [frame_infos_sorted[vidx] for vidx in chunk_vidxs]
                print(f"predict_stereo_video | {left_dir} chunk frames {chunk_vidxs[0]}–{chunk_vidxs[-1]}")

                left_tmp  = tempfile.mkdtemp()
                right_tmp = tempfile.mkdtemp()
                try:
                    for local_idx, frame_info in enumerate(chunk_frame_infos):
                        os.symlink(frame_info["left_image_path"],
                                   os.path.join(left_tmp,  f"{local_idx}.jpg"))
                        os.symlink(frame_info["right_image_path"],
                                   os.path.join(right_tmp, f"{local_idx}.jpg"))

                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        left_state  = self.video_predictor.init_state(left_tmp,  offload_video_to_cpu=True)
                        right_state = self.video_predictor.init_state(right_tmp, offload_video_to_cpu=True)
                        self.video_predictor.add_new_points_or_box(
                            left_state,  frame_idx=0, obj_id=1, points=lpc, labels=lpl, box=lbx)
                        self.video_predictor.add_new_points_or_box(
                            right_state, frame_idx=0, obj_id=1, points=rpc, labels=rpl, box=rbx)
                        left_masks  = {}
                        right_masks = {}
                        for local_idx, _, masks in self.video_predictor.propagate_in_video(left_state):
                            left_masks[local_idx]  = (masks[0] > 0).cpu().numpy()
                        for local_idx, _, masks in self.video_predictor.propagate_in_video(right_state):
                            right_masks[local_idx] = (masks[0] > 0).cpu().numpy()
                finally:
                    shutil.rmtree(left_tmp)
                    shutil.rmtree(right_tmp)

                for local_idx, frame_info in enumerate(chunk_frame_infos):
                    frame_id = frame_info["frame_id"]
                    if self.data_saver.check_is_mask_processed(frame_id):
                        continue
                    left_np   = np.array(PILImage.open(frame_info["left_image_path"]))
                    right_np  = np.array(PILImage.open(frame_info["right_image_path"]))
                    left_mask  = left_masks[local_idx]
                    right_mask = right_masks[local_idx]

                    left_image_data = ImageData(left_np, frame_info["left_image_name"],
                                                frame_info["left_image_path"], frame_info["frame_id"])
                    left_image_data.prediction_outputs.append(PredictionOutput(
                        left_input_prompts,
                        mask=left_mask,
                        masked_image=apply_mask(left_np, left_mask),
                        score=0.0,
                        area_ratio=left_mask.sum() / (left_np.shape[0] * left_np.shape[1])))
                    left_image_data.current_mask_idx = 0

                    right_image_data = ImageData(right_np, frame_info["right_image_name"],
                                                 frame_info["right_image_path"], frame_info["frame_id"])
                    right_image_data.prediction_outputs.append(PredictionOutput(
                        right_input_prompts,
                        mask=right_mask,
                        masked_image=apply_mask(right_np, right_mask),
                        score=0.0,
                        area_ratio=right_mask.sum() / (right_np.shape[0] * right_np.shape[1])))
                    right_image_data.current_mask_idx = 0

                    yield StereoImageData(frame_id=frame_id, left=left_image_data, right=right_image_data)

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