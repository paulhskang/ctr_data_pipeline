
import atexit
import copy
import cv2
import json
import numpy as np
import os
import pandas as pd
from typing import List

from ctr_labeller.types import convert_mask_torch_to_opencv

def convert_dict_array_values_to_lists(dictionary, exceptions: List):
    for key in dictionary:
        if key in exceptions or dictionary[key] is None:
            continue
        dictionary[key] = dictionary[key].tolist()

def convert_dict_list_values_to_arrays(dictionary, exceptions: List):
    for key in dictionary:
        if key in exceptions or dictionary[key] is None:
            continue
        dictionary[key] = np.array(dictionary[key])

class DataSaver:
    def __init__(self, save_root_path, must_have_csv = False, save_image_and_masks = True):
        self.save_root_path = save_root_path
        self.save_image_and_masks = save_image_and_masks
        # self.fields = ["left_image", "right_image", "is_processed", "left_mask_fail", "right_mask_fail", "left_mask", "right_mask"]

        self.no_initial_csv = False
        self.reference_file_path = os.path.join(save_root_path, "reference.csv")
        if os.path.isfile(self.reference_file_path):
            data_frame = pd.read_csv(self.reference_file_path, index_col=0)
            self.reference_dict = data_frame.to_dict(orient='index')
        elif must_have_csv: # MOST LIKELY WE WILL ALWAYS HAVE CSV NOW
            raise ValueError("must have csv file reference.csv in save_root_path: {}".format(save_root_path))
        else:
            self.no_initial_csv = True
            self.reference_dict = {}

        self.left_input_prompts = None
        self.right_input_prompts = None
        self.is_input_prompts_available = False
        self.prompts_file_path = os.path.join(save_root_path, "input_prompts.json")
        if os.path.isfile(self.prompts_file_path):
            self.is_input_prompts_available = True
            f = open(self.prompts_file_path)
            data = json.load(f)
            for input_prompt in data["left_input_prompts"]:
                convert_dict_list_values_to_arrays(input_prompt, ["name"])
            for input_prompt in data["right_input_prompts"]:
                convert_dict_list_values_to_arrays(input_prompt, ["name"])
            self.left_input_prompts = data["left_input_prompts"]
            self.right_input_prompts = data["right_input_prompts"]

        self.mask_path = os.path.join(save_root_path, "masks")
        if not os.path.exists(self.mask_path): # Means no format path
            os.mkdir(self.mask_path)

        self.image_and_masks_path = os.path.join(save_root_path, "image_and_masks")
        if not os.path.exists(self.image_and_masks_path): # Means no format path
            os.mkdir(self.image_and_masks_path)

        atexit.register(self.__destructor)

    def check_is_mask_processed(self, frame_id):
        if not "is_processed" in self.reference_dict[frame_id]:
            return False    
        is_processed = self.reference_dict[frame_id]["is_processed"]
        if pd.isna(is_processed):
            return False
        return is_processed
    
    def __save_current_mask(self, image_data, collected_batch_num):
        current_prediction_output = image_data.prediction_outputs[image_data.current_mask_idx]

        # if os.path.exists(fullpath_mask): # Not working properly
        #     print("{} path exists! Overriding".format(fullpath_mask))
        try:
            os.makedirs(os.path.join(self.mask_path, str(collected_batch_num)))
        except FileExistsError:
            pass
        mask_name = "mask_{}".format(image_data.name)
        fullpath_mask = os.path.join(self.mask_path, str(collected_batch_num), mask_name)
        cv2.imwrite(fullpath_mask, convert_mask_torch_to_opencv(current_prediction_output.mask))
        relative_mask_path_from_root = os.path.join(os.path.split(self.mask_path)[1], str(collected_batch_num), mask_name)
        
        # Image and masks
        relative_image_and_mask_path_from_root = ""
        if not self.save_image_and_masks:
            return relative_mask_path_from_root, relative_image_and_mask_path_from_root

        # if os.path.exists(fullpath_image_and_mask): # Not working properly
        #     print("{} path exists! Overriding".format(fullpath_image_and_mask))
        try:
            os.makedirs(os.path.join(self.image_and_masks_path, str(collected_batch_num)))
        except FileExistsError:
            pass
        image_and_mask_name = "image_and_mask_{}".format(image_data.name)
        fullpath_image_and_mask = os.path.join(self.image_and_masks_path, str(collected_batch_num), image_and_mask_name)
        cv2.imwrite(fullpath_image_and_mask, cv2.cvtColor(current_prediction_output.masked_image, cv2.COLOR_RGB2BGR))
        relative_image_and_mask_path_from_root = os.path.join(
            os.path.split(self.image_and_masks_path)[1], str(collected_batch_num), image_and_mask_name)

        return relative_mask_path_from_root, relative_image_and_mask_path_from_root

    def save_current_stereo_masks(self, frame_id, collected_batch_num, image_data_left, image_data_right):
        left_mask_path = ""
        left_image_and_mask_path = ""
        if image_data_left.is_save_mask:
            left_mask_path, left_image_and_mask_path = self.__save_current_mask(image_data_left, collected_batch_num)
        right_mask_path = ""
        right_image_and_mask_path = ""
        if image_data_right.is_save_mask:
            right_mask_path, right_image_and_mask_path = self.__save_current_mask(image_data_right, collected_batch_num)

        # Initial csv has left_image_path and right_image_path
        self.reference_dict[frame_id]["is_processed"] = True
        self.reference_dict[frame_id]["left_mask_fail"] = not image_data_left.is_save_mask
        self.reference_dict[frame_id]["right_mask_fail"] = not image_data_right.is_save_mask
        self.reference_dict[frame_id]["left_mask_path"] = left_mask_path
        self.reference_dict[frame_id]["right_mask_path"] = right_mask_path
        self.reference_dict[frame_id]["left_image_and_mask_path"] = left_image_and_mask_path
        self.reference_dict[frame_id]["right_image_and_mask_path"] = right_image_and_mask_path

    def __destructor(self):
        df = pd.DataFrame.from_dict(self.reference_dict, orient='index')
        df.index.name = "frame_id"
        df.to_csv(self.reference_file_path)
        print("DataSaver | Finished saving reference.csv")

    def save_csv(self):
        df = pd.DataFrame.from_dict(self.reference_dict, orient='index')
        df.index.name = "frame_id"
        df.to_csv(self.reference_file_path)
        print("DataSaver | Finished saving reference.csv")

    def get_input_prompts(self):
        return self.left_input_prompts, self.right_input_prompts
    
    def save_input_prompts(self, left_input_prompts, right_input_prompts, filename = ""):
        left_input_prompts = copy.deepcopy(left_input_prompts)
        right_input_prompts = copy.deepcopy(right_input_prompts)
        file_to_save = "input_prompts.json"
        if filename != "":
            file_to_save = filename
        for input_prompt in left_input_prompts:
            convert_dict_array_values_to_lists(input_prompt, ["name"])
        for input_prompt in right_input_prompts:
            convert_dict_array_values_to_lists(input_prompt, ["name"])

        entire_dict = {}
        entire_dict["right_input_prompts"] = right_input_prompts
        entire_dict["left_input_prompts"] = left_input_prompts
        
        with open(os.path.join(self.save_root_path, file_to_save), 'w') as f:
            json.dump(entire_dict, f, ensure_ascii=False, indent=2)
