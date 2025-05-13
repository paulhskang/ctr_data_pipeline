
import atexit
import copy
import cv2
from datetime import datetime
import json
import numpy as np
import os
import pandas as pd
from typing import List
from ctr_labeller.types import ImageData

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
    def __init__(self, save_root_path, input_prompt_json_name, save_image_appended_with_masks = False):
        self.save_root_path = save_root_path
        self.input_prompt_json_name = input_prompt_json_name
        print(self.input_prompt_json_name)
        self.save_image_appended_with_masks = save_image_appended_with_masks

        self.reference_file_path = os.path.join(save_root_path, "reference.csv")
        if os.path.isfile(self.reference_file_path):
            data_frame = pd.read_csv(self.reference_file_path, index_col=0)
            self.reference_dict = data_frame.to_dict(orient='index')
        else:
            raise ValueError("Must have csv file reference.csv in save_root_path: {}".format(save_root_path))

        self.left_input_prompts = None
        self.right_input_prompts = None

        atexit.register(self.__destructor)

    def check_is_mask_processed(self, frame_id):
        if not "is_processed" in self.reference_dict[frame_id]:
            return False
        is_processed = self.reference_dict[frame_id]["is_processed"]
        if pd.isna(is_processed):
            return False
        left_mask_fail = self.reference_dict[frame_id]["left_mask_fail"]
        right_mask_fail = self.reference_dict[frame_id]["right_mask_fail"]
        return (is_processed and not left_mask_fail and not right_mask_fail)
    
    def __find_images_folder(self, full_image_data_path):
        is_image_folder_found = False
        image_folder_from_root = full_image_data_path
        while not is_image_folder_found:
            split = os.path.split(image_folder_from_root)
            if split[1] == "imgs":
                is_image_folder_found = True
                break
            # else
            image_folder_from_root = split[0]
        return image_folder_from_root

    def __get_relative_folders_in_between(self, images_folder, full_image_data_path):
        rel_path = os.path.relpath(full_image_data_path, images_folder)
        return os.path.split(rel_path)[0]

    def __save_image_instance(self, image, image_name, image_data_path, prepend_img_str, prepend_folder):
       # image_data.path is full path
       # Will try to create masks folder one up because assumes images are in imgs folder
        images_folder_from_root = self.__find_images_folder(image_data_path)
        rel_path_in_between = self.__get_relative_folders_in_between(images_folder_from_root, image_data_path)
        up_one_images_folder_from_root = os.path.dirname(images_folder_from_root)

        prepend_folder_with_rel_path_in_between_from_root = os.path.join(up_one_images_folder_from_root, prepend_folder, rel_path_in_between)
        if not os.path.isdir(prepend_folder_with_rel_path_in_between_from_root):
            os.makedirs(prepend_folder_with_rel_path_in_between_from_root)

        save_image_name = prepend_img_str + "{}".format(image_name)
        fullpath_image = os.path.join(prepend_folder_with_rel_path_in_between_from_root, save_image_name)
        cv2.imwrite(fullpath_image, image)
        relative_image_path_from_save_folder = os.path.relpath(fullpath_image, self.save_root_path)
        return relative_image_path_from_save_folder

    def __save_current_mask(self, image_data: ImageData):
        current_prediction_output = image_data.prediction_outputs[image_data.current_mask_idx]
        
        relative_mask_path_from_save_folder = self.__save_image_instance(image=convert_mask_torch_to_opencv(current_prediction_output.mask),
                                                                         image_name=image_data.name, image_data_path=image_data.path,
                                                                         prepend_img_str="mask_", prepend_folder="masks")
        relative_image_and_mask_path_from_root = ""
        if self.save_image_appended_with_masks:
            relative_image_and_mask_path_from_root = self.__save_image_instance(image=cv2.cvtColor(current_prediction_output.masked_image, cv2.COLOR_RGB2BGR),
                                                                         image_name=image_data.name, image_data_path=image_data.path,
                                                                         prepend_img_str="image_and_mask_", prepend_folder="image_and_masks")
        return relative_mask_path_from_save_folder, relative_image_and_mask_path_from_root

    def save_current_stereo_masks(self, frame_id,
                                  image_data_left: ImageData, image_data_right: ImageData):
        left_mask_path = ""
        rel_left_image_and_mask_path = ""
        if image_data_left.is_save_mask:
            left_mask_path, rel_left_image_and_mask_path = self.__save_current_mask(image_data_left)
        right_mask_path = ""
        rel_right_image_and_mask_path = ""
        if image_data_right.is_save_mask:
            right_mask_path, rel_right_image_and_mask_path = self.__save_current_mask(image_data_right)

        # Initial csv has left_image_path and right_image_path
        self.reference_dict[frame_id]["is_processed"] = True
        self.reference_dict[frame_id]["left_mask_fail"] = not image_data_left.is_save_mask
        self.reference_dict[frame_id]["right_mask_fail"] = not image_data_right.is_save_mask
        self.reference_dict[frame_id]["left_mask_path"] = left_mask_path
        self.reference_dict[frame_id]["right_mask_path"] = right_mask_path
        if self.save_image_appended_with_masks:
            self.reference_dict[frame_id]["left_image_and_mask_path"] = rel_left_image_and_mask_path
            self.reference_dict[frame_id]["right_image_and_mask_path"] = rel_right_image_and_mask_path

    def __destructor(self):
        self.save_csv()

    def save_csv(self):
        df = pd.DataFrame.from_dict(self.reference_dict, orient='index')
        df.index.name = "frame_id"
        df.to_csv(self.reference_file_path)
        print("DataSaver | Finished saving or updating [reference.csv]")

    def is_input_prompts_available(self):
        if self.input_prompt_json_name == "":
            return False
        prompts_file_path = os.path.join(self.save_root_path, self.input_prompt_json_name)
        if not os.path.isfile(prompts_file_path):
            print("DataSaver | No input prompt file [{}].".format(self.input_prompt_json_name))
            return False
        try:
            f = open(prompts_file_path)
            data = json.load(f)
            self.left_input_prompts = data["left_input_prompts"]
            self.right_input_prompts = data["right_input_prompts"]
            if (self.left_input_prompts["point_coords"] != [] or self.left_input_prompts["box"] is not None) \
                and (self.right_input_prompts["point_coords"] != [] or self.right_input_prompts["box"] is not None):
                return True
        except:
            print("DataSaver | Input prompts from [{}] not in valid format".format(self.input_prompt_json_name))
        # else:
        return False

    def get_input_prompts(self):
        return self.left_input_prompts, self.right_input_prompts
    
    def save_input_prompts(self, left_input_prompts, right_input_prompts):
        left_input_prompts = copy.deepcopy(left_input_prompts)
        right_input_prompts = copy.deepcopy(right_input_prompts)
        left_right_input_prompts = {}
        left_right_input_prompts["right_input_prompts"] = right_input_prompts
        left_right_input_prompts["left_input_prompts"] = left_input_prompts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompts_file_path = os.path.join(self.save_root_path,
                                         "input_prompts_{}.json".format(timestamp))
        with open(prompts_file_path, 'w') as f:
            json.dump(left_right_input_prompts, f, ensure_ascii=False, indent=2)
        return prompts_file_path

