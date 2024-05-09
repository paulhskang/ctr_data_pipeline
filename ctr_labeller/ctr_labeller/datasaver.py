
import cv2
import os
import pandas as pd
import atexit
from ctr_labeller.types import convert_mask_torch_to_opencv

class DataSaver:
    def __init__(self, save_root_path, must_have_csv = False):
        self.save_root_path = save_root_path
        self.fields = ["left_image", "right_image", "is_processed", "left_mask_fail", "right_mask_fail", "left_mask", "right_mask"]

        self.reference_file_path = os.path.join(save_root_path, "reference.csv")
        if os.path.isfile(self.reference_file_path):
            data_frame = pd.read_csv(self.reference_file_path, index_col=0)
            self.reference_dict = data_frame.to_dict(orient='index')
        elif must_have_csv:
            raise ValueError("must have csv file reference.csv in save_root_path: ".format(save_root_path))
        else:
            self.reference_dict = {}

        self.mask_path = os.path.join(save_root_path, "masks") 
        if not os.path.exists(self.mask_path): # Means no format path
            os.mkdir(self.mask_path)

        atexit.register(self.__destructor)

    def check_is_mask_processed(self, frame_id):
        if not frame_id in self.reference_dict:
            return False
        if not "is_processed" in self.reference_dict[frame_id]:
            return False    
        is_processed = self.reference_dict[frame_id]["is_processed"]
        if pd.isna(is_processed):
            return False
        return is_processed

    def __save_current_mask(self, image_data):
        mask_name = "mask_{}".format(image_data.name)
        fullpath_mask = os.path.join(self.mask_path, "mask_{}".format(image_data.name))
        # if os.path.exists(fullpath_mask): # Not working properly
        #     print("{} path exists! Overriding".format(fullpath_mask))
        fullpath_image_and_mask = os.path.join(self.mask_path, "image_and_mask_{}".format(image_data.name))
        # if os.path.exists(fullpath_image_and_mask): # Not working properly
        #     print("{} path exists! Overriding".format(fullpath_image_and_mask))

        current_prediction_output = image_data.prediction_outputs[image_data.current_mask_idx]
        cv2.imwrite(fullpath_mask, convert_mask_torch_to_opencv(current_prediction_output.mask))
        cv2.imwrite(fullpath_image_and_mask, cv2.cvtColor(current_prediction_output.masked_image, cv2.COLOR_RGB2BGR))
        return os.path.join("mask", mask_name)

    def save_current_stereo_masks(self, frame_id, image_data_left, image_data_right):
        left_mask_name = ""
        if image_data_left.is_save_mask:
            left_mask_name = self.__save_current_mask(image_data_left)
        right_mask_name = ""
        if image_data_right.is_save_mask:
            right_mask_name = self.__save_current_mask(image_data_right)
        self.reference_dict[frame_id] = {
            "left_image": image_data_left.name, "right_image": image_data_right.name, 
            "is_processed": True, 
            "left_mask_fail": not image_data_left.is_save_mask, 
            "right_mask_fail": not image_data_right.is_save_mask, 
            "left_mask": left_mask_name, "right_mask": right_mask_name}

    def __destructor(self):
        df = pd.DataFrame.from_dict(self.reference_dict, orient='index')
        df.index.name = "frame_id"
        df.to_csv(self.reference_file_path)
