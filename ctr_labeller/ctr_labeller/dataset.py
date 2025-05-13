from PIL import Image
import torch
import os
import copy
import math
import numpy as np

from ctr_labeller.datasaver import DataSaver


class StereoDataSet(torch.utils.data.Dataset):
    def __init__(self, root_path, datasaver: DataSaver, batch_num = -1) -> None:
        self.datasaver = datasaver
        self.root_path = root_path
        self.frame_infos = []
        for key, value in self.datasaver.reference_dict.items():
            if self.__is_frame_not_valid(key, value, batch_num):
                continue
            frame_info = {
                "frame_id": key,
                "left_image_path": os.path.join(root_path, value["left_image_path"]),
                "right_image_path": os.path.join(root_path, value["right_image_path"])}
            frame_info["left_image_name"] = os.path.split(frame_info["left_image_path"])[1]
            frame_info["right_image_name"] = os.path.split(frame_info["right_image_path"])[1]
            self.frame_infos.append(frame_info)

    def __is_frame_not_valid(self, key, value, batch_num):
        if not key in self.datasaver.reference_dict:
            return True
        if self.datasaver.check_is_mask_processed(key): 
            return True
        if batch_num >= 0:
            if not "batch_num" in value:
                print("Stereo DataSet | Warning!!! batch_num specified, " + 
                        "but batch_num not found in reference.csv, will process this frame [{}]".format(key))
            elif math.isnan(value["batch_num"]):
                print("Stereo DataSet | Warning!!! batch_num specified, " + 
                        "but batch_num is empty in reference.csv, will process this frame [{}]".format(key))
            elif batch_num != value["batch_num"]:
                return True
        # else
        return False

    def __len__(self):
        return len(self.frame_infos)

    def __getitem__(self, idx):
        frame_info = copy.deepcopy(self.frame_infos[idx])
        frame_info["left_image"] = np.array(Image.open(frame_info["left_image_path"]))
        frame_info["right_image"] = np.array(Image.open(frame_info["right_image_path"]))
        return frame_info

