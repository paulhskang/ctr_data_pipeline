from PIL import Image
import torch
import os
import copy
import numpy as np

from ctr_labeller.datasaver import DataSaver

class StereoDataSet(torch.utils.data.Dataset):
    def __init__(self, root_path, datasaver: DataSaver, batch_num = -1) -> None:
        self.datasaver = datasaver
        self.root_path = root_path
        self.frame_infos = []
        for key, value in self.datasaver.reference_dict.items():
            if not key in self.datasaver.reference_dict:
                continue
            if self.datasaver.check_is_mask_processed(key): 
                continue

            if batch_num != -1:
                if "batch_num" in self.datasaver.reference_dict and \
                    batch_num != self.datasaver.reference_dict["batch_num"]:
                    continue

            frame_info = {
                "frame_id": key,
                "left_image_path": os.path.join(root_path, value["left_image_path"]),
                "right_image_path": os.path.join(root_path, value["right_image_path"])}
            frame_info["left_image_name"] = os.path.split(frame_info["left_image_path"])[1]
            frame_info["right_image_name"] = os.path.split(frame_info["right_image_path"])[1]
            self.frame_infos.append(frame_info)

    def __len__(self):
        return len(self.frame_infos)

    def __getitem__(self, idx):
        frame_info = copy.deepcopy(self.frame_infos[idx])
        frame_info["left_image"] = np.array(Image.open(frame_info["left_image_path"]))
        frame_info["right_image"] = np.array(Image.open(frame_info["right_image_path"]))
        return frame_info

    # The non-csv way, maybe not use:
    # def __init__(self, root_path, left_prefix, right_prefix, filetype = "png") -> None:
    #     left_paths = os.path.join(root_path, "{}*.{}".format(left_prefix, filetype))
    #     self.left_filenames = sorted(glob.glob(left_paths))
    #     right_paths = os.path.join(root_path, "{}*.{}".format(right_prefix, filetype))
    #     self.right_filenames = sorted(glob.glob(right_paths))
    #     assert len(self.left_filenames) == len(self.right_filenames)
