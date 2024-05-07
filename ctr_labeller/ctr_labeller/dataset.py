from PIL import Image
import torch
import os

from ctr_labeller.datasaver import DataSaver
from ctr_labeller.types import ImageData, StereoImageData2

class StereoDataloader(torch.utils.data.Dataset):
    def __init__(self, root_path, left_prefix, right_prefix, filetype = "png") -> None:
        self.datasaver = DataSaver(root_path, must_have_csv=True)
        self.root_path = root_path
        self.frame_infos = []
        for key, value in self.datasaver.reference_dict:
            # People online say you have to know beforehand how to organize your data into pytorch dataset
            if self.datasaver.check_is_mask_processed(key): 
                continue
            self.frame_infos.append({
                "frame_id": key,
                "left_image_name": value["left_image"],
                "left_image_path": os.path.join(root_path, value["left_image"]),
                "right_image_name": value["right_image_name"],
                "right_image_path": os.path.join(root_path, value["right_image"])})

    def __len__(self):
        return len(self.frame_infos)

    def __get_item__(self, idx):
        frame_info = self.frame_infos[idx]
        left_image_data = ImageData(
            image=Image.open(frame_info["left_image_path"]), 
            name=frame_info["left_image_name"])
        right_image_data = ImageData(
            image=Image.open(frame_info["right_image_path"]),
            name=frame_info["right_image_name"])
        return StereoImageData2(
            frame_id=frame_info["frame_id"], left=left_image_data, right=right_image_data)

    # The non-csv way, maybe not use:
    # def __init__(self, root_path, left_prefix, right_prefix, filetype = "png") -> None:
    #     left_paths = os.path.join(root_path, "{}*.{}".format(left_prefix, filetype))
    #     self.left_filenames = sorted(glob.glob(left_paths))
    #     right_paths = os.path.join(root_path, "{}*.{}".format(right_prefix, filetype))
    #     self.right_filenames = sorted(glob.glob(right_paths))
    #     assert len(self.left_filenames) == len(self.right_filenames)
