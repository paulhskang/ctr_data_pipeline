import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from ctr_labeller.types import load_stereo_image_data, print_stereo_names
from ctr_labeller.datasaver import DataSaver
from ctr_labeller.ui import CTRLabellerApp, CTRLabellerAppConfig
from ctr_labeller.predictor import SAMBatchedPredictor
from ctr_labeller.config.utils import parse_config, configure
from ctr_labeller.dataset import StereoDataloader
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def debug_input(image, input_box, input_point = None):
    plt.figure(figsize=(10,10))
    show_box(input_box, plt.gca())
    if input_point:
        show_points(input_point, np.array([1]), plt.gca(), marker_size=40)
    plt.imshow(image)
    plt.axis('on')

@configure
class CTRLabellerConfig:
    data_path: str
    app_config: CTRLabellerAppConfig
    debug_inputs: bool = True
    apply_mask: bool = True
    test_num: int = -1
    sort_based_on: str = "None"
    save_root_path: str = ""

if __name__ == "__main__":

    # Config
    config = parse_config(CTRLabellerConfig, yaml_arg='--config')

    # SAM Input
    
    # TODO, another gui for clicking points and boxes and save as prompts
    # if config.debug_inputs:
    #     img_idx = 0
    #     debug_input(stereo_image_data[img_idx].left.image, left_input_box)
    #     debug_input(stereo_image_data[img_idx].right.image, right_input_box)
    #     plt.show()



    # New Stuff
    stereo_image_dataset = StereoDataloader(config.save_root_path, "cam1_", "cam2_")
    loader = torch.utils.data.DataLoader(stereo_image_dataset, batch_size=4,
                                         pin_memory=True, num_workers=4, shuffle=False)


    # SAM Create Masks
    if config.apply_mask:
        left_input_prompts = [
            {
                "name": "box_and_point",
                "box": np.array([300, 500, 1600, 1400]),
                "point_coords": np.array([[710, 260]]),
                "point_labels": np.array([1])
            },
            {
                "name": "point",
                "box": None,
                "point_coords": np.array([[710, 260]]),
                "point_labels": np.array([1])
            }
        ]

        right_input_prompts = [
            {
                "name": "box_and_point",
                "box": np.array([500, 600, 1500, 1400]),
                "point_coords": np.array([[1129, 400]]),
                "point_labels": np.array([1])
            },
            {
                "name": "point",
                "box": None,
                "point_coords": np.array([[1129, 400]]),
                "point_labels": np.array([1])
            }
        ]

        predictor = SAMBatchedPredictor(stereo_image_dataset.datasaver, config.sort_based_on)
        stereo_image_datas = []
        print("SAM is creating masks ...")

        num = 0
        for batch in loader:
            num = num + len(batch)
            stereo_image_datas += predictor.predict_stereo(batch, left_input_prompts, right_input_prompts)
            print(num)
            if num > config.test_num:
                break

        print("SAM finished creating masks!")

    # Start the app
    app = CTRLabellerApp(config.app_config, stereo_image_dataset.datasaver)
    app.title("CTR SAM Labeller, press [n] to save and proceed with next set")

    app.set_stereo_image_datas(stereo_image_datas)

    app.bind("n", app.keypress_event)
    app.mainloop()
