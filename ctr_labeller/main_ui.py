import numpy as np
import matplotlib.pyplot as plt
import torch
import threading

from ctr_labeller.types import StereoImageDataQueue
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

class SAMBatchedPredictorThread(threading.Thread):
    def __init__(self, datasaver, stereo_image_queue, dataloader, config, left_input_prompts, right_input_prompts):
        super(SAMBatchedPredictorThread, self).__init__()
        self.predictor = SAMBatchedPredictor(datasaver, config.sort_based_on)
        self.stereo_image_queue = stereo_image_queue
        self.dataloader = dataloader
        self.config = config
        self.left_input_prompts = left_input_prompts
        self.right_input_prompts = right_input_prompts

        # self.done = False
        self.start()

    def run(self):
        num = 0
        batch_len = len(self.dataloader)
        print("batch_len: ", batch_len)
        is_last_batch = False
        batch_idx = 0
        for batch in self.dataloader:
            num = num + len(batch)
            if batch_idx >= batch_len - 1:
                is_last_batch = True
            print("before")
            stereo_image_datas = self.predictor.predict_stereo(batch, self.left_input_prompts, self.right_input_prompts)
            print("after")
            stereo_image_queue.add_images(stereo_image_datas, is_last_batch)
            batch_idx += 1

        print ("------ BATCH IS DONE ------ ")
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

    grid_num = config.app_config.selection_grid_size[0] * config.app_config.selection_grid_size[1]
    stereo_image_dataset = StereoDataloader(config.save_root_path, "cam1_", "cam2_")
    loader = torch.utils.data.DataLoader(stereo_image_dataset, batch_size=grid_num,
                                         pin_memory=True, num_workers=4, shuffle=False)

    # SAM Create Masks
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

    stereo_image_queue = StereoImageDataQueue(size_to_get=grid_num)
    predictor_thread = SAMBatchedPredictorThread(
        stereo_image_dataset.datasaver, stereo_image_queue,
        loader, config, left_input_prompts, right_input_prompts)

    # Start the app
    app = CTRLabellerApp(config.app_config, stereo_image_dataset.datasaver, stereo_image_queue)
    app.title("CTR SAM Labeller, press [n] to save and proceed with next set")

    app.bind("n", app.keypress_event)
    app.mainloop()
    predictor_thread.join()
