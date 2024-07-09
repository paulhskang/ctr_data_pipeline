import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import threading
import pathlib

from ctr_labeller.ui import CTRLabellerApp, CTRLabellerAppConfig
from ctr_labeller.input_prompt_ui import InputPromptGenerationApp
from ctr_labeller.predictor import SAMBatchedPredictor
from ctr_labeller.config.utils import parse_config, configure
from ctr_labeller.dataset import StereoDataSet
from ctr_labeller.datasaver import DataSaver
from ctr_labeller.types import StereoImageDataQueue

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
    save_image_and_masks: bool = True
    sort_based_on: str = "None"
    max_size_to_add: int = 40 # Depends on how much RAM on CPU to load images

class SAMBatchedPredictorThread(threading.Thread):
    def __init__(self, sam_predictor: SAMBatchedPredictor, stereo_image_queue, dataloader, config, left_input_prompts, right_input_prompts):
        super(SAMBatchedPredictorThread, self).__init__()
        self.sam_predictor = sam_predictor
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
        print("SAMBatchedPredictorThread | batch_len: ", batch_len)
        # is_last_batch = False
        batch_idx = 0
        for batch in self.dataloader:
            # if batch_idx >= batch_len - 1:
            #     is_last_batch = True
            print("SAMBatchedPredictorThread | predicting frame_ids: {} ".format(batch["frame_id"].cpu().detach().numpy()))
            stereo_image_datas = self.sam_predictor.predict_stereo(batch, self.left_input_prompts, self.right_input_prompts)
            self.stereo_image_queue.wait_add_images(stereo_image_datas)
            batch_idx += 1
            num = num + len(batch)
        print ("SAMBatchedPredictorThread | ------ BATCH IS DONE ------")

def main():
    # Parse config
    config = parse_config(CTRLabellerConfig, yaml_arg='--config')

    # Load data
    full_data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), config.data_path)
    datasaver = DataSaver(full_data_path, must_have_csv=True, save_image_and_masks= config.save_image_and_masks)
    stereo_image_dataset = StereoDataSet(full_data_path, datasaver)
    if len(stereo_image_dataset) == 0:
        print("Dataset has all been processed or empty, terminating!!!")
        return

    sam_predictor = SAMBatchedPredictor(datasaver, config.sort_based_on)

    # Input prompt generation
    if datasaver.get_input_prompts() is None:
        input_prompt_app = InputPromptGenerationApp(
                                stereo_image_dataset,
                                sam_predictor,
                                config.app_config.selection_image_height_py)
        left_input_prompts, right_input_prompts = input_prompt_app.generate_input_prompts()
    else:
        left_input_prompts, right_input_prompts = datasaver.get_input_prompts()

    # SAM Create Masks
    grid_num = config.app_config.selection_grid_size[0] * config.app_config.selection_grid_size[1]
    loader = torch.utils.data.DataLoader(stereo_image_dataset, batch_size=grid_num,
                                         pin_memory=True, num_workers=4, shuffle=False)
    stereo_image_queue = StereoImageDataQueue(max_size_to_add=config.max_size_to_add)
    predictor_thread = SAMBatchedPredictorThread(
        sam_predictor, stereo_image_queue,
        loader, config, left_input_prompts, right_input_prompts)

    # Labeller App
    app = CTRLabellerApp(config.app_config, stereo_image_dataset.datasaver, stereo_image_queue)
    app.start()
    app.title("CTR SAM Labeller, press [n] to save and proceed with next set")
    app.mainloop()
    predictor_thread.join()
    return

if __name__ == "__main__":
    main()
