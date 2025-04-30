import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import threading
import pathlib

from ctr_labeller.ui.apps import CTRLabellerApp, CTRLabellerAppConfig, InputPromptGenerationApp
from ctr_labeller.ui.controllers import ImageSelectorConfig, ImageSelector, StereoImageSelector, ImageSelectorState
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
    use_gui: bool = True
    create_input_prompts: bool = False
    input_prompt_image_height: int = 1080
    batch_num: int = -1
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
        self.exit_flag = False

        self.start()

    def run(self):
        num = 0
        batch_len = len(self.dataloader)
        print("SAMBatchedPredictorThread | batch_len: ", batch_len)
        # is_last_batch = False
        batch_idx = 0
        for batch in self.dataloader:
            if self.exit_flag:
                print ("SAMBatchedPredictorThread | Exiting thread")
                return
            # if batch_idx >= batch_len - 1:
            #     is_last_batch = True
            print("SAMBatchedPredictorThread | predicting frame_ids: {} ".format(batch["frame_id"].cpu().detach().numpy()))
            stereo_image_datas = self.sam_predictor.predict_stereo(batch, self.left_input_prompts, self.right_input_prompts)
            self.stereo_image_queue.wait_add_images(stereo_image_datas)
            batch_idx += 1
            num = num + len(batch)
            print("SAMBatchedPredictorThread | finished frame_ids: {} ".format(batch["frame_id"].cpu().detach().numpy()))

        print ("SAMBatchedPredictorThread | ------ BATCH IS DONE ------")

def main():
    # Parse config
    config = parse_config(CTRLabellerConfig, yaml_arg='--config')

    # Load data
    datasaver = DataSaver(config.data_path, save_image_and_masks=config.save_image_and_masks)
    stereo_image_dataset = StereoDataSet(config.data_path, datasaver, config.batch_num)
    if len(stereo_image_dataset) == 0:
        print("Dataset has all been processed or empty, terminating!!!")
        return

    sam_predictor = SAMBatchedPredictor(datasaver, config.sort_based_on)

    # Input prompt generation
    if datasaver.is_input_prompts_available() and not config.create_input_prompts:
        left_input_prompts, right_input_prompts = datasaver.get_input_prompts()
    else:
        input_prompt_app = InputPromptGenerationApp(
                                stereo_image_dataset,
                                sam_predictor,
                                config.input_prompt_image_height)
        input_prompt_app.start()
        input_prompt_app.mainloop()
        left_input_prompts, right_input_prompts = input_prompt_app.get_input_prompts()
        input_prompt_app.destroy()
    datasaver.save_input_prompts(left_input_prompts, right_input_prompts)
    # print("Left input prompts: ", left_input_prompts)
    # print("Right input prompts: ", right_input_prompts)

    # SAM Create Masks
    grid_num = config.app_config.selection_grid_size[0] * config.app_config.selection_grid_size[1]
    loader = torch.utils.data.DataLoader(stereo_image_dataset, batch_size=grid_num,
                                         pin_memory=True, num_workers=4, shuffle=False)
    stereo_image_queue = StereoImageDataQueue(max_size_to_add=config.max_size_to_add)
    predictor_thread = SAMBatchedPredictorThread(
        sam_predictor, stereo_image_queue,
        loader, config, left_input_prompts, right_input_prompts)

    try:
        if config.use_gui:
            # Labeller App
            app = CTRLabellerApp(config.app_config, stereo_image_dataset.datasaver, stereo_image_queue)
            app.start()
            app.mainloop()
        else:
            # on main thread, save masks as they are segmented
            left_state = ImageSelectorState(config.app_config.selection_image_height_py, config.app_config.zoom_factor)
            right_state = ImageSelectorState(config.app_config.selection_image_height_py, config.app_config.zoom_factor)
            selector = StereoImageSelector(
                ImageSelector(ImageSelectorConfig(), left_state),
                ImageSelector(ImageSelectorConfig(), right_state)
            )
            count = 0
            ref_size = len(datasaver.reference_dict)
            print("Num of images: ", ref_size)
            while True:
                # get next available mask on queue
                stereo_image_data = stereo_image_queue.wait_any_available_images_up_to(1)
                
                if not stereo_image_data:
                    continue
                stereo_image_data = stereo_image_data[0]
                if datasaver.check_is_mask_processed(stereo_image_data.frame_id):
                    continue
                selector.set_context(
                    stereo_image_data.frame_id,
                    stereo_image_data.collected_batch_num,
                    stereo_image_data.left,
                    stereo_image_data.right,
                    3)      # ImageSelectionType.MASK_AND_PROMPT
                selector.left_image_selector.state.current_image_data.is_save_mask = True
                selector.right_image_selector.state.current_image_data.is_save_mask = True
                # save mask
                print("Saving frame_id: ", selector.current_frame_id)
                datasaver.save_current_stereo_masks(selector.current_frame_id,
                                                    selector.current_collected_batch_num, 
                                                    selector.left_image_selector.state.current_image_data, 
                                                    selector.right_image_selector.state.current_image_data)
                
                # end program when all images are done
                num_processed = sum(pd.DataFrame.from_dict(datasaver.reference_dict, orient='index',columns=['is_processed'])["is_processed"])
                if num_processed  >= ref_size:
                    print("Finished all ", num_processed, " stereo masks.")
                    break

                if count >= 50:
                    datasaver.save_csv()
                    count = 0
                count += 1
    except KeyboardInterrupt:
        print("Keyboard input: ending program.")
    except:
        print("Error during processing - ending program.")
        
    predictor_thread.exit_flag = True
    predictor_thread.join()
    return

if __name__ == "__main__":
    main()
