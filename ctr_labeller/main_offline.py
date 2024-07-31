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

def create_selector(state: ImageSelectorState):
    isc = ImageSelectorConfig()
    # isc.click_event_type = ClickEventType.ZOOM
    # isc.save_img_check_button, isc.is_select_var = button_generator.create_check_button(text = "Save Mask?", \
    #     onvalue = True, offvalue = False, state="disabled", height=state.c_draw_height_py//state.c_scaler)
    # isc.toggle_mask_button = button_generator.create_button(text ="Toggle Mask",
    #                                 state="disabled", height=state.c_draw_height_py//state.c_scaler)
    return ImageSelector(isc, state) 

def present_next(self, selectors):
    # This obtains max self.selector_num
    stereo_image_datas = self.stereo_image_queue.get_any_available_images_up_to(self.selector_num)

    selection_idx = 0
    for stereo_image_data in stereo_image_datas:
        if self.datasaver.check_is_mask_processed(stereo_image_data.frame_id):
            continue
        selectors[selection_idx].set_context(
            stereo_image_data.frame_id,
            stereo_image_data.collected_batch_num,
            stereo_image_data.left,
            stereo_image_data.right,
            3)      # ImageSelectionType.MASK_AND_PROMPT
        selection_idx += 1

    while selection_idx < self.selector_num:
        selectors[selection_idx].disable_context()
        # self.selectors[selection_idx].disable_context(
        #     self.config.selection_image_height_py * 9 // 16, 
        #     self.config.selection_image_height_py)
        self.presenters[selection_idx].present_current_state()
        selection_idx += 1
    return

def save_selections(self, selectors):
    # print("CTRLabellerApp | Saving selections")
    for selector in selectors:
        if not selector.is_active:
            continue
        frame_id, collected_batch_num, image_left, image_right = CTRLabellerApp.get_selector_current_context(selector)
        self.datasaver.save_current_stereo_masks(frame_id, collected_batch_num, image_left, image_right)

@configure
class CTRLabellerConfig:
    data_path: str
    app_config: CTRLabellerAppConfig
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
            print("SAMBatchedPredictorThread | finished frame_ids: {} ".format(batch["frame_id"].cpu().detach().numpy()))

        print ("SAMBatchedPredictorThread | ------ BATCH IS DONE ------")

def main():
    # Parse config
    config = parse_config(CTRLabellerConfig, yaml_arg='--config')

    # Load data
    datasaver = DataSaver(config.data_path, must_have_csv=True, save_image_and_masks= config.save_image_and_masks)
    stereo_image_dataset = StereoDataSet(config.data_path, datasaver, config.batch_num)
    if len(stereo_image_dataset) == 0:
        print("Dataset has all been processed or empty, terminating!!!")
        return

    sam_predictor = SAMBatchedPredictor(datasaver, config.sort_based_on)

    # Input prompt generation
    if datasaver.is_input_prompts_available:
        left_input_prompts, right_input_prompts = datasaver.get_input_prompts()
    else:
        input_prompt_app = InputPromptGenerationApp(
                                stereo_image_dataset,
                                sam_predictor,
                                config.input_prompt_image_height)
        input_prompt_app.start()
        input_prompt_app.mainloop()
        left_input_prompts, right_input_prompts = input_prompt_app.get_input_prompts()
        print("Selected input prompts", left_input_prompts, right_input_prompts)
        input_prompt_app.destroy()

    # SAM Create Masks
    grid_num = config.app_config.selection_grid_size[0] * config.app_config.selection_grid_size[1]
    loader = torch.utils.data.DataLoader(stereo_image_dataset, batch_size=grid_num,
                                         pin_memory=True, num_workers=4, shuffle=False)
    stereo_image_queue = StereoImageDataQueue(max_size_to_add=config.max_size_to_add)
    predictor_thread = SAMBatchedPredictorThread(
        sam_predictor, stereo_image_queue,
        loader, config, left_input_prompts, right_input_prompts)

    # on main thread, save masks as they are segmented
    left_state = ImageSelectorState(config.app_config.selection_image_height_py, config.app_config.zoom_factor)
    right_state = ImageSelectorState(config.app_config.selection_image_height_py, config.app_config.zoom_factor)
    selector = StereoImageSelector(
        create_selector(left_state),
        create_selector(right_state)
    )



    ref_size = len(datasaver.reference_dict)
    while True:
        # get next available mask on queue
        stereo_image_data = stereo_image_queue.get_any_available_images_up_to(1)
        

        if not stereo_image_data:
            continue
        # print(len(stereo_image_data))
        stereo_image_data = stereo_image_data[0]
        # print(stereo_image_data)
        # print(len(stereo_image_data))
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
        # frame_id, collected_batch_num, image_left, image_right = CTRLabellerApp.get_selector_current_context(selector)
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

    predictor_thread.join()
    print("Completed offline masking.")
    return

if __name__ == "__main__":
    main()
