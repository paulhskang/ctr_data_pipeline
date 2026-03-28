# Copyright (c) 2025, Wilfred and Joyce Posluns Centre for Guided Innovation and Therapeutic Intervention (PCIGITI), University of Toronto
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import pandas as pd
import torch
import threading
import argparse

from ctr_labeller.dataset import StereoDataSet
from ctr_labeller.datasaver import DataSaver
from ctr_labeller.predictor import SAMBatchedPredictor
from ctr_labeller.ui.apps import CTRLabellerApp, CTRLabellerAppConfig, InputPromptGenerationApp
from ctr_labeller.ui.controllers import ImageSelectorConfig, ImageSelector, StereoImageSelector, \
                                        ImageSelectorState, ImageSelectionType
from ctr_labeller.types import StereoImageDataQueue

class SAMBatchedPredictorThread(threading.Thread):
    def __init__(self, sam_predictor: SAMBatchedPredictor, stereo_image_queue, dataloader, 
                 config, left_input_prompts, right_input_prompts):
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
        batch_idx = 0
        for batch in self.dataloader:
            if self.exit_flag:
                print ("SAMBatchedPredictorThread | Exiting thread")
                return
            print("SAMBatchedPredictorThread | predicting frame_ids: {} ".format(batch["frame_id"].cpu().detach().numpy()))
            stereo_image_datas = self.sam_predictor.predict_stereo(batch, self.left_input_prompts, self.right_input_prompts)
            self.stereo_image_queue.wait_add_images(stereo_image_datas)
            batch_idx += 1
            num = num + len(batch)
            print("SAMBatchedPredictorThread | finished frame_ids: {} ".format(batch["frame_id"].cpu().detach().numpy()))

        print ("SAMBatchedPredictorThread | ------ BATCH IS DONE ------")

def arg_parser():
    ''' load CLI arguments '''
    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument("--data-path",                      help="Directory containing reference file", type=str)
    # optional arguments
    parser.add_argument("--use-gui",                        help="User actively specifies which masks to save with the GUI", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--input-prompt-json-name",         help="Name of input prompt file", type=str, default="")
    parser.add_argument("--input-prompt-app-image-height",  help="Input prompt app image height", type=int, default=1080)
    parser.add_argument("--batch-num",                      help="Process this batch number only", type=int, default=-1)
    parser.add_argument("--save-image-appended-with-masks", help="Save images with mask overlays", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--sort-based-on",                  help="Criteria on how masks are selected", type=str, default="None")
    parser.add_argument("--max-size-to-add",                help="Number of images to load at one time for processing", type=int, default=40)

    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    config = arg_parser()
    app_config = CTRLabellerAppConfig()

    datasaver = DataSaver(config.data_path, 
                          config.input_prompt_json_name,
                          save_image_appended_with_masks=config.save_image_appended_with_masks)
    stereo_image_dataset = StereoDataSet(config.data_path, datasaver, config.batch_num)
    if len(stereo_image_dataset) == 0:
        print("Main | Dataset has all been processed or empty, terminating!!!")
        return

    sam_predictor = SAMBatchedPredictor(datasaver, config.sort_based_on)

    # Input prompt generation
    if datasaver.is_input_prompts_available():
        left_input_prompts, right_input_prompts = datasaver.get_input_prompts()
        print("Main | Using input prompts from [{}]".format(config.input_prompt_json_name))
    else:
        input_prompt_app = InputPromptGenerationApp(
                                stereo_image_dataset,
                                sam_predictor,
                                config.input_prompt_app_image_height)
        input_prompt_app.start()
        input_prompt_app.mainloop()
        left_input_prompts, right_input_prompts = input_prompt_app.get_input_prompts()
        input_prompt_app.destroy()
        new_prompts_filepath = datasaver.save_input_prompts(left_input_prompts, right_input_prompts) # Only save when newly generated
        print("Main | Saving new input prompts to [{}]".format(new_prompts_filepath))

    # print("Left input prompts: ", left_input_prompts)
    # print("Right input prompts: ", right_input_prompts)

    # SAM Create Masks
    grid_num = app_config.selection_grid_size[0] * app_config.selection_grid_size[1]
    loader = torch.utils.data.DataLoader(stereo_image_dataset, batch_size=grid_num,
                                         pin_memory=True, num_workers=4, shuffle=False)
    stereo_image_queue = StereoImageDataQueue(max_size_to_add=config.max_size_to_add)
    predictor_thread = SAMBatchedPredictorThread(
        sam_predictor, stereo_image_queue,
        loader, config, left_input_prompts, right_input_prompts)

    try:
        if config.use_gui:
            # Labeller App
            app = CTRLabellerApp(app_config, stereo_image_dataset.datasaver, stereo_image_queue)
            app.start()
            app.mainloop()
        else:
            # on main thread, save masks as they are segmented
            left_state = ImageSelectorState(app_config.selection_image_height_py, app_config.zoom_factor)
            right_state = ImageSelectorState(app_config.selection_image_height_py, app_config.zoom_factor)
            selector = StereoImageSelector(
                ImageSelector(ImageSelectorConfig(), left_state),
                ImageSelector(ImageSelectorConfig(), right_state)
            )
            count = 0
            ref_size = len(datasaver.reference_dict)
            print("Main | Num of images: ", ref_size)
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
                    stereo_image_data.left,
                    stereo_image_data.right,
                    ImageSelectionType.MASK_AND_PROMPT)
                selector.left_image_selector.state.current_image_data.is_save_mask = True
                selector.right_image_selector.state.current_image_data.is_save_mask = True
                # save mask
                print("Main | Saving frame_id: ", selector.current_frame_id)
                datasaver.save_current_stereo_masks(selector.current_frame_id,
                                                    selector.left_image_selector.state.current_image_data, 
                                                    selector.right_image_selector.state.current_image_data)
                
                # end program when all images are done
                num_processed = sum(pd.DataFrame.from_dict(datasaver.reference_dict, 
                                                           orient='index',columns=['is_processed'])["is_processed"])
                if num_processed  >= ref_size:
                    print("Main | Finished all ", num_processed, " stereo masks.")
                    break

                if count >= 50:
                    datasaver.save_csv()
                    count = 0
                count += 1
    except KeyboardInterrupt:
        print("Main | Keyboard input: ending program.")
    except:
        print("Main | Error during processing - ending program.")
        
    predictor_thread.exit_flag = True
    predictor_thread.join()
    return

if __name__ == "__main__":
    main()
