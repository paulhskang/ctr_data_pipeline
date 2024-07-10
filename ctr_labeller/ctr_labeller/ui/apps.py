from typing import Tuple
import tkinter as tk

from ctr_labeller.config.utils import configure
from ctr_labeller.datasaver import DataSaver
from ctr_labeller.types import StereoImageDataQueue
from ctr_labeller.ui.presenters import StereoImagePresenter, OrganizedButtonGenerator
from ctr_labeller.ui.controllers import StereoImageSelector, ImageSelectorConfig, ImageSelector, ClickEventType
from ctr_labeller.ui.types import ImageSelectorState

@configure
class CTRLabellerAppConfig:
    zoom_factor: float = 2.0
    selection_image_height_py: int = 1200
    selection_grid_size: Tuple[int, int] = (1, 1)
    frame_padx: int = 10
    frame_pady: int = 10

class CTRLabellerApp(tk.Tk):
    def __init__(self, config: CTRLabellerAppConfig, datasaver: DataSaver,
                 stereo_image_queue: StereoImageDataQueue, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.config = config
        self.selector_num = self.config.selection_grid_size[0] * self.config.selection_grid_size[1]
        self.stereo_image_queue = stereo_image_queue
        self.datasaver = datasaver

        self.presenters = []
        self.selectors = []
        for i in range(self.config.selection_grid_size[0]):
            for j in range(self.config.selection_grid_size[1]):
                left_state = ImageSelectorState(config.selection_image_height_py, config.zoom_factor)
                right_state = ImageSelectorState(config.selection_image_height_py, config.zoom_factor)
                stereo_img_presenter = StereoImagePresenter(self, left_state, right_state)
                stereo_img_presenter.grid(row=i, column=j, padx=self.config.frame_padx, pady=self.config.frame_pady)
                stereo_img_selector = StereoImageSelector(
                    CTRLabellerApp.__create_selector(OrganizedButtonGenerator(stereo_img_presenter.left, (2, 0)), left_state),
                    CTRLabellerApp.__create_selector(OrganizedButtonGenerator(stereo_img_presenter.right, (2, 0)), right_state))
                self.selectors.append(stereo_img_selector)
                self.presenters.append(stereo_img_presenter)

        # State data
        self.img_idx = 0

    def __create_selector(button_generator: OrganizedButtonGenerator, state: ImageSelectorState):
        isc = ImageSelectorConfig()
        isc.click_event_type = ClickEventType.ZOOM
        isc.save_img_check_button, isc.is_select_var = button_generator.create_check_button(text = "Save Mask?", \
            onvalue = True, offvalue = False, state="disabled", height=state.c_draw_height_py//state.c_scaler)
        isc.toggle_mask_button = button_generator.create_button(text ="Toggle Mask",
                                       state="disabled", height=state.c_draw_height_py//state.c_scaler)
        return ImageSelector(isc, state)
    
    def start(self):
        self.__disable_all()
        self.bind("n", self.keypress_event)

    # def set_stereo_image_datas(self, stereo_image_datas):
    #     self.img_idx = 0
    #     self.stereo_image_datas = stereo_image_datas
        # self.is_done = self.__present_next()
            
    # Return value
    # False: There is more iterations
    # True: The iterations are done
    def __present_next(self):
        # This obtains max self.selector_num
        stereo_image_datas = self.stereo_image_queue.get_any_available_images_up_to(self.selector_num)

        selection_idx = 0
        for stereo_image_data in stereo_image_datas:
            if self.datasaver.check_is_mask_processed(stereo_image_data.frame_id):
                continue
            self.selectors[selection_idx].set_context(
                stereo_image_data.frame_id,
                stereo_image_data.left,
                stereo_image_data.right)
            self.presenters[selection_idx].present_current_state()
            selection_idx += 1

        while selection_idx < self.selector_num:
            self.selectors[selection_idx].disable_context(
                self.config.selection_image_height_py * 9 // 16, 
                self.config.selection_image_height_py)
            self.presenters[selection_idx].present_current_state()
            selection_idx += 1
        return

    def __save_selections(self):
        # print("CTRLabellerApp | Saving selections")
        for selector in self.selectors:
            if not selector.is_active:
                continue
            frame_id, image_left, image_right = CTRLabellerApp.get_selector_current_context(selector)
            self.datasaver.save_current_stereo_masks(frame_id, image_left, image_right)

    def get_selector_current_context(selector):
        selector.left_image_selector.state.current_image_data.is_save_mask = selector.left_image_selector.state.is_select_var.get()
        selector.right_image_selector.state.current_image_data.is_save_mask = selector.right_image_selector.state.is_select_var.get()
        return selector.current_frame_id, selector.left_image_selector.state.current_image_data, selector.right_image_selector.state.current_image_data
    
    def __disable_all(self):
        for i in range(self.selector_num):
            self.selectors[i].disable_context()
            self.presenters[i].present_current_state()

    def keypress_event(self, input):
        # print(type(input)) # What is tkinter giving here?
        self.__save_selections() # Save the currently presented from __present_next()
        self.__disable_all()

        self.update()
        self.__present_next()
        print("CTRLabellerApp | Presenting next set of images")

class InputPromptGenerationApp(tk.Tk):
    def __init__(self, selection_image_height_py, sam_predictor, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.selection_image_height_py = selection_image_height_py
        self.sam_predictor = sam_predictor
        self.left_input_prompts = None
        self.right_input_prompts = None
