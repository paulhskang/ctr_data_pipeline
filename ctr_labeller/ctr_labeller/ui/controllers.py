from dataclasses import dataclass
from enum import Enum
import numpy as np
import tkinter as tk
from typing import Union

from ctr_labeller.types import ImageData
from ctr_labeller.ui.types import ImageSelectorState
from ctr_labeller.ui.presenters import StereoImagePresenter, create_img_with_input_prompt

class SaveWidget():
    def __init__(self, save_img_check_button, is_select_var, state: ImageSelectorState):
        self.state = state
        self.state.is_select_var = is_select_var
        self.state.save_img_check_button = save_img_check_button
        self.disable()
    def enable(self):
        self.state.is_select_var.set(True) # reset button
        self.state.save_img_check_button.configure(state="active")
    def disable(self):
        self.state.is_select_var.set(True)
        self.state.save_img_check_button.configure(state="disabled")

class MaskTogglerWidget():
    # def __init__(self, toggle_mask_button, select_image_function, parent_state: ImageSelectorState):
    def __init__(self, toggle_mask_button, select_image_function, state: ImageSelectorState):
        self.state = state
        self.state.toggle_mask_button = toggle_mask_button
        self.state.toggle_mask_button.configure(command=self.__toggle_mask)
        self.select_image_function = select_image_function
        self.disable()
    def enable(self):
        self.state.toggle_mask_button.configure(state="active")
    def disable(self):
        self.state.toggle_mask_button.configure(state="disabled")
    def __toggle_mask(self):
        image_data = self.state.current_image_data
        if len(image_data.prediction_outputs) < 1:
            return
        image_data.current_mask_idx = (image_data.current_mask_idx + 1) % len(image_data.prediction_outputs)
        self.select_image_function(ImageSelectionType.MASK_AND_PROMPT)
        self.state.is_zoomed = False
        self.state.trigger_presenter_function()
    
class ClickZoomWidget():
    def __init__(self, state: ImageSelectorState):
        self.state = state
        self.state.zoom_function = self.__create_zoomed_image
        self.state.canvas.bind("<Button-1>", self.__on_click_toggle_zoom)
        self.chosen_y = 0
        self.chosen_x = 0
    def enable(self):
        pass
    def disable(self):
        pass
    def __create_zoomed_image(self, img):        
        new_h = int(img.shape[0] // self.state.c_zoom_factor)
        new_w = int(img.shape[1] // self.state.c_zoom_factor)
        y = np.clip(self.chosen_y - new_h // 2, 0, img.shape[0])
        x = np.clip(self.chosen_x - new_w // 2, 0, img.shape[1])
        return img[y:y+new_h, x:x+new_w]
    def __on_click_toggle_zoom(self, event):
        self.state.is_zoomed = not self.state.is_zoomed
        self.chosen_y = int(event.y * self.state.resize_img_scale)
        self.chosen_x = int(event.x * self.state.resize_img_scale)
        self.state.trigger_presenter_function()

class ClickEventType(Enum):
    ZOOM = 1
    KEYPOINT = 2
    BOUNDING_BOX = 3
    
class ImageSelectionType(Enum):
    BLANK = 0
    IMAGE = 1
    # CURRENT_IMAGE = 2
    MASK_AND_PROMPT = 3
    
@dataclass
class ImageSelectorConfig:
    is_select_var = None
    save_img_check_button = None
    toggle_mask_button = None
    click_event_type: ClickEventType = None

class ImageSelector:
    def __init__(self, config: ImageSelectorConfig, state: ImageSelectorState):
        self.state = state
        # self.config = config
        self.widgets = []
        if config.save_img_check_button is not None:
            self.widgets.append(SaveWidget(config.save_img_check_button, config.is_select_var, self.state))
        if config.toggle_mask_button is not None:
            self.widgets.append(MaskTogglerWidget(config.toggle_mask_button, self.select_image, self.state))
        if config.click_event_type is not None:
            if config.click_event_type == ClickEventType.ZOOM:
                self.widgets.append(ClickZoomWidget(self.state))
            else:
                raise Exception("Other clickevent types not supported yet")
        self.disable_context()

    def select_image(self, selection: ImageSelectionType):
        if selection == ImageSelectionType.BLANK:
            blank_image = np.zeros((
                self.state.c_draw_height_py * 9//16,
                self.state.c_draw_height_py, 3), np.uint8)
            self.state.current_image = blank_image
            self.state.current_image_label = None
            self.state.current_mask_label = None
            return
        elif selection == ImageSelectionType.IMAGE:
            self.state.current_image = self.state.current_image_data.image
            self.state.current_image_label = "Image: {}".format(self.state.current_image_data.name)
            self.state.current_mask_label_mask_label = None
            return
        # else selection == ImageSelectionType.MASK_AND_PROMPT:
        image_data = self.state.current_image_data
        if len(image_data.prediction_outputs) < 1:
            raise Exception("selection is Mask, but no mask outputs")
        pred_output = image_data.prediction_outputs[image_data.current_mask_idx]
        self.state.current_image = create_img_with_input_prompt(
            pred_output.masked_image,
            pred_output.input_prompt)
        self.state.current_image_label = "Image: {}".format(image_data.name)
        self.state.current_mask_label = "Mask: {},\nscore: {:.5f}, area_ratio: {:.5f}".format(
            pred_output.input_prompt["name"],
            pred_output.score,
            pred_output.area_ratio)
        return

    def set_context(self, image_data: ImageData, selection: ImageSelectionType):
        self.state.current_image_data = image_data
        self.select_image(selection=selection)
        for widget in self.widgets:
            widget.enable()
    
    def disable_context(self):
        for widget in self.widgets:
            widget.disable()
        self.select_image(ImageSelectionType.BLANK)
        self.state.current_image_data = None

class StereoImageSelector:
    def __init__(self, left_image_selector: ImageSelector, right_image_selector: ImageSelector):
        self.left_image_selector = left_image_selector
        self.right_image_selector = right_image_selector
        self.current_frame_id = -1
        self.is_active = False

    def set_context(self, frame_id, left_img_data, right_img_data):
        self.current_frame_id = frame_id
        self.left_image_selector.set_context(left_img_data, ImageSelectionType.MASK_AND_PROMPT)
        self.right_image_selector.set_context(right_img_data, ImageSelectionType.MASK_AND_PROMPT)
        self.is_active = True

    def disable_context(self):
        self.current_frame_id = -1
        self.left_image_selector.disable_context()
        self.right_image_selector.disable_context()
        self.is_active = False
