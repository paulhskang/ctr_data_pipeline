import cv2
import copy
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ctr_labeller.types import ImageData
from ctr_labeller.ui.types import ImageSelectorState, create_img_with_input_prompts
from ctr_labeller.predictor import SAMBatchedPredictor

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
    def __init__(self, toggle_mask_button, select_image_function, state: ImageSelectorState):
        self.state = state
        self.state.toggle_mask_button = toggle_mask_button
        self.state.toggle_mask_button.configure(command=self._toggle_mask)
        self.select_image_function = select_image_function
        self.toggle_state = False
        self.disable()
    def enable(self):
        self.state.toggle_mask_button.configure(state="active")
    def disable(self):
        self.state.toggle_mask_button.configure(state="disabled")
    def _toggle_mask(self):
        if self.toggle_state:
            self.select_image_function(ImageSelectionType.IMAGE_AND_PROMPT)
            self.toggle_state = False
        else:
            image_data = self.state.current_image_data
            if len(image_data.prediction_outputs) < 1:
                return
            image_data.current_mask_idx = (image_data.current_mask_idx + 1) % len(image_data.prediction_outputs)

            self.select_image_function(ImageSelectionType.MASK_AND_PROMPT)
            self.toggle_state = True
        self.state.is_zoomed = False
        self.state.trigger_presenter_function()

class GenerateMaskTogglerWidget():
    def __init__(self, predictor: SAMBatchedPredictor, generate_mask_button, state: ImageSelectorState):
        self.predictor = predictor
        self.state = state
        self.state.generate_mask_button = generate_mask_button
        self.state.generate_mask_button.configure(command=self._generate_mask)
        self.disable()
    def enable(self):
        self.state.generate_mask_button.configure(state="active")
    def disable(self):
        self.state.generate_mask_button.configure(state="disabled")
    def _generate_mask(self):
        image_data = self.state.current_image_data
        image_data.prediction_outputs = []
        self.predictor.predict_one(image_data, self.state.current_input_prompts)

class ClickInputPromptWidget():
    def __init__(self, toggle_type_button, state: ImageSelectorState):
        self.state = state
        self.bounding_box = np.array([0, 0, 0, 0])
        self.is_start_bounding_box_set = False
        self.is_end_bounding_box_set = False
        self.is_bounding_box_set = False
        self.toggle_type_button = toggle_type_button
        self.toggle_type_button.configure(command=self.__toggle_set_canvas)
        self.toggle_type_button.configure(text="Setting Keypoint")
        self.is_keypoint_not_bounding_box = True
        self.state.canvas.bind("<Button-1>", self.__set_keypoint)
        self.state.canvas.bind("<Button-2>", None)
    def enable(self):
        self.toggle_type_button.configure(state="active")
    def disable(self):
        self.toggle_type_button.configure(state="disabled")

    def __toggle_set_canvas(self):
        self.is_keypoint_not_bounding_box = not self.is_keypoint_not_bounding_box
        if self.is_keypoint_not_bounding_box:
            self.toggle_type_button.configure(text="Setting Keypoint")
            self.state.canvas.bind("<Button-1>", self.__set_keypoint)
            self.state.canvas.bind("<Button-3>", None)
            return
        # else: # bounding box
        self.toggle_type_button.configure(text="Setting Bounding Box")
        self.state.canvas.bind("<Button-1>", self.__set_start_bounding_box)
        self.state.canvas.bind("<Button-3>", self.__set_end_bounding_box) 
    
    def __set_keypoint(self, event):
        self.state.current_image_data.prediction_outputs = []
        self.state.current_mask_label = None

        curr_keypoint = [int(event.x * self.state.resize_img_scale), int(event.y * self.state.resize_img_scale)]
        self.state.add_keypoint(curr_keypoint)
        self.state.current_image = create_img_with_input_prompts(self.state.current_image_data.image,\
                                                self.state.current_input_prompts)
        self.state.trigger_presenter_function()  
    
    def __set_start_bounding_box(self, event):
        self.state.current_image_data.prediction_outputs = []
        self.state.current_mask_label = None

        self.bounding_box[0] = int(event.x * self.state.resize_img_scale)
        self.bounding_box[1] = int(event.y * self.state.resize_img_scale)
        self.is_start_bounding_box_set = True
        # self.__check_for_input_prompts_and_generate()
        self.is_bounding_box_set = self.is_start_bounding_box_set and self.is_end_bounding_box_set
        if self.is_bounding_box_set:
            self.state.add_bounding_box(self.bounding_box)
            self.state.current_image = create_img_with_input_prompts(self.state.current_image_data.image,\
                                                    self.state.current_input_prompts)
        self.state.trigger_presenter_function()
    
    def __set_end_bounding_box(self, event):
        self.state.current_image_data.prediction_outputs = []
        self.state.current_mask_label = None

        self.bounding_box[2] = int(event.x * self.state.resize_img_scale)
        self.bounding_box[3] = int(event.y * self.state.resize_img_scale)
        self.is_end_bounding_box_set = True
        # self.__check_for_input_prompts_and_generate()
        self.is_bounding_box_set = self.is_start_bounding_box_set and self.is_end_bounding_box_set
        if self.is_bounding_box_set:
            self.state.add_bounding_box(self.bounding_box)
            self.state.current_image = create_img_with_input_prompts(self.state.current_image_data.image,\
                                                    self.state.current_input_prompts)
        self.state.trigger_presenter_function()    

class DeleteLastKeypointWidget():
    def __init__(self, delete_last_keypoint_button, state: ImageSelectorState):
        self.state = state
        self.state.delete_last_keypoint_button = delete_last_keypoint_button
        self.state.delete_last_keypoint_button.configure(command=self._delete_last_keypoint)
        self.disable()
    def enable(self):
        self.state.delete_last_keypoint_button.configure(state="active")
    def disable(self):
        self.state.delete_last_keypoint_button.configure(state="disabled")
    def _delete_last_keypoint(self):
        self.state.current_image_data.prediction_outputs = []
        self.state.current_mask_label = None

        self.state.remove_keypoint()
        self.state.current_image = create_img_with_input_prompts(self.state.current_image_data.image,\
                                                self.state.current_input_prompts)
        self.state.trigger_presenter_function()

class ClearBoundingBoxWidget():
    def __init__(self, clear_bounding_box_button, state: ImageSelectorState):
        self.state = state
        self.state.clear_bounding_box_button = clear_bounding_box_button
        self.state.clear_bounding_box_button.configure(command=self._clear_bounding_box)
        self.disable()
    def enable(self):
        self.state.clear_bounding_box_button.configure(state="active")
    def disable(self):
        self.state.clear_bounding_box_button.configure(state="disabled")
    def _clear_bounding_box(self):
        self.state.current_image_data.prediction_outputs = []
        self.state.current_mask_label = None

        self.state.remove_bounding_box()
        self.state.current_image = create_img_with_input_prompts(self.state.current_image_data.image,\
                                                self.state.current_input_prompts)
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
    INPUT_PROMPT = 2
    
class ImageSelectionType(Enum):
    BLANK = 0
    IMAGE = 1
    IMAGE_AND_PROMPT = 2
    MASK_AND_PROMPT = 3
    
@dataclass
class ImageSelectorConfig:
    is_select_var = None
    save_img_check_button = None # Must be with is_select_var
    toggle_mask_button = None
    generate_mask_button = None
    delete_last_keypoint_button = None
    clear_bounding_box_button = None
    predictor: SAMBatchedPredictor = None # For generate_masks_button
    click_event_type: ClickEventType = None
    toggle_type_button = None # For ClickEventType.INPUT_PROMPT

class ImageSelector:
    def __init__(self, config: ImageSelectorConfig, state: ImageSelectorState):
        self.state = state
        # self.config = config
        self.widgets = []
        if config.save_img_check_button is not None:
            self.widgets.append(SaveWidget(config.save_img_check_button, config.is_select_var, self.state))
        if config.toggle_mask_button is not None:
                self.widgets.append(MaskTogglerWidget(config.toggle_mask_button, self.select_image, self.state))
        if config.generate_mask_button is not None and isinstance(config.predictor, SAMBatchedPredictor):
            self.widgets.append(GenerateMaskTogglerWidget(config.predictor, config.generate_mask_button, self.state))
        if config.click_event_type is not None:
            if config.click_event_type == ClickEventType.ZOOM:
                self.widgets.append(ClickZoomWidget(self.state))
            elif config.click_event_type == ClickEventType.INPUT_PROMPT:
                self.widgets.append(ClickInputPromptWidget(config.toggle_type_button, self.state))
                if config.delete_last_keypoint_button is not None:
                    self.widgets.append(DeleteLastKeypointWidget(config.delete_last_keypoint_button, self.state))
                if config.clear_bounding_box_button is not None:
                    self.widgets.append(ClearBoundingBoxWidget(config.clear_bounding_box_button, self.state))
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
            self.state.current_image = copy.deepcopy(self.state.current_image_data.image)
            self.state.current_image_label = "Image: {}".format(self.state.current_image_data.name)
            self.state.current_mask_label = None
            return
        elif selection == ImageSelectionType.IMAGE_AND_PROMPT:
            print(self.state.current_input_prompts)
            self.state.current_image_label = "Image: {}".format(self.state.current_image_data.name)
            self.state.current_mask_label = None
            self.state.current_image = create_img_with_input_prompts(
                self.state.current_image_data.image,
                self.state.current_input_prompts)
            return
        # else selection == ImageSelectionType.MASK_AND_PROMPT:
        image_data = self.state.current_image_data
        if len(image_data.prediction_outputs) < 1:
            raise Exception("selection is Mask, but no mask outputs")
        pred_output = image_data.prediction_outputs[image_data.current_mask_idx]
        self.state.current_image = create_img_with_input_prompts(
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
        self.current_collected_batch_num = -1
        self.is_active = False
        self.disable_context()

    def set_context(self, frame_id, collected_batch_num, left_img_data, right_img_data, selection: ImageSelectionType):
        self.current_frame_id = frame_id
        self.current_collected_batch_num = collected_batch_num
        self.left_image_selector.set_context(left_img_data, selection)
        self.right_image_selector.set_context(right_img_data, selection)
        self.is_active = True

    def disable_context(self):
        self.current_frame_id = -1
        self.current_collected_batch_num = -1
        self.left_image_selector.disable_context()
        self.right_image_selector.disable_context()
        self.is_active = False
