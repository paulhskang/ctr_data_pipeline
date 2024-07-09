import copy
import cv2
from dataclasses import dataclass
from enum import Enum

import numpy as np
import tkinter as tk
from typing import Tuple
import threading
from PIL import ImageTk, Image

from ctr_labeller.config.utils import configure
from ctr_labeller.datasaver import DataSaver
from ctr_labeller.types import StereoImageDataQueue

def create_img_with_input_prompt(img, input_prompt):
    prompted_img = copy.deepcopy(img)
    if input_prompt["box"] is not None:
        box_prompt = input_prompt["box"]
        cv2.rectangle(prompted_img,
                        (box_prompt[0], box_prompt[1]), (box_prompt[2], box_prompt[3]),
                        color=(0, 255, 0), thickness=2)
    if input_prompt["point_coords"] is not None:
        point_coord = input_prompt["point_coords"][0]
        cv2.drawMarker(prompted_img,
                        (point_coord[0], point_coord[1]), color=(0, 255, 0), markerType=cv2.MARKER_DIAMOND,
                        markerSize=20, thickness=2, line_type=cv2.LINE_AA)
    return prompted_img

class MaskWidget():
    NO_MASK_NAME = "No Mask"
    def __init__(self, label_parent_frame, label_row, label_column,
                 button_frame, button_row, button_column,
                 draw_function, parent_state):
        self.mask_label = tk.Label(label_parent_frame, text=self.NO_MASK_NAME, bg="skyblue")
        self.mask_label.grid(row=label_row, column=label_column, ipadx=10, sticky="nsew")

        self.toggle_mask_button = tk.Button(button_frame, text ="Toggle Mask", command = self._toggle_mask,
                                       state="disabled", height=parent_state.c_draw_height_py//parent_state.c_scaler)
        self.toggle_mask_button.grid(row=button_row, column=button_column, sticky="nsew")
        self.draw_function = draw_function
        self.parent_state = parent_state

    def set_context(self, image_data):
        if len(image_data.prediction_outputs) < 1:
            return
        pred_output = image_data.prediction_outputs[image_data.current_mask_idx]
        image_to_draw = create_img_with_input_prompt(
            pred_output.masked_image,
            pred_output.input_prompt)
        self.toggle_mask_button.configure(state="active")
        self.mask_label.configure(text="Mask: {},\nscore: {:.5f}, area_ratio: {:.5f}".format(
            pred_output.input_prompt["name"],
            pred_output.score,
            pred_output.area_ratio))
        self.parent_state.current_image = self.draw_function(image_to_draw)

    def disable_context(self):
        self.mask_label.configure(text=self.NO_MASK_NAME)
        self.toggle_mask_button.configure(state="disabled")

    def _toggle_mask(self):
        self.parent_state.is_zoomed = False
        image_data = self.parent_state.current_image_data
        if len(image_data.prediction_outputs) < 1:
            return
        image_data.current_mask_idx = (image_data.current_mask_idx + 1) % len(image_data.prediction_outputs)
        pred_output = image_data.prediction_outputs[image_data.current_mask_idx]
        prompted_img = create_img_with_input_prompt(
            pred_output.masked_image,
            pred_output.input_prompt)
        self.mask_label.configure(text="Mask: {},\nscore: {:.5f}, area_ratio: {:.5f}".format(
            pred_output.input_prompt["name"],
            pred_output.score,
            pred_output.area_ratio))
        self.parent_state.current_image = self.draw_function(prompted_img)

class SaveWidget():
    def __init__(self, button_frame, row, column, parent_state):
        self.is_select_var = tk.BooleanVar(value=True) # Image is to be saved by default
        self.save_img_check_button = tk.Checkbutton(button_frame, \
            text = "Save Mask?", variable = self.is_select_var, \
            onvalue = True, offvalue = False, state="disabled", height=parent_state.c_draw_height_py//parent_state.c_scaler)
        self.save_img_check_button.grid(row=row, column=column, sticky="nsew")

    def set_context(self):
        self.is_select_var.set(True) # reset button
        self.save_img_check_button.configure(state="active")

    def disable_context(self):
        self.is_select_var.set(True)
        self.save_img_check_button.configure(state="disabled")

class ClickEventType(Enum):
    ZOOM = 1
    KEYPOINT_AND_BOUNDING_BOX_GENERATION = 2

@dataclass
class ImageSelectionConfig:
    mask_widget: bool
    save_widget: bool
    click_event_type: ClickEventType
    draw_height_py: int
    visualize_input_prompt: bool
    zoom_factor: float

class ImageSelectionState:
    def __init__(self, draw_height_py, visualize_input_prompt, zoom_factor):
        # constants
        self.c_visualize_input_prompt = visualize_input_prompt
        self.c_draw_height_py = draw_height_py
        self.c_zoom_factor = zoom_factor
        self.c_scaler = 150

        # variable state
        self.is_zoomed = False
        self.current_image_data = None
        self.current_image = None

class ImageSelection(tk.Frame):
    NO_IMAGE_NAME = "No Image"
    def __init__(self, root, config: ImageSelectionConfig):
        tk.Frame.__init__(self, master=root)

        self.state = ImageSelectionState(config.draw_height_py, config.visualize_input_prompt, config.zoom_factor)
        self.config = config

        # Canvas Frame
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<Button-1>", self.on_click_toggle_zoom)
    
        # Label Frame
        self.label_frame = tk.Frame(self)
        self.label_frame.grid(row=1, column=0, sticky="nsew")
        self.label_frame.grid_columnconfigure(0, weight=1)

        self.label = tk.Label(self.label_frame, text=self.NO_IMAGE_NAME, bg="skyblue")
        self.label.grid(row=0, column=0, ipadx=10, sticky="nsew")

        # Button Frame
        self.button_frame = tk.Frame(self)
        self.button_frame.grid(row=2, column=0, sticky="nsew")

        if config.mask_widget:
            self.mask_widget = MaskWidget(self.label_frame, 0, 1, self.button_frame, 0, 0, self.__draw_img_impl, self.state)

        if config.save_widget:
            self.save_widget = SaveWidget(self.button_frame, 0, 1, self.state)

        blank_image = np.zeros((self.state.c_draw_height_py , self.state.c_draw_height_py , 3), np.uint8)
        self.state.current_image  = self.__draw_img_impl(blank_image)
        
        # State data
        self.state.image_data = None
        self.state.is_zoomed = False
    
    def __resize_img_to_window(self, img):
        scale = img.shape[1]/self.state.c_draw_height_py
        return cv2.resize(img, (self.state.c_draw_height_py, int(img.shape[0]//scale)), interpolation=cv2.INTER_LINEAR)
        
    def __draw_img_impl(self, img):
        img = self.__resize_img_to_window(img)
        self.image_x = img.shape[0]
        self.image_y = img.shape[1]   
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.image = tk_img
        self.canvas.configure(width=self.image_y, height=self.image_x)
        self.canvas.create_image(10, 10, anchor=tk.NW, image=tk_img)
        return img
    
    def set_context(self, image_data):
        self.state.current_image_data = image_data
        self.label.configure(text="Image: {}".format(image_data.name))
        if self.mask_widget:
            self.mask_widget.set_context(image_data)
        else:
            image_to_draw = image_data.image
            self.state.current_image = self.__draw_img_impl(image_to_draw)

        if self.save_widget:
            self.save_widget.set_context()
        self.state.is_zoomed = False

    def disable_context(self, img_x_size, img_y_size):
        blank_image = np.zeros((img_x_size, img_y_size, 3), np.uint8)
        self.state.current_image = self.__draw_img_impl(blank_image)
        self.label.configure(text=self.NO_IMAGE_NAME)

        if self.mask_widget:
            self.mask_widget.disable_context()

        if self.save_widget:
            self.save_widget.disable_context()
        self.state.is_zoomed = False

    # def toggle_mask(self):
    #     self.is_zoomed = False
    #     if len(self.image_data.prediction_outputs) >= 1:
    #         self.image_data.current_mask_idx = (self.image_data.current_mask_idx + 1) % len(self.image_data.prediction_outputs)
    #         pred_output = self.image_data.prediction_outputs[self.image_data.current_mask_idx]
    #         prompted_img = create_img_with_input_prompt(
    #             pred_output.masked_image,
    #             pred_output.input_prompt)
    #         self.current_image = self.__draw_img_impl(prompted_img)
    #         self.mask_label.configure(text="Mask: {},\nscore: {:.5f}, area_ratio: {:.5f}".format(
    #             pred_output.input_prompt["name"],
    #             pred_output.score,
    #             pred_output.area_ratio))
        
    def __create_zoomed_image(self, img, px, py):        
        new_h = int(img.shape[0] // self.state.c_zoom_factor)
        new_w = int(img.shape[1] // self.state.c_zoom_factor)

        y = np.clip(py - new_h // 2, 0, img.shape[0])
        x = np.clip(px - new_w // 2, 0, img.shape[1])

        return img[y:y+new_h, x:x+new_w]
    
    def on_click_toggle_zoom(self, event):
        self.state.is_zoomed = not self.state.is_zoomed
        if self.state.is_zoomed:
            self.__draw_img_impl(self.__create_zoomed_image(self.state.current_image, event.x, event.y))
        else: # Not zoomed
            self.__draw_img_impl(self.state.current_image)

class StereoImageSelection(tk.Frame):
    def __init__(self, root, draw_height_py, 
                 visualize_input_prompt: bool = False, zoom_factor: float = 2.0):
        tk.Frame.__init__(self, master=root)

        self.image_frame = tk.Frame(self)
        self.image_frame.grid(row=0, column=0)

        # Image and label
        isc = ImageSelectionConfig(True, True, ClickEventType.ZOOM, draw_height_py=draw_height_py,
                             visualize_input_prompt=visualize_input_prompt, zoom_factor=zoom_factor)
        self.left_image_selection = ImageSelection(self.image_frame, isc)
        self.left_image_selection.grid(row=0, column=0)
        self.right_image_selection = ImageSelection(self.image_frame, isc)
        self.right_image_selection.grid(row=0, column=1)
        self.current_frame_id = -1
        self.is_active = False

    # In this app, images should all have the same size
    def set_context(self, frame_id, left_img_data, right_img_data):
        self.current_frame_id = frame_id
        self.left_image_selection.set_context(left_img_data)
        self.right_image_selection.set_context(right_img_data)
        self.is_active = True

    def disable_context(self, img_x_size, img_y_size):
        self.current_frame_id = -1
        self.left_image_selection.disable_context(img_x_size, img_y_size)
        self.right_image_selection.disable_context(img_x_size, img_y_size)      
        self.is_active = False

    def get_current_context(self):
        self.left_image_selection.state.current_image_data.is_save_mask = self.left_image_selection.save_widget.is_select_var.get()
        self.right_image_selection.state.current_image_data.is_save_mask = self.right_image_selection.save_widget.is_select_var.get()
        return self.current_frame_id, self.left_image_selection.state.current_image_data, self.right_image_selection.state.current_image_data

@configure
class CTRLabellerAppConfig:
    visualize_input_prompt: bool = False
    zoom_factor: float = 2.0
    selection_grid_size: Tuple[int, int] = (1, 1)
    selection_image_height_py: int = 1200
    frame_padx: int = 10
    frame_pady: int = 10

class CTRLabellerApp(tk.Tk):
    def __init__(self, config: CTRLabellerAppConfig, datasaver: DataSaver,
                 stereo_image_queue: StereoImageDataQueue, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.config = config
        self.selection_num = self.config.selection_grid_size[0] * self.config.selection_grid_size[1]
        self.stereo_image_queue = stereo_image_queue

        self.datasaver = datasaver
        self.selections = []
        for i in range(self.config.selection_grid_size[0]):
            for j in range(self.config.selection_grid_size[1]):
                selection = StereoImageSelection(self, self.config.selection_image_height_py,
                                                 self.config.visualize_input_prompt, self.config.zoom_factor)
                selection.grid(row=i, column=j, padx=self.config.frame_padx, pady=self.config.frame_pady)
                self.selections.append(selection)

        # State data
        self.img_idx = 0

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
        # This obtains max self.selection_num
        stereo_image_datas = self.stereo_image_queue.get_any_available_images_up_to(self.selection_num)

        selection_idx = 0
        for stereo_image_data in stereo_image_datas:
            if self.datasaver.check_is_mask_processed(stereo_image_data.frame_id):
                continue
            self.selections[selection_idx].set_context(
                stereo_image_data.frame_id,
                stereo_image_data.left,
                stereo_image_data.right)
            selection_idx += 1

        while selection_idx < self.selection_num:
            self.selections[selection_idx].disable_context(
                self.config.selection_image_height_py * 9 // 16, 
                self.config.selection_image_height_py)
            selection_idx += 1
        return

    def __save_selections(self):
        # print("CTRLabellerApp | Saving selections")
        for selection in self.selections:
            if not selection.is_active:
                continue
            frame_id, image_left, image_right = selection.get_current_context()
            self.datasaver.save_current_stereo_masks(frame_id, image_left, image_right)

    def __disable_all(self):
        for i in range(self.selection_num):
            self.selections[i].disable_context(self.config.selection_image_height_py * 9 // 16, 
                                               self.config.selection_image_height_py)

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

    # def generate_masks():
    #     left_image_data = ImageData(batch_data["left_image"][i].cpu().detach().numpy(),
    #                                 batch_data["left_image_name"][i],
    #                                 batch_data["left_image_path"][i])
    #     self.predict_one(left_image_data, left_input_prompts)

    #     pass
