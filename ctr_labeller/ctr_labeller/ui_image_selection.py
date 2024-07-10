import copy
import cv2
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

from dataclasses import dataclass
from enum import Enum
from ctr_labeller.types import ImageData

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

class ImageSelectionState:
    def __init__(self, draw_height_py, visualize_input_prompt, zoom_factor):
        # constants
        self.c_visualize_input_prompt = visualize_input_prompt
        self.c_draw_height_py = draw_height_py
        self.c_zoom_factor = zoom_factor
        self.c_scaler = 150

        # variable state
        self.is_zoomed = False
        self.current_image_data: ImageData = None
        self.current_image = None
        
class MaskWidget():
    NO_MASK_NAME = "No Mask"
    def __init__(self, label_parent_frame, label_row, label_column,
                 button_frame, button_row, button_column,
                 draw_function, parent_state: ImageSelectionState):
        self.mask_label = tk.Label(label_parent_frame, text=self.NO_MASK_NAME, bg="skyblue")
        self.mask_label.grid(row=label_row, column=label_column, ipadx=10, sticky="nsew")

        self.toggle_mask_button = tk.Button(button_frame, text ="Toggle Mask", command = self.__toggle_mask,
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

    def __toggle_mask(self):
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
    def __init__(self, button_frame, row, column, parent_state: ImageSelectionState):
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

class ClickZoomWidget():
    def __init__(self, canvas, draw_function, parent_state: ImageSelectionState):
        self.draw_function = draw_function
        self.parent_state = parent_state
        canvas.bind("<Button-1>", self.__on_click_toggle_zoom)

    def __create_zoomed_image(self, img, px, py):        
        new_h = int(img.shape[0] // self.parent_state.c_zoom_factor)
        new_w = int(img.shape[1] // self.parent_state.c_zoom_factor)

        y = np.clip(py - new_h // 2, 0, img.shape[0])
        x = np.clip(px - new_w // 2, 0, img.shape[1])

        return img[y:y+new_h, x:x+new_w]
    
    def __on_click_toggle_zoom(self, event):
        self.parent_state.is_zoomed = not self.parent_state.is_zoomed
        if self.parent_state.is_zoomed:
            self.draw_function(self.__create_zoomed_image(self.parent_state.current_image, event.x, event.y))
        else: # Not zoomed
            self.draw_function(self.parent_state.current_image)

class ClickMaskWidget():
    def __init__(self, canvas, button_frame, button_row, button_column, parent_state):
        self.keypoint = np.array([0, 0])
        self.bounding_box = np.array([0, 0, 0, 0])
        self.toggle_set_canvas_button = tk.Button(button_frame, text ="Set Keypoint", command = self.__toggle_set_canvas,
                state="disabled", height=parent_state.c_draw_height_py//parent_state.c_scaler)
        self.toggle_set_canvas_button.grid(row=button_row, column=button_column, sticky="nsew")

        self.canvas = canvas
        self.is_keypoint_not_bounding_box = True
        self.canvas.bind("<Button-1>", self.__set_keypoint)
        self.canvas.bind("<Button-2>", None)

    def __toggle_set_canvas(self):
        self.is_keypoint_not_bounding_box = not self.is_keypoint_not_bounding_box
        if self.is_keypoint_not_bounding_box:
            self.canvas.bind("<Button-1>", self.__set_keypoint)
            self.canvas.bind("<Button-2>", None) 
        else: # bounding box
            self.canvas.bind("<Button-1>", self.__set_start_bounding_box)
            self.canvas.bind("<Button-2>", self.__set_end_bounding_box) 
        
    def __set_keypoint(self, px, py):
        self.keypoint = np.array([px, py])
        
    def __set_start_bounding_box(self, px, py):
        self.bounding_box[0] = px
        self.bounding_box[1] = py
        
    def __set_end_bounding_box(self, px, py):
        self.bounding_box[2] = px
        self.bounding_box[3] = py
    
class ClickKeypointMaskWidget():
    def __init__(self, canvas):
        canvas.bind("<Button-1>", self.__set_start_bounding_box)
        canvas.bind("<Button-2>", self.__set_end_bounding_box)
        
class GenerateMaskWidget():
    def __init__(self, frame, row, col, predictor, parent_state: ImageSelectionState):
        self.generate_mask_button = tk.Button(frame, text ="Generate Mask", command = self.__generate_mask,
                        state="disabled", height=parent_state.c_draw_height_py//parent_state.c_scaler)

class ClickEventType(Enum):
    NONE = -1
    ZOOM = 1
    KEYPOINT = 2
    BOUNDING_BOX = 3

@dataclass
class ImageSelectionConfig:
    mask_widget: bool
    save_widget: bool
    click_event_type: ClickEventType
    draw_height_py: int
    visualize_input_prompt: bool
    zoom_factor: float
        
class ImageSelection(tk.Frame):
    NO_IMAGE_NAME = "No Image"
    def __init__(self, root, config: ImageSelectionConfig):
        tk.Frame.__init__(self, master=root)

        self.state = ImageSelectionState(config.draw_height_py, config.visualize_input_prompt, config.zoom_factor)
        self.config = config

        # Canvas Frame
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0)
    
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

        if config.click_event_type == ClickEventType.ZOOM:
            self.click_widget = ClickZoomWidget(self.canvas, self.__draw_img_impl, self.state)
        
        # Start state
        blank_image = np.zeros((self.state.c_draw_height_py , self.state.c_draw_height_py , 3), np.uint8)
        self.state.current_image  = self.__draw_img_impl(blank_image)
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
