import cv2
import copy
import tkinter as tk
from typing import List
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

class ImageSelectorState:
    def __init__(self, draw_height_py, zoom_factor = 2):
        # constants
        self.c_draw_height_py = draw_height_py
        self.c_zoom_factor = zoom_factor
        self.c_scaler = 150

        # set from presenter
        self.trigger_presenter_function = None
        
        self.canvas = None
        self.is_zoomed = False
        self.zoom_function = None

        self.resize_img_scale = None
        self.current_input_prompts: dict[dict] = None
        self.current_image_data: ImageData = None
        self.current_image = None
        self.current_image_label: str = None
        self.current_mask_label: str = None
        
        self.toggle_mask_button: tk.Button = None
        self.is_select_var: tk.BooleanVar = None
        self.save_img_check_button: tk.Checkbutton = None
