import cv2
import numpy as np
import copy
import tkinter as tk
from typing import List
from ctr_labeller.types import ImageData

def create_img_with_input_prompts(img, current_input_prompts):
    prompted_img = copy.deepcopy(img)
    draw_box(prompted_img, current_input_prompts)
    draw_keypoints(prompted_img, current_input_prompts)
    return prompted_img

def draw_box(img, current_input_prompts):
    if current_input_prompts["box"] is not None:
        cv2.rectangle(img,
                        (current_input_prompts["box"][0], current_input_prompts["box"][1]),\
                            (current_input_prompts["box"][2], current_input_prompts["box"][3]),
                        color=(0, 255, 0), thickness=2)

def draw_keypoints(img, current_input_prompts):
    if current_input_prompts["point_coords"] != []:
        for pt in current_input_prompts["point_coords"]:
            cv2.drawMarker(img,
                            (pt[0], pt[1]), color=(0, 255, 0), markerType=cv2.MARKER_DIAMOND,
                            markerSize=20, thickness=2, line_type=cv2.LINE_AA)

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
        self.current_input_prompts = {
            "name": "None",
            "box": None,
            "point_coords": [],
            "point_labels": None
        }
        self.current_image_data: ImageData = None
        self.current_image = None
        self.current_image_label: str = None
        self.current_mask_label: str = None
        
        self.toggle_mask_button: tk.Button = None
        self.is_select_var: tk.BooleanVar = None
        self.save_img_check_button: tk.Checkbutton = None

    def add_keypoint(self, keypoint):
        self.current_input_prompts["point_coords"].append(keypoint)
        self.current_input_prompts["point_labels"] = np.ones((len(self.current_input_prompts["point_coords"],)))
        self.update_input_prompt_name()

    def remove_keypoint(self):
        if self.current_input_prompts["point_coords"] != []: self.current_input_prompts["point_coords"].pop()
        self.current_input_prompts["point_labels"] = np.ones((len(self.current_input_prompts["point_coords"],)))
        self.update_input_prompt_name()

    def add_bounding_box(self, bounding_box):
        self.current_input_prompts["box"] = bounding_box
        self.update_input_prompt_name()

    def remove_bounding_box(self):
        self.current_input_prompts["box"] = None
        self.update_input_prompt_name()

    def update_input_prompt_name(self):
        if self.current_input_prompts["box"] is not None and self.current_input_prompts["point_coords"] != []:
            self.current_input_prompts["name"] = "box_and_point"
        elif self.current_input_prompts["point_coords"] != []:
            self.current_input_prompts["name"] = "point"
        elif self.current_input_prompts["box"] is not None:
            self.current_input_prompts["name"] = "box"
        else:
            self.current_input_prompts["name"] = "None"