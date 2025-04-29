import cv2
import numpy as np
import copy
import tkinter as tk
from typing import List
from ctr_labeller.types import ImageData

def add_keypoint(curr_input_prompts, keypoint):
   curr_input_prompts["point_coords"].append(keypoint)
   curr_input_prompts["point_labels"] = np.ones((len(curr_input_prompts["point_coords"],)))

def remove_keypoint(curr_input_prompts):
    if curr_input_prompts["point_coords"] is not None:
        if len(curr_input_prompts["point_coords"]): curr_input_prompts["point_coords"].pop()
    curr_input_prompts["point_labels"] = np.ones((len(curr_input_prompts["point_coords"],)))

def add_bounding_box(curr_input_prompts, bounding_box):
    curr_input_prompts["box"] = bounding_box

def remove_bounding_box(curr_input_prompts):
    curr_input_prompts["box"] = None

def create_img_with_input_prompts(img, input_prompt):
    prompted_img = copy.deepcopy(img)
    if input_prompt["box"] is not None:
        draw_box(prompted_img, input_prompt["box"])
    if input_prompt["point_coords"] is not None:
        draw_keypoints(prompted_img, input_prompt["point_coords"])
    return prompted_img

def draw_box(img, box_prompt):
    cv2.rectangle(img,
                    (box_prompt[0], box_prompt[1]), (box_prompt[2], box_prompt[3]),
                    color=(0, 255, 0), thickness=2)

def draw_keypoints(img, keypoints):
    for pt in keypoints:
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
            "name": None,
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
