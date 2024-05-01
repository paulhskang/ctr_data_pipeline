import copy
import cv2
import os
from dataclasses import dataclass
import numpy as np
import tkinter as tk
from typing import Tuple
from PIL import ImageTk, Image

from ctr_labeller.types import convert_mask_torch_to_opencv
from ctr_labeller.config.utils import configure

NO_IMAGE_NAME = "No Image"
NO_MASK_NAME = "No Mask"

class ImageSelection(tk.Frame):
    def __init__(self, root, draw_height_px, visualize_input_prompt = False):
        tk.Frame.__init__(self, master=root)

        self.visualize_input_prompt = visualize_input_prompt
        
        # Canvas Frame
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0)

        self.draw_height_px = draw_height_px

        # Label Frame
        self.label_frame = tk.Frame(self)
        self.label_frame.grid(row=1, column=0, sticky="nsew")
        self.label_frame.grid_columnconfigure(0, weight=1)

        self.label = tk.Label(self.label_frame, text=NO_IMAGE_NAME, bg="skyblue")
        self.label.grid(row=0, column=0, ipadx=10, sticky="nsew")

        self.mask_label = tk.Label(self.label_frame, text=NO_MASK_NAME, bg="skyblue")
        self.mask_label.grid(row=0, column=1, ipadx=10, sticky="nsew")

        # Button Frame
        self.button_frame = tk.Frame(self)
        self.button_frame.grid(row=2, column=0, sticky="nsew")
        scaler = 150
        
        ## Zoom Button
        self.toggle_zoom_button = tk.Button(self.button_frame, text ="Zoom to Mask", command = self.toggle_zoom,
                                       state="disabled", height=self.draw_height_px//scaler)
        self.toggle_zoom_button.grid(row=0, column=0, sticky="nsew")
        
        ## Toggle Button
        self.toggle_mask_button = tk.Button(self.button_frame, text ="Toggle Mask", command = self.toggle_mask,
                                       state="disabled", height=self.draw_height_px//scaler)
        self.toggle_mask_button.grid(row=0, column=1, sticky="nsew")
        
        ## Select button
        self.is_select_var = tk.BooleanVar(value=True) # Image is to be saved by default
        self.save_img_check_button = tk.Checkbutton(self.button_frame, \
            text = "Save Mask?", variable = self.is_select_var, \
            onvalue = True, offvalue = False, state="disabled", height=draw_height_px//scaler)
        self.save_img_check_button.grid(row=0, column=2, sticky="nsew")

        blank_image = np.zeros((self.draw_height_px , self.draw_height_px , 3), np.uint8)
        self.__draw_img_impl(blank_image)
        
        # State data
        self.image_data = None
        self.current_prompted_image = None
        self.is_zoomed = False
    
    def __resize_img_to_window(self, img):
        scale = img.shape[1]/self.draw_height_px
        return cv2.resize(img, (self.draw_height_px, int(img.shape[0]//scale)), interpolation=cv2.INTER_LINEAR)
        
    def __draw_img_impl(self, img):
        img = self.__resize_img_to_window(img)
        self.image_x = img.shape[0]
        self.image_y = img.shape[1]   
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.image = tk_img
        self.canvas.configure(width=self.image_y, height=self.image_x)
        self.canvas.create_image(10, 10, anchor=tk.NW, image=tk_img)

    def __create_img_with_input_prompt(self, img, input_prompt = None):
        prompted_img = copy.deepcopy(img)
        if self.visualize_input_prompt and input_prompt:
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
    
    def set_context(self, image_data):
        self.image_data = image_data
        image_to_draw = image_data.image
        if len(image_data.prediction_outputs) >= 1:
            pred_output = image_data.prediction_outputs[image_data.current_mask_idx]
            image_to_draw = self.__create_img_with_input_prompt(
                pred_output.masked_image,
                pred_output.input_prompt)
            self.current_prompted_image = image_to_draw
            self.toggle_zoom_button.configure(state="active")
            self.toggle_mask_button.configure(state="active")
            self.mask_label.configure(text="Mask: {},\nscore: {:.5f}, area_ratio: {:.5f}".format(
                pred_output.input_prompt["name"],
                pred_output.score,
                pred_output.area_ratio))
        self.__draw_img_impl(image_to_draw)

        # Dynamic changes to buttons
        self.is_select_var.set(True) # reset button
        self.save_img_check_button.configure(state="active")
        self.label.configure(text="Image: {}".format(image_data.name))

    def disable_context(self, img_x_size, img_y_size):
        blank_image = np.zeros((img_x_size, img_y_size, 3), np.uint8)
        self.__draw_img_impl(blank_image)
        self.label.configure(text=NO_IMAGE_NAME)
        self.mask_label.configure(text=NO_MASK_NAME)
        self.toggle_zoom_button.configure(state="disabled")
        self.toggle_mask_button.configure(state="disabled")
        self.is_select_var.set(True)
        self.save_img_check_button.configure(state="disabled")

    def toggle_mask(self):
        if len(self.image_data.prediction_outputs) >= 1:
            self.image_data.current_mask_idx = (self.image_data.current_mask_idx + 1) % len(self.image_data.prediction_outputs)
            pred_output = self.image_data.prediction_outputs[self.image_data.current_mask_idx]
            self.current_prompted_image = self.__create_img_with_input_prompt(
                pred_output.masked_image,
                pred_output.input_prompt)
            self.__draw_img_impl(self.current_prompted_image)
            self.mask_label.configure(text="Mask: {},\nscore: {:.5f}, area_ratio: {:.5f}".format(
                pred_output.input_prompt["name"],
                pred_output.score,
                pred_output.area_ratio))
            
    def __create_zoomed_image_to_mask(self, img, mask):        
        opencv_mask = convert_mask_torch_to_opencv(mask)
        opencv_mask = opencv_mask.reshape(opencv_mask.shape[0], -1)
        contours, _ = cv2.findContours(opencv_mask, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = cv2.boundingRect(contours[0])

        # keep same aspect ratio        
        x, y, w, h = bbox
        original_aspect_ratio = float(img.shape[0])/ float(img.shape[1])
        bbox_aspect_ratio = float(h)/float(w)

        new_bbox = np.array([0, 0, 0, 0])
        if bbox_aspect_ratio < original_aspect_ratio:
            new_bbox[2] = w
            new_bbox[3] = int(w * original_aspect_ratio)
            new_bbox[0] = x
            new_bbox[1] = np.clip(y + h // 2 - new_bbox[3] //2, 0, img.shape[0])
        else: # bbox_aspect_ratio > original_aspect_ratio:
            new_bbox[3] = h
            new_bbox[2] = int(h / original_aspect_ratio)
            new_bbox[1] = y
            new_bbox[0] = np.clip(x + w // 2 - new_bbox[2] //2, 0, img.shape[1])
        x, y, w, h = new_bbox
        return img[y:y+h, x:x+w]
        # return img[self.input_prompt_min_xy[0]:self.input_prompt_max_xy[0]]
    
    def toggle_zoom(self):
        self.is_zoomed = not self.is_zoomed
        if self.is_zoomed:
            self.__draw_img_impl(self.__create_zoomed_image_to_mask(
                self.current_prompted_image, self.image_data.prediction_outputs[self.image_data.current_mask_idx].mask))
        else: # Not zoomed
            self.__draw_img_impl(self.current_prompted_image)
            
class StereoImageSelection(tk.Frame):
    def __init__(self, root, draw_height_px, visualize_input_prompt = False):
        tk.Frame.__init__(self, master=root)

        self.image_frame = tk.Frame(self)
        self.image_frame.grid(row=0, column=0)

        # Image and label
        self.left_image_selection = ImageSelection(self.image_frame,
                                                   draw_height_px, visualize_input_prompt)
        self.left_image_selection.grid(row=0, column=0)
        self.right_image_selection = ImageSelection(self.image_frame,
                                                    draw_height_px, visualize_input_prompt)
        self.right_image_selection.grid(row=0, column=1)
        self.current_frame_id = -1
        self.is_activate = False

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
        self.left_image_selection.image_data.is_save_mask = self.left_image_selection.is_select_var.get()
        self.right_image_selection.image_data.is_save_mask = self.right_image_selection.is_select_var.get()
        return self.current_frame_id, self.left_image_selection.image_data, self.right_image_selection.image_data

import pandas as pd
import atexit

class CTRLabellerDataSaver:
    def __init__(self, save_root_path):
        self.save_root_path = save_root_path
        self.fields = ["left_image", "right_image", "is_processed", "left_mask_fail", "right_mask_fail", "left_mask", "right_mask"]
        self.mask_path = os.path.join(save_root_path, "masks") 
        if not os.path.exists(self.mask_path): # Means no format path
            os.mkdir(self.mask_path)

        self.reference_file_path = os.path.join(save_root_path, "reference.csv")
        if os.path.isfile(self.reference_file_path):
            data_frame = pd.read_csv(self.reference_file_path, index_col=0)
            self.reference_dict = data_frame.to_dict(orient='index')
        else:
            self.reference_dict = {}
        atexit.register(self.destructor)

    def check_is_mask_processed(self, frame_id):
        if frame_id in self.reference_dict:
            return True
        return False

    def save_current_mask(self, image_data):
        mask_name = "mask_{}".format(image_data.name)
        fullpath_mask = os.path.join(self.mask_path, "mask_{}".format(image_data.name))
        # if os.path.exists(fullpath_mask): # Not working properly
        #     print("{} path exists! Overriding".format(fullpath_mask))
        fullpath_image_and_mask = os.path.join(self.mask_path, "image_and_mask_{}".format(image_data.name))
        # if os.path.exists(fullpath_image_and_mask): # Not working properly
        #     print("{} path exists! Overriding".format(fullpath_image_and_mask))

        current_prediction_output = image_data.prediction_outputs[image_data.current_mask_idx]
        cv2.imwrite(fullpath_mask, convert_mask_torch_to_opencv(current_prediction_output.mask))
        cv2.imwrite(fullpath_image_and_mask, cv2.cvtColor(current_prediction_output.masked_image, cv2.COLOR_RGB2BGR))
        return os.path.join("mask", mask_name)

    def save_current_stereo_masks(self, frame_id, image_data_left, image_data_right):
        left_mask_name = ""
        if image_data_left.is_save_mask:
            left_mask_name = self.save_current_mask(image_data_left)
        right_mask_name = ""
        if image_data_right.is_save_mask:
            right_mask_name = self.save_current_mask(image_data_right)
        right_mask_name = self.save_current_mask(image_data_right)
        self.reference_dict[frame_id] = {"left_image": image_data_left.name, "right_image": image_data_right.name,
             "is_processed": True, "left_mask_fail": not image_data_left.is_save_mask, "right_mask_fail": not image_data_right.is_save_mask, 
             "left_mask": left_mask_name, "right_mask": right_mask_name}

    def destructor(self):
        df = pd.DataFrame.from_dict(self.reference_dict, orient='index')
        df.index.name = "frame_id"
        df.to_csv(self.reference_file_path)

@configure
class CTRLabellerAppConfig:
    visualize_input_prompt: bool = False
    selection_grid_size: Tuple[int, int] = (1, 1)
    selection_image_height_px: int = 1200
    frame_padx: int = 10
    frame_pady: int = 10

class CTRLabellerApp(tk.Tk):
    def __init__(self, config: CTRLabellerAppConfig, data_saver: CTRLabellerDataSaver, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.config = config
        self.data_saver = data_saver
        self.selection_num = self.config.selection_grid_size[0] * self.config.selection_grid_size[1]
        self.selections = []
        for i in range(self.config.selection_grid_size[0]):
            for j in range(self.config.selection_grid_size[1]):
                selection = StereoImageSelection(self,
                    self.config.selection_image_height_px, self.config.visualize_input_prompt)
                selection.grid(row=i, column=j, padx=self.config.frame_padx, pady=self.config.frame_pady)
                self.selections.append(selection)

        # State data
        self.img_idx = 0
        self.is_done = False
        self.stereo_image_data = None

    def set_stereo_image_data(self, stereo_image_data):
        self.img_idx = 0
        self.stereo_image_data = stereo_image_data
        self.is_done = self.__present_next()
            
    # Return value
    # False: There is more iterations
    # True: The iterations are done
    def __present_next(self):
        selection_idx = 0

        while(True):
            if not self.data_saver.check_is_mask_processed(self.stereo_image_data.frame_ids[self.img_idx]):
                self.selections[selection_idx].set_context(
                    self.stereo_image_data.frame_ids[self.img_idx],
                    self.stereo_image_data.left[self.img_idx],
                    self.stereo_image_data.right[self.img_idx])
                selection_idx += 1
            self.img_idx += 1
            # Checks
            is_selection_over = selection_idx >= self.selection_num
            is_img_idx_over = self.img_idx >= len(self.stereo_image_data.left)
            if is_selection_over:
                return is_img_idx_over # There is a next iteration
            if is_img_idx_over: # selection not over
                break # To do for loop, set blank images

        while selection_idx < self.selection_num:
            self.selections[selection_idx].disable_context(
                self.selections[selection_idx-1].left_image_selection.image_x, self.selections[selection_idx-1].left_image_selection.image_y)
            selection_idx += 1
        return True

    def __save_selections(self):
        print("Saving selections")
        for selection in self.selections:
            if not selection.is_active:
                continue
            frame_id, image_left, image_right = selection.get_current_context()
            self.data_saver.save_current_stereo_masks(frame_id, image_left, image_right)

    def keypress_event(self, input):
        # print(type(input)) # What is tkinter giving here?
        self.__save_selections() # Save the currently presented from __present_next()
        if self.is_done:
            self.quit()
            print("All images are done")
            return
    
        print("Presenting next set of images")
        self.is_done = self.__present_next()
