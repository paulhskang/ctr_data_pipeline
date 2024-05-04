import copy
import cv2
import numpy as np
import tkinter as tk
from typing import Tuple
from PIL import ImageTk, Image

from ctr_labeller.config.utils import configure
from ctr_labeller.datasaver import DataSaver

NO_IMAGE_NAME = "No Image"
NO_MASK_NAME = "No Mask"

class ImageSelection(tk.Frame):
    def __init__(self, root, draw_height_px,
                 visualize_input_prompt: bool = False, zoom_factor: float = 2.0):
        tk.Frame.__init__(self, master=root)

        self.visualize_input_prompt = visualize_input_prompt
        self.zoom_factor = zoom_factor

        # Canvas Frame
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<Button-1>", self.on_click_toggle_zoom)
    
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
        # self.toggle_zoom_button = tk.Button(self.button_frame, text ="Zoom to Mask", command = self.toggle_zoom,
        #                                state="disabled", height=self.draw_height_px//scaler)
        # self.toggle_zoom_button.grid(row=0, column=0, sticky="nsew")
        
        ## Toggle Button
        self.toggle_mask_button = tk.Button(self.button_frame, text ="Toggle Mask", command = self.toggle_mask,
                                       state="disabled", height=self.draw_height_px//scaler)
        self.toggle_mask_button.grid(row=0, column=0, sticky="nsew")
        
        ## Select button
        self.is_select_var = tk.BooleanVar(value=True) # Image is to be saved by default
        self.save_img_check_button = tk.Checkbutton(self.button_frame, \
            text = "Save Mask?", variable = self.is_select_var, \
            onvalue = True, offvalue = False, state="disabled", height=draw_height_px//scaler)
        self.save_img_check_button.grid(row=0, column=1, sticky="nsew")

        blank_image = np.zeros((self.draw_height_px , self.draw_height_px , 3), np.uint8)
        self.current_image  = self.__draw_img_impl(blank_image)
        
        # State data
        self.image_data = None
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
        return img

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
            # self.toggle_zoom_button.configure(state="active")
            self.toggle_mask_button.configure(state="active")
            self.mask_label.configure(text="Mask: {},\nscore: {:.5f}, area_ratio: {:.5f}".format(
                pred_output.input_prompt["name"],
                pred_output.score,
                pred_output.area_ratio))
        self.current_image = self.__draw_img_impl(image_to_draw)

        # Dynamic changes to buttons
        self.is_select_var.set(True) # reset button
        self.save_img_check_button.configure(state="active")
        self.label.configure(text="Image: {}".format(image_data.name))
        self.is_zoomed = False

    def disable_context(self, img_x_size, img_y_size):
        blank_image = np.zeros((img_x_size, img_y_size, 3), np.uint8)
        self.current_image = self.__draw_img_impl(blank_image)
        self.label.configure(text=NO_IMAGE_NAME)
        self.mask_label.configure(text=NO_MASK_NAME)
        # self.toggle_zoom_button.configure(state="disabled")
        self.toggle_mask_button.configure(state="disabled")
        self.is_select_var.set(True)
        self.save_img_check_button.configure(state="disabled")
        self.is_zoomed = False

    def toggle_mask(self):
        self.is_zoomed = False
        if len(self.image_data.prediction_outputs) >= 1:
            self.image_data.current_mask_idx = (self.image_data.current_mask_idx + 1) % len(self.image_data.prediction_outputs)
            pred_output = self.image_data.prediction_outputs[self.image_data.current_mask_idx]
            prompted_img = self.__create_img_with_input_prompt(
                pred_output.masked_image,
                pred_output.input_prompt)
            self.current_image = self.__draw_img_impl(prompted_img)
            self.mask_label.configure(text="Mask: {},\nscore: {:.5f}, area_ratio: {:.5f}".format(
                pred_output.input_prompt["name"],
                pred_output.score,
                pred_output.area_ratio))
        
    def __create_zoomed_image(self, img, px, py):        
        new_h = int(img.shape[0] // self.zoom_factor)
        new_w = int(img.shape[1] // self.zoom_factor)

        y = np.clip(py - new_h // 2, 0, img.shape[0])
        x = np.clip(px - new_w // 2, 0, img.shape[1])

        return img[y:y+new_h, x:x+new_w]
    
    def on_click_toggle_zoom(self, event):
        self.is_zoomed = not self.is_zoomed
        if self.is_zoomed:
            self.__draw_img_impl(self.__create_zoomed_image(self.current_image, event.x, event.y))
        else: # Not zoomed
            self.__draw_img_impl(self.current_image)

class StereoImageSelection(tk.Frame):
    def __init__(self, root, draw_height_px, 
                 visualize_input_prompt: bool = False, zoom_factor: float = 2.0):
        tk.Frame.__init__(self, master=root)

        self.image_frame = tk.Frame(self)
        self.image_frame.grid(row=0, column=0)

        # Image and label
        self.left_image_selection = ImageSelection(self.image_frame,
                                                   draw_height_px, visualize_input_prompt, zoom_factor)
        self.left_image_selection.grid(row=0, column=0)
        self.right_image_selection = ImageSelection(self.image_frame,
                                                    draw_height_px, visualize_input_prompt, zoom_factor)
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

@configure
class CTRLabellerAppConfig:
    visualize_input_prompt: bool = False
    zoom_factor: float = 2.0
    selection_grid_size: Tuple[int, int] = (1, 1)
    selection_image_height_px: int = 1200
    frame_padx: int = 10
    frame_pady: int = 10

class CTRLabellerApp(tk.Tk):
    def __init__(self, config: CTRLabellerAppConfig, data_saver: DataSaver, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.config = config
        self.data_saver = data_saver
        self.selection_num = self.config.selection_grid_size[0] * self.config.selection_grid_size[1]
        self.selections = []
        for i in range(self.config.selection_grid_size[0]):
            for j in range(self.config.selection_grid_size[1]):
                selection = StereoImageSelection(self, self.config.selection_image_height_px,
                                                 self.config.visualize_input_prompt, self.config.zoom_factor)
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
                self.selections[selection_idx-1].left_image_selection.image_x, 
                self.selections[selection_idx-1].left_image_selection.image_y)
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
