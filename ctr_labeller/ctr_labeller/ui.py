import numpy as np
import cv2
import tkinter as tk
from dataclasses import dataclass
from typing import Tuple

from PIL import ImageTk, Image

NO_IMAGE_NAME = "No Image"
NO_MASK_NAME = "No Mask"
class ImageSelection(tk.Frame):
    def __init__(self, root, draw_height_px):
        tk.Frame.__init__(self, master=root)

        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0)

        self.draw_height_px = draw_height_px

        self.label_frame = tk.Frame(self)
        self.label_frame.grid(row=1, column=0, sticky="nsew")
        self.label_frame.grid_columnconfigure(0, weight=1)

        self.label = tk.Label(self.label_frame, text=NO_IMAGE_NAME, bg="skyblue")
        self.label.grid(row=0, column=0, ipadx=10, sticky="nsew")

        self.mask_label = tk.Label(self.label_frame, text=NO_MASK_NAME, bg="skyblue")
        self.mask_label.grid(row=0, column=1, ipadx=10, sticky="nsew")

        self.toggle_button = tk.Button(self, text ="Toggle Mask", command = self.toggle_mask,
                                       state="disabled", height=self.draw_height_px//150)
        self.toggle_button.grid(row=2, column=0, sticky="nsew")

        self.image_data = None
        blank_image = np.zeros((self.draw_height_px , self.draw_height_px , 3), np.uint8)
        self.__draw_img_impl(blank_image)
    
    def __draw_img_impl(self, img):
        scale = img.shape[1]/self.draw_height_px
        resized_cv_img = cv2.resize(img, (self.draw_height_px, int(img.shape[0]//scale)), interpolation=cv2.INTER_LINEAR)
        self.image_x = resized_cv_img.shape[0]
        self.image_y = resized_cv_img.shape[1]
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(resized_cv_img))
        self.canvas.image = tk_img
        self.canvas.configure(width=self.image_y, height=self.image_x)
        self.canvas.create_image(10, 10, anchor=tk.NW, image=tk_img)

    def set_context(self, image_data):
        self.image_data = image_data
        if len(image_data.prediction_outputs) >= 1:
            self.__draw_img_impl(image_data.prediction_outputs[image_data.current_mask_idx].masked_image)
            self.toggle_button.configure(state="active")
            self.mask_label.configure(text="Mask: {}".format(
                image_data.prediction_outputs[image_data.current_mask_idx].input_prompt["name"]))
        else:
            self.__draw_img_impl(image_data.image)

        # Dynamic changes to buttons
        self.label.configure(text="Image: {}".format(image_data.name))

    def disable_context(self, img_x_size, img_y_size):
        blank_image = np.zeros((img_x_size, img_y_size, 3), np.uint8)
        self.__draw_img_impl(blank_image)
        self.label.configure(text=NO_IMAGE_NAME)
        self.mask_label.configure(text=NO_MASK_NAME)
        self.toggle_button.configure(state="disabled")

    def toggle_mask(self):
        if len(self.image_data.prediction_outputs) >= 1:
            self.image_data.current_mask_idx = (self.image_data.current_mask_idx + 1) % len(self.image_data.prediction_outputs)
            self.__draw_img_impl(self.image_data.prediction_outputs[self.image_data.current_mask_idx].masked_image)
            self.mask_label.configure(text="Mask: {}".format(
                self.image_data.prediction_outputs[self.image_data.current_mask_idx].input_prompt["name"]))

class StereoImageSelection(tk.Frame):
    def __init__(self, root, draw_height_px):
        tk.Frame.__init__(self, master=root)

        self.image_frame = tk.Frame(self)
        self.image_frame.grid(row=0, column=0)

        # Image and label
        self.left_image_selection = ImageSelection(self.image_frame, draw_height_px)
        self.left_image_selection.grid(row=0, column=0)
        self.right_image_selection = ImageSelection(self.image_frame, draw_height_px)
        self.right_image_selection.grid(row=0, column=1)

        ## Select button
        self.is_select_var = tk.BooleanVar(value=True) # Image is to be saved by default
        self.save_img_check_button = tk.Checkbutton(self, \
            text = "Save Images?", variable = self.is_select_var, \
            onvalue = True, offvalue = False, state="disabled", height=draw_height_px//100)
        self.save_img_check_button.grid(row=1, column=0, sticky="nsew")

    # In this app, images should all have the same size
    def set_context(self, left_img_data, right_img_data):
        self.left_image_selection.set_context(left_img_data)
        self.right_image_selection.set_context(right_img_data)

        self.is_select_var.set(True) # reset button
        self.save_img_check_button.configure(state="active")
        
    def disable_context(self, img_x_size, img_y_size):
        self.left_image_selection.disable_context(img_x_size, img_y_size)
        self.right_image_selection.disable_context(img_x_size, img_y_size)      
        self.is_select_var.set(True)
        self.save_img_check_button.configure(state="disabled")

@dataclass
class CTRLabellerAppConfig:
    visualize_input_prompt: bool = False
    selection_grid_size: Tuple[int, int] = (1, 1)
    selection_image_height_px: int = 1200
    frame_padx: int = 10
    frame_pady: int = 10

class CTRLabellerApp(tk.Tk):
    def __init__(self, config: CTRLabellerAppConfig, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # TODO, from config
        self.config = config
        self.selection_num = self.config.selection_grid_size[0] * self.config.selection_grid_size[1]

        self.selections = []
        for i in range(self.config.selection_grid_size[0]):
            for j in range(self.config.selection_grid_size[1]):
                selection = StereoImageSelection(self, self.config.selection_image_height_px)
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
            self.selections[selection_idx].set_context(
                self.stereo_image_data.left[self.img_idx],
                self.stereo_image_data.right[self.img_idx])
            self.img_idx += 1
            selection_idx += 1
            is_selection_over = selection_idx >= self.selection_num
            is_img_idx_over = self.img_idx >= len(self.stereo_image_data.left)
            if is_selection_over:
                return is_img_idx_over # There is a next iteration
            if is_img_idx_over: # selection not over
                break # To do for loop, set blank images

        while selection_idx < self.selection_num:
            self.selections[selection_idx].disable_context(self.selections[selection_idx-1].image_x, self.selections[selection_idx-1].image_y)
            selection_idx += 1
        return True

    def __save_selections(self):
        print("Saving selections")

    def keypress_event(self, input):
        # print(type(input)) # What is tkinter giving here?
        self.__save_selections() # Save the currently presented from __present_next()
        if self.is_done:
            self.quit()
            print("All images are done")
            return
    
        print("Presenting next set of images")
        self.is_done = self.__present_next()