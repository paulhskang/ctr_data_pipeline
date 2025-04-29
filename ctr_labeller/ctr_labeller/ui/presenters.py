import cv2
import tkinter as tk
from PIL import ImageTk, Image

from ctr_labeller.ui.types import ImageSelectorState

class ImageSelectionPresenter(tk.Frame):
    NO_IMAGE_NAME = "No Image"
    NO_MASK_NAME = "No Mask"
    def __init__(self, parent_frame, state:ImageSelectorState):
        tk.Frame.__init__(self, master=parent_frame)
        # Canvas Frame
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0)

        # Label Frame
        self.label_frame = tk.Frame(self)
        self.label_frame.grid(row=1, column=0, sticky="nsew")
        self.label_frame.grid_columnconfigure(0, weight=1)

        self.image_label = tk.Label(self.label_frame, text=self.NO_IMAGE_NAME, bg="skyblue")
        self.image_label.grid(row=0, column=0, ipadx=10, sticky="nsew")

        self.mask_label = tk.Label(self.label_frame, text=self.NO_MASK_NAME, bg="skyblue")
        self.mask_label.grid(row=0, column=1, ipadx=10, sticky="nsew")
        
        self.state = state
        self.state.canvas = self.canvas
        self.state.trigger_presenter_function = self.present_current_state
        
    def present_current_state(self):
        img_to_draw = self.state.current_image
        if self.state.is_zoomed:
            img_to_draw = self.state.zoom_function(self.state.current_image)
        self.draw_image(self.resize_image_to_window(img_to_draw))
        self.set_image_label(self.state.current_image_label)
        self.set_mask_label(self.state.current_mask_label)
    
    def resize_image_to_window(self, image):
        scale = image.shape[1]/self.state.c_draw_height_py
        self.state.resize_img_scale = scale
        resized_img = cv2.resize(image, (self.state.c_draw_height_py, int(image.shape[0]//scale)), interpolation=cv2.INTER_LINEAR)
        return resized_img
               
    def draw_image(self, image):
        self.image_x = image.shape[0]
        self.image_y = image.shape[1]   
        tk_image = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.canvas.image = tk_image
        self.canvas.configure(width=self.image_y, height=self.image_x)
        self.canvas.create_image(10, 10, anchor=tk.NW, image=tk_image)
        return image
    
    def set_image_label(self, text = None):
        if text == None:
            self.image_label.configure(text=self.NO_IMAGE_NAME)
            return
        # else
        self.image_label.configure(text=text)

    def set_mask_label(self, text = None):
        if text == None:
            self.mask_label.configure(text=self.NO_MASK_NAME)
            return
        # else
        self.mask_label.configure(text=text)

class StereoImagePresenter(tk.Frame):
    def __init__(self, parent_frame, left_image_state, right_image_state):
        tk.Frame.__init__(self, master=parent_frame)
        self.image_frame = tk.Frame(self)
        self.image_frame.grid(row=0, column=0)
        self.left = ImageSelectionPresenter(self.image_frame, left_image_state)
        self.left.grid(row=0, column=0)
        self.right = ImageSelectionPresenter(self.image_frame, right_image_state)
        self.right.grid(row=0, column=1)

    def present_current_state(self):
        self.left.present_current_state()
        self.right.present_current_state()

class OrganizedButtonGenerator(tk.Frame):
    def __init__(self, parent_frame, button_frame_location, max_col = 6):
        tk.Frame.__init__(self, master=parent_frame)
        self.button_frame = tk.Frame(parent_frame)
        self.grid(row=button_frame_location[0], column=button_frame_location[1], sticky="nsew")
        self.max_col = max_col
        self.current_row_idx = 0
        self.current_col_idx = 0

    def __organize_button(self, button):
        button.grid(row=self.current_row_idx, column=self.current_col_idx, sticky="nsew")
        self.current_col_idx += 1
        if self.current_col_idx == self.max_col:
            self.current_row_idx += 1
            self.current_col_idx = 0
        
    def create_check_button(self, **kwargs):
        select_var =  tk.BooleanVar(value=True)
        check_button = tk.Checkbutton(self, variable=select_var, **kwargs)
        self.__organize_button(check_button)
        return check_button, select_var
    
    def create_button(self, **kwargs):
        button = tk.Button(self, **kwargs)
        self.__organize_button(button)
        return button
