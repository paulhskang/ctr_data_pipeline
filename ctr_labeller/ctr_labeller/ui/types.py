import tkinter as tk
from ctr_labeller.types import ImageData

class ImageSelectorState:
    def __init__(self, draw_height_py, zoom_factor):
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
        self.current_image_data: ImageData = None
        self.current_image = None
        self.current_image_label: str = None
        self.current_mask_label: str = None
        
        self.toggle_mask_button: tk.Button = None
        self.is_select_var: tk.BooleanVar = None
        self.save_img_check_button: tk.Checkbutton = None
