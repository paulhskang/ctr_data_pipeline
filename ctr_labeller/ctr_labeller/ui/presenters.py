import cv2
import copy
import tkinter as tk
from PIL import ImageTk, Image

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

class ImageSelectionPresenter(tk.Frame):
    NO_IMAGE_NAME = "No Image"
    NO_MASK_NAME = "No Mask"
    def __init__(self, parent_frame, draw_height_py):
        tk.Frame.__init__(self, master=parent_frame)
        self.draw_height_py = draw_height_py
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

    def resize_image_to_window(self, image):
        scale = image.shape[1]/self.state.c_draw_height_py
        return cv2.resize(image, (self.state.c_draw_height_py, int(image.shape[0]//scale)), interpolation=cv2.INTER_LINEAR)
       
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
    def __init__(self, parent_frame, draw_height_py):
        tk.Frame.__init__(self, master=parent_frame)
        self.image_frame = tk.Frame(self)
        self.image_frame.grid(row=0, column=0)
        self.left = ImageSelectionPresenter(self.image_frame, draw_height_py)
        self.left.grid(row=0, column=0)
        self.right = ImageSelectionPresenter(self.image_frame, draw_height_py)
        self.right.grid(row=0, column=1)

    #     self.image_frame = tk.Frame(self)
    #     self.image_frame.grid(row=0, column=0)

    #     # Image and label
    #     isc = ImageSelectionConfig(mask_widget=True, save_widget=True,
    #                                click_event_type=ClickEventType.ZOOM, draw_height_py=draw_height_py,
    #                                visualize_input_prompt=visualize_input_prompt, zoom_factor=zoom_factor)
    #     self.left_image_selection = ImageSelection(self.image_frame, isc)
    #     self.left_image_selection.grid(row=0, column=0)
    #     self.right_image_selection = ImageSelection(self.image_frame, isc)
    #     self.right_image_selection.grid(row=0, column=1)
    #     self.current_frame_id = -1
    #     self.is_active = False

