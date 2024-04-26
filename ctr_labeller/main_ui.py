import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

import tkinter as tk
from PIL import ImageTk,Image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


NO_IMAGE_NAME = "No Image"

class ImageSelection(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, master=root)

        self.image_frame = tk.Frame(self)
        self.image_frame.grid(row=0, column=0)

        # Image
        self.left_canvas = tk.Canvas(self.image_frame)
        self.left_canvas.grid(row=0, column=0)
        self.left_label = tk.Label(self.image_frame, text=NO_IMAGE_NAME, bg="skyblue")
        self.left_label.grid(row=1, column=0, padx=10, ipadx=10, sticky="nsew")

        self.right_canvas = tk.Canvas(self.image_frame)
        self.right_canvas.grid(row=0, column=1)
        self.right_label = tk.Label(self.image_frame, text=NO_IMAGE_NAME, bg="skyblue")
        self.right_label.grid(row=1, column=1, padx=10, ipadx=10, sticky="nsew")

        blank_image = np.zeros((400, 400, 3), np.uint8)
        self.__draw_img_impl(self.left_canvas, blank_image, scale=1)
        self.__draw_img_impl(self.right_canvas, blank_image, scale=1)

        ## Select button
        self.is_select_var = tk.BooleanVar(value=True) # Image is to be saved by default
        self.save_img_check_button = tk.Checkbutton(self, \
            text = "Save Image?", variable = self.is_select_var, \
            onvalue = True, offvalue = False, state="disabled", height=self.image_y//100)
        self.save_img_check_button.grid(row=2, column=0, sticky="nsew")

    def __draw_img_impl(self, canvas, img, scale):
        resized_cv_img = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale)) 
        self.image_x = resized_cv_img.shape[0]
        self.image_y = resized_cv_img.shape[1]
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(resized_cv_img))
        canvas.image = tk_img
        canvas.create_image(10, 10, anchor=tk.NW, image=tk_img)

    # In this app, images should all have the same size
    def set_context(self, left_img_data, right_img_data, scale):
        self.__draw_img_impl(self.left_canvas, left_img_data.image, scale=scale)
        self.__draw_img_impl(self.right_canvas, right_img_data.image, scale=scale)

        # Dynamic changes to buttons
        self.left_label.configure(text=left_img_data.name)
        self.right_label.configure(text=right_img_data.name)

        self.is_select_var.set(True) # reset button
        self.save_img_check_button.configure(state="active", height=self.image_y//100)
        
    def disable_context(self, img_x_size, img_y_size):
        blank_image = np.zeros((img_x_size, img_y_size, 3), np.uint8)
        self.__draw_img_impl(self.left_canvas, blank_image, scale=1)
        self.__draw_img_impl(self.right_canvas, blank_image, scale=1)

        self.left_label.configure(text=NO_IMAGE_NAME)
        self.right_label.configure(text=NO_IMAGE_NAME)
        self.is_select_var.set(True)
        self.save_img_check_button.configure(state="disabled", height=self.image_y//100)

class CTRLabellerApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.frame_padx = 10
        self.frame_pady = 10
        self.selection_grid_size = (3, 2)
        self.selection_num = self.selection_grid_size[0] * self.selection_grid_size[1]

        self.selections = []
        for i in range(self.selection_grid_size[0]):
            for j in range(self.selection_grid_size[1]):
                selection = ImageSelection(self)
                selection.grid(row=i, column=j, padx=self.frame_padx, pady=self.frame_pady)
                self.selections.append(selection)

        self.is_done = False
        self.stereo_image_datas = []

    def set_stereo_image_datas(self, stereo_image_datas):
        self.stereo_image_datas = stereo_image_datas
        self.img_idx = 0
        self.is_done = self.__present_next()

    # Return value
    # False: There is more iterations
    # True: The iterations are done
    def __present_next(self):
        selection_idx = 0
        scale = 4

        while(True):
            self.selections[selection_idx].set_context(
                self.stereo_image_datas[self.img_idx].left, self.stereo_image_datas[self.img_idx].right, scale)
            self.img_idx += 1
            selection_idx += 1
            is_selection_over = selection_idx >= self.selection_num
            is_img_idx_over = self.img_idx >= len(self.stereo_image_datas)
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

from dataclasses import dataclass
from typing import Tuple

@dataclass
class ImageData:
    # Loading
    image: np.ndarray
    name: str

    # After Processing
    mask: np.ndarray = None

@dataclass
class StereoImageData:
    left: ImageData
    right: ImageData

def load_image_data(path):
    image_datas = []
    for file in sorted(glob.glob(path)):
        img = cv2.imread(file)
        img_data = ImageData(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), file.split("/")[-1])
        image_datas.append(img_data)
    return np.array(image_datas)

def print_stereo_names(stereo_image_datas, print_range):
    for i in print_range:
        print("left: {}, right: {}".format(stereo_image_datas[i].left.name, stereo_image_datas[i].right.name))

def load_stereo_image_data(left_path, right_path):
    # Test here for alot of images
    test_large_multiplier = 1 # default is 1
    left_image_datas = np.array([])
    right_image_datas = np.array([])
    for i in range(test_large_multiplier):
        left_image_datas = np.append(left_image_datas, load_image_data(left_path))
        right_image_datas = np.append(right_image_datas, load_image_data(right_path))
        print(i)

    assert len(left_image_datas) == len(right_image_datas)
    stereo_image_datas = [StereoImageData(left_image_datas[i], right_image_datas[i]) \
                                for i in range(len(left_image_datas))]
    return stereo_image_datas

if __name__ == "__main__":
    # Loading Images

    stereo_image_datas = load_stereo_image_data(
        "data/ctr_capture_apr_25_24/cam1_*.png",
        "data/ctr_capture_apr_25_24/cam2_*.png")

    print("Please double check names if stereo data is properly correlated")
    print_stereo_names(stereo_image_datas, range(len(stereo_image_datas)))

    # SAM Create Masks
    print("SAM is creating masks ...")

    print("SAM finished creating masks!")

    # Start the app
    app = CTRLabellerApp()
    app.title("CTR SAM Labeller")

    test_range = None # None for full range
    app.set_stereo_image_datas(stereo_image_datas[0:test_range])

    print("Number of images: ", len(stereo_image_datas))
    print("Press [n] to save and present next picture")

    app.bind("n", app.keypress_event)
    app.mainloop()


    # # image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2),
    # #            interpolation = cv2.INTER_LINEAR)

    # input_box = np.array([1000, 400, 1700, 760]) # xyxy format
    # input_point = np.array([[1400, 740]])
    # input_label = np.array([1])

    # # Show the image for debugging
    # plt.figure(figsize=(10,10))
    # show_box(input_box, plt.gca())
    # show_points(input_point, input_label, plt.gca(), marker_size=40)
    # plt.imshow(images[0])
    # plt.axis('on')
    # plt.show()

    # # load SAM checkpoint and parameters
    # import sys
    # sys.path.append("..")
    # from segment_anything import sam_model_registry, SamPredictor

    # sam_checkpoint = "sam_vit_h_4b8939.pth"
    # model_type = "vit_h"

    # device = "cuda"

    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)

    # predictor = SamPredictor(sam)



    # for i in range(len(images)):
    #     image = images[i]
    #     predictor.set_image(image)

    #     # Predict masks
    #     masks, scores, logits = predictor.predict(
    #         point_coords=input_point,
    #         point_labels=input_label,
    #         box=input_box[None, :],
    #         multimask_output=True,
    #     )

    #     #
    #     for j, (mask, score) in enumerate(zip(masks, scores)):
    #         plt.figure(figsize=(10,10))
    #         plt.imshow(image)
    #         show_mask(mask, plt.gca())
    #         show_box(input_box, plt.gca())
    #         show_points(input_point, input_label, plt.gca(), marker_size=40)
    #         plt.title(f"Image  {i}, Mask {j+1}, Score: {score:.3f}", fontsize=18)
    #         plt.axis('off')
    #         plt.show()
    #         break

