import numpy as np
import cv2
import tkinter as tk
from PIL import ImageTk,Image

NO_IMAGE_NAME = "No Image"

class ImageSelection(tk.Frame):
    def __init__(self, root, draw_height_px):
        tk.Frame.__init__(self, master=root)

        self.draw_height_px = draw_height_px
        self.image_frame = tk.Frame(self)
        self.image_frame.grid(row=0, column=0)

        # Image and label
        self.left_canvas = tk.Canvas(self.image_frame)
        self.left_canvas.grid(row=0, column=0)
        self.left_label = tk.Label(self.image_frame, text=NO_IMAGE_NAME, bg="skyblue")
        self.left_label.grid(row=1, column=0, padx=10, ipadx=10, sticky="nsew")

        self.right_canvas = tk.Canvas(self.image_frame)
        self.right_canvas.grid(row=0, column=1)
        self.right_label = tk.Label(self.image_frame, text=NO_IMAGE_NAME, bg="skyblue")
        self.right_label.grid(row=1, column=1, padx=10, ipadx=10, sticky="nsew")

        blank_image = np.zeros((self.draw_height_px , self.draw_height_px , 3), np.uint8)
        self.__draw_img_impl(self.left_canvas, blank_image)
        self.__draw_img_impl(self.right_canvas, blank_image)

        ## Select button
        self.is_select_var = tk.BooleanVar(value=True) # Image is to be saved by default
        self.save_img_check_button = tk.Checkbutton(self, \
            text = "Save Images?", variable = self.is_select_var, \
            onvalue = True, offvalue = False, state="disabled", height=self.image_y//100)
        self.save_img_check_button.grid(row=2, column=0, sticky="nsew")

    def __draw_img_impl(self, canvas, img):
        scale = img.shape[1]/self.draw_height_px
        resized_cv_img = cv2.resize(img, (self.draw_height_px, int(img.shape[0]//scale)), interpolation=cv2.INTER_LINEAR)
        self.image_x = resized_cv_img.shape[0]
        self.image_y = resized_cv_img.shape[1]
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(resized_cv_img))
        canvas.image = tk_img
        canvas.configure(width=self.image_y, height=self.image_x)
        canvas.create_image(10, 10, anchor=tk.NW, image=tk_img)

    # In this app, images should all have the same size
    def set_context(self, left_img_data, right_img_data, use_mask):
        if use_mask:
            self.__draw_img_impl(self.left_canvas, left_img_data.prediction_outputs[0].masked_image)
            self.__draw_img_impl(self.right_canvas, right_img_data.prediction_outputs[0].masked_image)
        else:
            self.__draw_img_impl(self.left_canvas, left_img_data.image)
            self.__draw_img_impl(self.right_canvas, right_img_data.image)

        # Dynamic changes to buttons
        self.left_label.configure(text=left_img_data.name)
        self.right_label.configure(text=right_img_data.name)

        self.is_select_var.set(True) # reset button
        self.save_img_check_button.configure(state="active", height=self.image_y//100)
        
    def disable_context(self, img_x_size, img_y_size):
        blank_image = np.zeros((img_x_size, img_y_size, 3), np.uint8)
        self.__draw_img_impl(self.left_canvas, blank_image)
        self.__draw_img_impl(self.right_canvas, blank_image)

        self.left_label.configure(text=NO_IMAGE_NAME)
        self.right_label.configure(text=NO_IMAGE_NAME)
        self.is_select_var.set(True)
        self.save_img_check_button.configure(state="disabled", height=self.image_y//100)

class CTRLabellerApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # TODO, from config
        self.frame_padx = 10
        self.frame_pady = 10
        self.selection_grid_size = (1, 1)
        self.selection_num = self.selection_grid_size[0] * self.selection_grid_size[1]
        self.selection_image_height_px = 1200

        self.selections = []
        for i in range(self.selection_grid_size[0]):
            for j in range(self.selection_grid_size[1]):
                selection = ImageSelection(self, self.selection_image_height_px)
                selection.grid(row=i, column=j, padx=self.frame_padx, pady=self.frame_pady)
                self.selections.append(selection)

        self.is_done = False
        self.stereo_image_data = None
        self.use_mask = False

    def set_stereo_image_data(self, stereo_image_data, use_mask):
        self.stereo_image_data = stereo_image_data
        self.use_mask = use_mask
        self.img_idx = 0
        self.is_done = self.__present_next()

    # Return value
    # False: There is more iterations
    # True: The iterations are done
    def __present_next(self):
        selection_idx = 0

        while(True):
            self.selections[selection_idx].set_context(
                self.stereo_image_data.left[self.img_idx],
                self.stereo_image_data.right[self.img_idx], self.use_mask)
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