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
        # Image
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0)
    
        blank_image = np.zeros((400, 400, 3), np.uint8)
        self.__draw_img_impl(blank_image)

        # Context panel
        self.context_frame = tk.Frame(self)
        self.context_frame.grid(row=1, column=0)

        ## Name label
        self.label = tk.Label(self.context_frame, text=NO_IMAGE_NAME)
        self.label.grid(row=0, column=0, padx=10, ipadx=10)

        ## Select button
        self.is_select_var = tk.BooleanVar(value=True) # Image is to be saved by default
        self.save_img_check_button = tk.Checkbutton(self.context_frame, \
            text = "Save Image?", variable = self.is_select_var, \
            onvalue = True, offvalue = False, state="disabled", height=self.image_y//100)
        self.save_img_check_button.grid(row=0, column=1, padx=10, ipadx=10)

    def __draw_img_impl(self, img):
        self.image_x = img.shape[0]
        self.image_y = img.shape[1]
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.image = tk_img
        self.canvas.create_image(10, 10, anchor=tk.NW, image=tk_img)

    # In this app, images should all have the same size
    def draw_img(self, cv_img, scale, name=""):
        resized_cv_img = cv2.resize(cv_img, (cv_img.shape[1]//scale, cv_img.shape[0]//scale)) 
        self.__draw_img_impl(resized_cv_img)
        
        # Dynamic changes to buttons
        self.label.config(text=name)
        self.save_img_check_button.configure(state="active", height=self.image_y//100)

class CTRLabellerApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.frame_padx=10
        self.frame_pady=5
        self.selection_num = 3

        self.selections = []
        for i in range(self.selection_num):
            selection = ImageSelection(self)
            selection.grid(row=0, column=i, padx=self.frame_padx, pady=self.frame_pady)
            self.selections.append(selection)

        self.is_done = False

    def set_images(self, images, image_names):
        self.images = images
        self.image_names = image_names
        self.img_idx = 0
        self.is_done = self.__present_next()
        print(self.is_done)

    # Return value
    # False: There is more iterations
    # True: The iterations are done
    def __present_next(self):
        selection_idx = 0
        scale = 4

        while(True):
            self.selections[selection_idx].draw_img(self.images[self.img_idx], scale, image_names[self.img_idx])
            self.img_idx += 1
            selection_idx += 1
            is_selection_over = selection_idx >= self.selection_num
            is_img_idx_over = self.img_idx >= len(self.images)
            if is_selection_over:
                return is_img_idx_over # There is a next iteration
            
            if is_img_idx_over: # selection not over
                break # To do for loop, set blank images

        while selection_idx < self.selection_num:
            blank_image = np.zeros((self.selections[selection_idx-1].image_x, self.selections[selection_idx-1].image_y, 3), np.uint8)
            self.selections[selection_idx].draw_img(blank_image, scale=1, name=NO_IMAGE_NAME)
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

    
if __name__ == "__main__":
    # Loading Images
    images = []
    image_names = []
    for file in glob.glob("data/ctr_capture_apr_25_24/cam1_*.png"):
        img = cv2.imread(file)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_names.append(file.split("/")[-1])

    # SAM Create Masks
    print("SAM is creating masks")

    app = CTRLabellerApp()
    app.title("CTR SAM Labeller")

    test_range = None
    app.set_images(images=images[0:test_range], image_names=image_names[0:test_range])
    print("Number of images: ", len(images))
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

