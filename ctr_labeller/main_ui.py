import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataclasses import dataclass
from ctr_labeller.types import load_stereo_image_data, print_stereo_names
from ctr_labeller.ui import CTRLabellerApp
from ctr_labeller.predictor import SAMBatchedPredictor, InputPrompt

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

def debug_input(image, input_box, input_point = None):
    plt.figure(figsize=(10,10))
    show_box(input_box, plt.gca())
    if input_point:
        show_points(input_point, np.array([1]), plt.gca(), marker_size=40)
    plt.imshow(image)
    plt.axis('on')

@dataclass
class CTRLabellerConfig:
    debug_inputs: bool = True
    apply_mask: bool = True
    test_num: int = None

if __name__ == "__main__":
    # Config
    config = CTRLabellerConfig()
    config.debug_inputs = False
    config.apply_mask = True
    # config.test_num = 4 # Set to None, for all

    # Loading Images
    stereo_image_data = load_stereo_image_data(
        "data/ctr_capture_apr_25_24/cam1_*.png",
        "data/ctr_capture_apr_25_24/cam2_*.png", config.test_num)

    print("Please double check names if stereo data is properly correlated")
    print_stereo_names(stereo_image_data, range(len(stereo_image_data.left)))

    # SAM Input
    
    # TODO, another gui for clicking points and boxes and save as prompts
    # if config.debug_inputs: 
    #     img_idx = 0
    #     debug_input(stereo_image_data[img_idx].left.image, left_input_box)
    #     debug_input(stereo_image_data[img_idx].right.image, right_input_box)
    #     plt.show()

    # SAM Create Masks
    if config.apply_mask:
        left_input_prompts = [
            {
                "name": "box_and_point",
                "box": np.array([300, 500, 1600, 1400]),
                "point_coords": np.array([[710, 260]]),
                "point_labels": np.array([1])
            },
            {
                "name": "point",
                "box": None,
                "point_coords": np.array([[710, 260]]),
                "point_labels": np.array([1])
            }
        ]

        right_input_prompts = [
            {
                "name": "box_and_point",
                "box": np.array([500, 600, 1500, 1400]),
                "point_coords": np.array([[1129, 400]]),
                "point_labels": np.array([1])
            },
            {
                "name": "point",
                "box": None,
                "point_coords": np.array([[1129, 400]]),
                "point_labels": np.array([1])
            }
        ]
        predictor = SAMBatchedPredictor()
        print("SAM is creating masks ...")
        predictor.predict(stereo_image_data.left, left_input_prompts)
        predictor.predict(stereo_image_data.right, right_input_prompts)
        print("SAM finished creating masks!")

    # Start the app
    app = CTRLabellerApp()
    app.title("CTR SAM Labeller")

    app.set_stereo_image_data(stereo_image_data)

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

