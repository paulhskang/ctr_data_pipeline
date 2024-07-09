import cv2
import copy
import numpy as np

np.set_printoptions(7, suppress=True)

CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX

def create_point_text(x, y):
    return str(x) + ', ' + str(y)

def draw_point(img, x, y, radius = 6):
    cv2.circle(img, (x, y), radius, 255, -1)
    cv2.putText(img, create_point_text(x,y), (x,y), CV2_FONT,
                1, (255, 0, 0), 2)

def draw_box(img, left_x, left_y, right_x, right_y):
    color = (255, 0, 0)
    thickness = 2
    image = cv2.rectangle(image, (left_x, left_y), (right_x, right_y), color, thickness)
    cv2.putText(img, create_point_text(left_x, left_y), CV2_FONT,
                1, (255, 0, 0), 2)
    cv2.putText(img, create_point_text(right_x, right_y), CV2_FONT,
                1, (255, 0, 0), 2)

class KeypointClickEvent:
    def __init__(self, name, img):
        self.name = name
        self.img = img
        self.x = 0
        self.y = 0

        self.radius = 6

        cv2.imshow(name, img)
        cv2.setMouseCallback(name, self.click_event)
        cv2.waitKey(0)

    def get_xy(self):
        return np.array([self.x, self.y])

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x = x
            self.y = y
            img = copy.deepcopy(self.img)
            draw_point(img, self.x, self.y, radius=self.radius)
            cv2.imshow(self.name, img)

class BoundingBoxClickEvent:
    def __init__(self, name, img):
        self.name = name
        self.img = img
        self.bounding_box = np.array([0, 0, 0, 0])
        self.radius = 6

        cv2.imshow(name, img)
        cv2.setMouseCallback(name, self.click_event)
        cv2.waitKey(0)

    def get_bounding_box(self):
        return self.bounding_box

    def click_event(self, event, x, y, flags, params):

        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            self.bounding_box[0] = x
            self.bounding_box[1] = y

        if event == cv2.EVENT_RBUTTONDOWN:
            self.bounding_box[1] = x
            self.bounding_box[2] = y

        img = copy.deepcopy(self.img)
        draw_box(img, self.bounding_box[0], self.bounding_box[1], self.bounding_box[2], self.bounding_box[3])
        cv2.imshow(self.name, img)

class InputPromptGenerationApp():
    def __init__(self, stereo_image_dataset, sam_predictor, resize_img_divider = 1):
        self.stereo_image_dataset = stereo_image_dataset
        self.resize_img_divider = resize_img_divider
        self.sam_predictor = sam_predictor

    def generate_input_prompts(self):
        frame = self.stereo_image_dataset[0]

        # left_ce = KeypointClickEvent("left_image: click set a keypoint on the image", frame["left_image"])
        # print("next")
        # left_point_coords_xy = left_ce.get_xy()
        # left_cfe = BoundingBoxClickEvent("left_image: left click to set left of bounding box, right click to set right of bounding box", frame["left_image"])
        # left_bounding_box = left_cfe.get_bounding_box()

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
        return left_input_prompts, right_input_prompts
