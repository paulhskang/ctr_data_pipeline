# from dataclasses import dataclass
from enum import Enum
import numpy as np

from ctr_labeller.types import ImageData
from ctr_labeller.ui.types import ImageSelectorState
from ctr_labeller.ui.presenters import ImageSelectionPresenter, StereoImagePresenter, create_img_with_input_prompt

class ImageSelectionType(Enum):
    BLANK = 0
    IMAGE = 1
    CURRENT_IMAGE = 2
    MASK_AND_PROMPT = 3




class ImageSelector:
    def __init__(self, image_selection_presenter: ImageSelectionPresenter):
        self.image_presenter = image_selection_presenter
        self.state = ImageSelectorState()
        self.disable_context()

    def select_image_and_present(self, selection: ImageSelectionType, mask_idx = -1):
        if selection == ImageSelectionType.BLANK:
            blank_image = np.zeros((
                self.image_presenter.draw_height_py,
                self.image_presenter.draw_height_py, 3), np.uint8)
            self.state.current_image = self.image_presenter.draw_image(
                self.image_presenter.resize_image_to_window(blank_image))
            self.image_presenter.set_image_label(None)
            self.image_presenter.set_mask_label(None)
            return
        elif selection == ImageSelectionType.IMAGE:
            self.state.current_image = self.image_presenter.draw_image(
                self.image_presenter.resize_image_to_window(self.state.current_image_data.image))
            self.image_presenter.set_image_label(text="Image: {}".format(self.state.current_image_data.name))
            self.image_presenter.set_mask_label(None)
            return
        # else selection == ImageSelectionType.MASK_AND_PROMPT:
        image_data = self.state.current_image_data
        if len(image_data.prediction_outputs) < 1:
            raise Exception("selection is Mask, but no mask outputs")
        pred_output = image_data.prediction_outputs[image_data.current_mask_idx]
        image_to_draw = create_img_with_input_prompt(
            pred_output.masked_image,
            pred_output.input_prompt)
        self.state.current_image = self.image_presenter.draw_image(
                self.image_presenter.resize_image_to_window(image_to_draw))
        self.image_presenter.set_image_label(text="Image: {}".format(image_data.name))
        self.image_presenter.set_mask_label(text="Mask: {},\nscore: {:.5f}, area_ratio: {:.5f}".format(
            pred_output.input_prompt["name"],
            pred_output.score,
            pred_output.area_ratio))
        return

    def set_context(self, image_data: ImageData, selection: ImageSelectionType):
        self.state.current_image_data = image_data
        self.select_image_and_present(selection=selection)

    def disable_context(self):
        self.select_image_and_present(ImageSelectionType.BLANK)
        self.state.current_image_data = None

class StereoImageSelector:
    def __init__(self, stereo_image_presenter: StereoImagePresenter):
        self.left_image_selector = ImageSelector(stereo_image_presenter.left)
        self.right_image_selector = ImageSelector(stereo_image_presenter.right)
        self.current_frame_id = -1
        self.is_active = False

    def set_context(self, frame_id, left_img_data, right_img_data):
        self.current_frame_id = frame_id
        self.left_image_selector.set_context(left_img_data)
        self.right_image_selector.set_context(right_img_data)
        self.is_active = True

    def disable_context(self):
        self.current_frame_id = -1
        self.left_image_selector.disable_context()
        self.right_image_selector.disable_context()
        self.is_active = False
