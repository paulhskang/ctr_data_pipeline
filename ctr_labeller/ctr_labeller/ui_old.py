import tkinter as tk
from typing import Tuple

from ctr_labeller.config.utils import configure
from ctr_labeller.datasaver import DataSaver
from ctr_labeller.types import StereoImageDataQueue
from ctr_labeller.ui_image_selection import ImageSelection, ImageSelectionConfig, ClickEventType

class StereoImageSelection(tk.Frame):
    def __init__(self, root, draw_height_py, 
                 visualize_input_prompt: bool = False, zoom_factor: float = 2.0):
        tk.Frame.__init__(self, master=root)

        self.image_frame = tk.Frame(self)
        self.image_frame.grid(row=0, column=0)

        # Image and label
        isc = ImageSelectionConfig(mask_widget=True, save_widget=True,
                                   click_event_type=ClickEventType.ZOOM, draw_height_py=draw_height_py,
                                   visualize_input_prompt=visualize_input_prompt, zoom_factor=zoom_factor)
        self.left_image_selection = ImageSelection(self.image_frame, isc)
        self.left_image_selection.grid(row=0, column=0)
        self.right_image_selection = ImageSelection(self.image_frame, isc)
        self.right_image_selection.grid(row=0, column=1)
        self.current_frame_id = -1
        self.is_active = False

    # In this app, images should all have the same size
    def set_context(self, frame_id, left_img_data, right_img_data):
        self.current_frame_id = frame_id
        self.left_image_selection.set_context(left_img_data)
        self.right_image_selection.set_context(right_img_data)
        self.is_active = True

    def disable_context(self, img_x_size, img_y_size):
        self.current_frame_id = -1
        self.left_image_selection.disable_context(img_x_size, img_y_size)
        self.right_image_selection.disable_context(img_x_size, img_y_size)      
        self.is_active = False

    def get_current_context(self):
        self.left_image_selection.state.current_image_data.is_save_mask = self.left_image_selection.save_widget.is_select_var.get()
        self.right_image_selection.state.current_image_data.is_save_mask = self.right_image_selection.save_widget.is_select_var.get()
        return self.current_frame_id, self.left_image_selection.state.current_image_data, self.right_image_selection.state.current_image_data

@configure
class CTRLabellerAppConfig:
    visualize_input_prompt: bool = False
    zoom_factor: float = 2.0
    selection_grid_size: Tuple[int, int] = (1, 1)
    selection_image_height_py: int = 1200
    frame_padx: int = 10
    frame_pady: int = 10

class CTRLabellerApp(tk.Tk):
    def __init__(self, config: CTRLabellerAppConfig, datasaver: DataSaver,
                 stereo_image_queue: StereoImageDataQueue, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.config = config
        self.selection_num = self.config.selection_grid_size[0] * self.config.selection_grid_size[1]
        self.stereo_image_queue = stereo_image_queue
        self.datasaver = datasaver
        self.selections = []
        for i in range(self.config.selection_grid_size[0]):
            for j in range(self.config.selection_grid_size[1]):
                selection = StereoImageSelection(self, self.config.selection_image_height_py,
                                                 self.config.visualize_input_prompt, self.config.zoom_factor)
                selection.grid(row=i, column=j, padx=self.config.frame_padx, pady=self.config.frame_pady)
                self.selections.append(selection)

        # State data
        self.img_idx = 0

    def start(self):
        self.__disable_all()
        self.bind("n", self.keypress_event)

    # def set_stereo_image_datas(self, stereo_image_datas):
    #     self.img_idx = 0
    #     self.stereo_image_datas = stereo_image_datas
        # self.is_done = self.__present_next()
            
    # Return value
    # False: There is more iterations
    # True: The iterations are done
    def __present_next(self):
        # This obtains max self.selection_num
        stereo_image_datas = self.stereo_image_queue.get_any_available_images_up_to(self.selection_num)

        selection_idx = 0
        for stereo_image_data in stereo_image_datas:
            if self.datasaver.check_is_mask_processed(stereo_image_data.frame_id):
                continue
            self.selections[selection_idx].set_context(
                stereo_image_data.frame_id,
                stereo_image_data.left,
                stereo_image_data.right)
            selection_idx += 1

        while selection_idx < self.selection_num:
            self.selections[selection_idx].disable_context(
                self.config.selection_image_height_py * 9 // 16, 
                self.config.selection_image_height_py)
            selection_idx += 1
        return

    def __save_selections(self):
        # print("CTRLabellerApp | Saving selections")
        for selection in self.selections:
            if not selection.is_active:
                continue
            frame_id, image_left, image_right = selection.get_current_context()
            self.datasaver.save_current_stereo_masks(frame_id, image_left, image_right)

    def __disable_all(self):
        for i in range(self.selection_num):
            self.selections[i].disable_context(self.config.selection_image_height_py * 9 // 16, 
                                               self.config.selection_image_height_py)

    def keypress_event(self, input):
        # print(type(input)) # What is tkinter giving here?
        self.__save_selections() # Save the currently presented from __present_next()
        self.__disable_all()

        self.update()
        self.__present_next()
        print("CTRLabellerApp | Presenting next set of images")

class InputPromptGenerationApp(tk.Tk):
    def __init__(self, selection_image_height_py, sam_predictor, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.selection_image_height_py = selection_image_height_py
        self.sam_predictor = sam_predictor
        self.left_input_prompts = None
        self.right_input_prompts = None

    # def generate_masks():
    #     left_image_data = ImageData(batch_data["left_image"][i].cpu().detach().numpy(),
    #                                 batch_data["left_image_name"][i],
    #                                 batch_data["left_image_path"][i])
    #     self.predict_one(left_image_data, left_input_prompts)

    #     pass
