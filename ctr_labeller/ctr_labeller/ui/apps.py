
from ctr_labeller.config.utils import configure
from ctr_labeller.datasaver import DataSaver
from ctr_labeller.types import StereoImageDataQueue
from ctr_labeller.ui.presenters import StereoImagePresenter

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
