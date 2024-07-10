from ctr_labeller.types import ImageData

class ImageSelectorState:
    def __init__(self, draw_height_py, visualize_input_prompt, zoom_factor):
        # constants
        self.c_visualize_input_prompt = visualize_input_prompt
        self.c_draw_height_py = draw_height_py
        self.c_zoom_factor = zoom_factor
        self.c_scaler = 150

        # variable state
        self.is_zoomed = False
        self.current_image_data: ImageData = None
        self.current_image = None
