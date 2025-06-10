# Copyright (c) 2025, Wilfred and Joyce Posluns Centre for Guided Innovation and Therapeutic Intervention (PCIGITI), University of Toronto
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
 
from dataclasses import dataclass, field
from queue import Queue
from typing import List
import numpy as np
import threading

@dataclass 
class PredictionOutput:
    input_prompt: dict
    mask: np.ndarray = None
    masked_image: np.ndarray = None
    score: float = 0.0
    area_ratio: float = 0.0

@dataclass
class ImageData:
    # Loading
    image: np.ndarray
    name: str
    path: str
    frame_id: int
    # Sam Processing
    current_mask_idx: int = -1
    prediction_outputs: List[PredictionOutput] = field(default_factory=lambda: [])
    
    # UI Processing
    is_save_mask: bool = False

@dataclass 
class StereoImageData:
    frame_id: str
    left: ImageData
    right: ImageData

def convert_mask_torch_to_opencv(sam_mask):
    h, w = sam_mask.shape[-2:]
    return sam_mask.reshape(h, w, 1) * 255

class StereoImageDataQueue:
    def __init__(self, max_size_to_add):
        self.queue = Queue(maxsize=max_size_to_add*2)
        self.add_cv = threading.Condition()
        self.max_size_to_add = max_size_to_add

    # Assumes List is not that large of a size at a time. i.e. > 100
    def wait_add_images(self, stereo_images: List[StereoImageData]):
        with self.add_cv:
            self.add_cv.wait_for(lambda : self.queue.qsize() < self.max_size_to_add) # Should add or sigint handler
            for stereo_image in stereo_images:
                self.queue.put(stereo_image)
            self.add_cv.notify_all()

    def wait_any_available_images_up_to(self, size_to_get):
        with self.add_cv:
            self.add_cv.wait_for(lambda : self.queue.qsize() >= size_to_get) # Should add or sigint handler
        return self.get_any_available_images_up_to(size_to_get)


    def get_any_available_images_up_to(self, max_size_to_get):
        stereo_images_return = []
        with self.add_cv:
            i = 0
            while i < max_size_to_get and self.queue.qsize() > 0:
                stereo_images_return.append(self.queue.get())
                i += 1
            if self.queue.qsize() < self.max_size_to_add:
                self.add_cv.notify_all()
        return stereo_images_return
