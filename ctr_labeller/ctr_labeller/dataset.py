import numpy as np
import torch
import torchvision

class StereoDataloader(torch.utils.data.IterableDataset):
    def __init__(self, data) -> None:
        self.data = data


    def __iter__(self):
        return iter(self.data)

    def load_rgb(self, path, normalize=False):
        img = torchvision.io.read_image(path)
        if normalize:
            img = img.float() / 255.0
        return np.array(img)
