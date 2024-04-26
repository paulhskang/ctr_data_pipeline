import numpy as np
import torch
import torchvision

class StereoDataloader(torch.utils.data.IterableDataset):
    def __init__(self) -> None:
        pass

    def load_rgb(self, path, normalize=False):
        img = torchvision.io.read_image(path)
        if normalize:
            img = img.float() / 255.0
        return np.array(img)