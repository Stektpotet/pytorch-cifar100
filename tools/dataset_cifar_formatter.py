from typing import List, Union

import numpy as np
from PIL.Image import Image

def to_cifar_format(images: Union[List[Image], List[np.ndarray], np.ndarray]):
    if isinstance(images, List):
        if isinstance(images[0], Image):
            images = [np.array(img) for img in images]
        images = np.stack(images)