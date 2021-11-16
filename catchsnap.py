import os
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

class CatchSnap(ImageFolder):

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader,
                 is_valid_file=None):
        self._root_dir = os.path.join(root, "catchsnap", "split")
        subset_dir = os.path.join(self._root_dir, 'train' if train else 'test')
        super(CatchSnap, self).__init__(subset_dir, transform, target_transform, loader, is_valid_file)
