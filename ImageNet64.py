from typing import Any, Optional, Callable

from torchvision.datasets import VisionDataset, CIFAR100


class ImageNet64(VisionDataset):

_train_files = ['imagenet/64/']

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None
                 ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)



    def __getitem__(self, index: int) -> Any:
        pass

    def __len__(self) -> int:
        pass