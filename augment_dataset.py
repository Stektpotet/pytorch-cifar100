import hashlib
import os
import shutil
from typing import Dict

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR100
from torchvision.datasets.utils import calculate_md5
from tqdm import tqdm

from augmentation_modules import ColorJitterExcludingMask


def _unpickle(file) -> Dict:
    import pickle
    with open(file, 'rb') as fo:
        file_dict = pickle.load(fo, encoding='latin1')
    return file_dict


def _pickle(data, file):
    import pickle
    with open(file, 'wb') as fo:
        pickle.dump(data, fo)


def create_augmented_dataset(data_root, train, augmentation_transform, num_augmented_per_sample, dataset_name):

    param_hash = hashlib.md5()
    param_hash.update(f"transform{augmentation_transform}num{num_augmented_per_sample}train{train}".encode('ascii'))
    folder_name = param_hash.hexdigest()

    original_data_path = data_root
    cifar100 = CIFAR100(original_data_path, download=True) # Ensure we have the original dataset in root folder
    augmented_set_name = "train" if train else "test"
    dataset_path = os.path.join(original_data_path, cifar100.base_folder, augmented_set_name)
    augmented_set_path = os.path.join(original_data_path, folder_name, augmented_set_name)
    augmented_set = _unpickle(dataset_path)

    # Repeat the filenames and labels
    augmented_set['filenames'] = augmented_set['filenames'] * num_augmented_per_sample
    augmented_set['fine_labels'] = augmented_set['fine_labels'] * num_augmented_per_sample
    augmented_set['coarse_labels'] = augmented_set['coarse_labels'] * num_augmented_per_sample

    original_images_np = augmented_set['data']  # type: np.ndarray
    images_np = np.tile(original_images_np, (num_augmented_per_sample, 1))  # Repeat the images
    images_np = images_np.reshape((-1, 3, 32, 32))
    images = torch.from_numpy(images_np)

    for i in tqdm(range(len(images)), desc=f"Augmenting samples for {augmented_set_name} set"):
        images[i] = augmentation_transform(images[i])

    aug_images_np = images.numpy().reshape(-1, 3072)
    augmented_set['data'] = aug_images_np

    shutil.move(os.path.join(original_data_path, cifar100.base_folder), os.path.join(original_data_path, folder_name))
    _pickle(augmented_set, augmented_set_path)
    augmented_set_hash = calculate_md5(augmented_set_path)

    hashes = (augmented_set_hash, 'f0ef6b0ae62326f3e7ffdfab6717acfc') if train else ('16019d7e3df5f24257cddd939b257f8d', augmented_set_hash)

    class_string = f"""\
from typing import Optional, Callable
from torchvision.datasets import CIFAR100
import warnings

class {dataset_name}(CIFAR100):
    base_folder = '{folder_name}'
    train_list = [['train', '{hashes[0]}'],]
    test_list = [['test', '{hashes[1]}'],]

    def __init__(self, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(r'{original_data_path}', train, transform, target_transform, False)

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        warnings.warn("The dataset needs to be regenerated! check out 'create_augmented_dataset' in augment_dataset.py")

    """

    with open(dataset_name.lower()+'.py', 'w') as f:
        f.write(class_string)

if __name__ == '__main__':

    augmentation_transform = T.Compose([
        T.RandomCrop(32, padding=3),
        T.RandomHorizontalFlip(),
        T.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1), scale=(0.9, 1.1),
        ),
        ColorJitterExcludingMask(0.4, 0.4, 0.4, 0.04)
    ])
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandomAffine(
            degrees=20,
            translate=(0.1, 0.1), scale=(0.9, 1.1),
        ),
        ColorJitterExcludingMask(0.5, 0.5, 0.5, 0.05)
    ])

    # create_augmented_dataset("./data/augmented", False, augmentation_transform, 10, "CIFAR100_Augmented10")
    # a = CIFAR100_Augmented10(False)
    #
    # img, l = a[0]
    #
    # plt.imshow(img)
    # plt.show()