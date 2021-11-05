import os

import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import ResNet, resnet18
from torchvision.transforms import transforms
from tqdm import tqdm

from utils import compute_mean_std


class TinyImagenet:

    @classmethod
    def mean_std(cls):
        return (-0.0397, -0.1037, -0.2050), (0.5538, 0.5380, 0.5641)

    def __init__(self, root, train_transform, test_transform):
        self.root = root
        self.train_transform = train_transform
        self.test_transform = test_transform

        self._train_set = ImageFolder(os.path.join(root, 'train'), transform=train_transform)
        self._train_set = ImageFolder(os.path.join(root, 'train'), transform=test_transform)
        self._test_set = ImageFolder(os.path.join(root, 'val'), transform=test_transform)
        self.num_classes = 200

    def get_loader(self, train: bool, batch_size: int, num_workers: int):
        return DataLoader(self._train_set if train else self._test_set, batch_size, train, num_workers=num_workers, prefetch_factor=4)


if __name__ == '__main__':

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(64, padding=4),
         transforms.ToTensor(),
         transforms.Normalize(*TinyImagenet.mean_std())])

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*TinyImagenet.mean_std())])

    tiny_imagenet = TinyImagenet('data/tiny-imagenet/', train_transform, test_transform)


    model = resnet18(True).cuda()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = CrossEntropyLoss()

    train_loader = tiny_imagenet.get_loader(True, 128, 4)
    test_loader = tiny_imagenet.get_loader(False, 128, 4)

    for epoch in range(1, 20):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Training epoch #{epoch}"):
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            model_output = model(x)
            loss = loss_fn(model_output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            mean_loss = 0.0
            acc = 0
            i = 0
            for x, y in tqdm(test_loader):
                x = x.cuda()
                y = y.cuda()
                model_output = model(x)
                acc += torch.argmax(model_output).eq(y).sum().item()
                mean_loss += loss_fn(model_output, y).item()
                i += 1
            mean_loss = mean_loss / i
            acc = acc / i
            print(f'#{epoch}\tMean Loss: {mean_loss}')
            print(f'#{epoch}\tAccuracy: {acc}')
