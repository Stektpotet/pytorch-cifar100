import math
import os

import numpy as np
from torch import  nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder


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
        return DataLoader(self._train_set if train else self._test_set, batch_size, train, num_workers=num_workers)


def load_data_npz(input_file):
    with np.load(input_file) as data_dict:
        x = data_dict['data']
        y = data_dict['labels']

    channel_size = x[0].shape[0] // 3
    img_size = int(round(math.sqrt(channel_size)))

    x = np.dstack((x[:, :channel_size], x[:, channel_size:channel_size*2], x[:, channel_size*2:]))
    x = x.reshape((-1, img_size, img_size, 3))
    return x, y

def get_network(net: str) -> nn.Module:
    return {
        'vgg11_bn':     models.vgg11_bn(num_classes=200),
        'vgg13_bn':     models.vgg13_bn(num_classes=200),
        'vgg16_bn':     models.vgg16_bn(num_classes=200),
        'vgg19_bn':     models.vgg19_bn(num_classes=200),
        'densenet121':  models.densenet121(num_classes=200),
        'densenet161':  models.densenet161(num_classes=200),
        'densenet169':  models.densenet169(num_classes=200),
        'densenet201':  models.densenet201(num_classes=200),
        'googlenet':    models.googlenet(num_classes=200),
        'inception_v3': models.inception_v3(num_classes=200),
        'resnet18':     models.resnet18(num_classes=200),
        'resnet34':     models.resnet34(num_classes=200),
        'resnet50':     models.resnet50(num_classes=200),
        'resnet101':    models.resnet101(num_classes=200),
        'resnet152':    models.resnet152(num_classes=200),
        'wrn50-2':      models.wide_resnet50_2(num_classes=200),
        'wrn101-2':     models.wide_resnet101_2(num_classes=200),
    }.get(net)


if __name__ == '__main__':
    pass
    # args = parse_cli_args()
    # net = get_network(args)

    # train_transform = transforms.Compose(
    #     [transforms.RandomHorizontalFlip(),
    #      transforms.RandomCrop(64, padding=4),
    #      transforms.ToTensor(),
    #      transforms.Normalize(*TinyImagenet.mean_std())])
    #
    # test_transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(*TinyImagenet.mean_std())])
    #
    # tiny_imagenet = TinyImagenet('data/tiny-imagenet/', train_transform, test_transform)
    #
    # model = models.wide_resnet50_2(pretrained=False, progress=True, num_classes=200).cuda()
    # model = resnet18(pretrained=False, progress=True, num_classes=200).cuda()
    # optimizer = optim.Adam(model.parameters(), lr=5e-4)
    # loss_fn = nn.CrossEntropyLoss().cuda()
    #
    # train_loader = tiny_imagenet.get_loader(True, 100, num_workers=4)
    # test_loader = tiny_imagenet.get_loader(False, 1000, num_workers=2)
    #
    # for epoch in range(1, 20):
    #     model.train()
    #     for x, y in tqdm(train_loader, desc=f"Training epoch #{epoch}"):
    #         x = x.cuda()
    #         y = y.cuda()
    #         optimizer.zero_grad()
    #         model_output = model(x)
    #         loss = loss_fn(model_output, y)
    #         loss.backward()
    #         optimizer.step()
    #
    #     model.eval()
    #     with torch.no_grad():
    #         mean_loss = 0.0
    #         acc = 0
    #         i = 0
    #         for x, y in tqdm(test_loader):
    #             x = x.cuda()
    #             y = y.cuda()
    #             model_output = model(x)
    #             acc += torch.argmax(model_output, dim=1).eq(y).sum().item() / len(y)
    #             mean_loss += loss_fn(model_output, y).item()
    #             i += 1
    #         mean_loss = mean_loss / i
    #         acc = acc / i
    #         print(f'#{epoch}\tMean Loss: {mean_loss}')
    #         print(f'#{epoch}\tAccuracy: {acc}')
