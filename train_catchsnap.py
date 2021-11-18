import argparse
import os
import random
import sys
import time
from typing import Tuple, Protocol

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed import group
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision import models

from arg_utils import parse_args, make_interval_parser, make_half_interval_parser, show_on_action, show_group_on_action, \
    show_group_on_value, parse_cli_args
from augmentation_modules import NormalizeExcludingMask, ColorJitterExcludingMask
from catchsnap import CatchSnap
from conf import settings
from margin_sampling import kmargin_accumulate
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, compute_mean_std


# class TrainArgs(Protocol):
#     batch_size: int
#     gpu: bool
#
# def train(epoch: int, use_kmargin: bool = False):
#     loader = train_loader
#     if use_kmargin:
#         loader = kmargin_accumulate(train_loader, net, args.batch_size, args.gpu)
#     net.train()
#     batch_index = 0
#     start = time.perf_counter()
#     for batch_index, (images, labels) in enumerate(loader):
#         if args.gpu:
#             labels = labels.cuda()
#             images = images.cuda()
#         optimizer.zero_grad()
#         outputs = net(images)
#         loss = loss_function(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#     if use_kmargin:
#
#
#     finish = time.perf_counter()
#     print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

def train_kmargin(epoch):
    start = time.time()
    net.train()
    batch_index = 0
    for batch_index, (images, labels) in enumerate(
            kmargin_accumulate(train_loader, net, args.batch_size, args.gpu, args.k_warm, args.margin)):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('KMargin Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=(batch_index + 1) * args.batch_size,
            total_samples=len(train_loader.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    print('KMargin Training Epoch: {epoch}\t{percentage:.2f}% of samples used...'.format(
        epoch=epoch,
        percentage=((batch_index + 1) * args.batch_size) * 100 / len(train_loader.dataset)) +
          '\nDropped {dropped_samples} out of {total_samples} samples.'.format(
              dropped_samples=len(train_loader.dataset) - ((batch_index + 1) * args.batch_size),
              total_samples=len(train_loader.dataset)
          ))

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


def train(epoch):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(train_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            elif 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=len(train_loader.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        # if epoch <= args.warm:
        #     warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(loader, epoch=0, tb=True, tag='Test'):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('{} set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        tag,
        epoch,
        test_loss / len(loader.dataset),
        correct.float() / len(loader.dataset),
        finish - start
    ))
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar(tag + '/Average loss', test_loss / len(loader.dataset), epoch)
        writer.add_scalar(tag + '/Accuracy', correct.float() / len(loader.dataset), epoch)

    return correct.float() / len(loader.dataset)


def get_network(args) -> nn.Module:
    net_type = {
        'vgg11_bn': models.vgg11_bn,
        'vgg13_bn': models.vgg13_bn,
        'vgg16_bn': models.vgg16_bn,
        'vgg19_bn': models.vgg19_bn,
        'densenet121': models.densenet121,
        'densenet161': models.densenet161,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        'googlenet': models.googlenet,
        'inception_v3': models.inception_v3,
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'wrn50-2': models.wide_resnet50_2,
        'wrn101-2': models.wide_resnet101_2,
    }.get(args.net)
    return net_type(num_classes=200)


CATCHSNAP_MEAN, CATCHSNAP_STD = (0.0524, 0.0473, 0.0391), (0.1711, 0.1561, 0.1354)

if __name__ == '__main__':

    args = parse_cli_args()
    net = get_network(args)
    print(args)

    args_suffix = f"_O{args.optimizer}_B{args.batch_size}_LR{args.lr}" + (f"_K{args.k_warm}_M{args.margin}" if args.kmargin else "")

    optim_from_args = {
        'sgd': optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, dampening=args.dampening,
                         weight_decay=args.weight_decay, nesterov=args.nesterov),
        'adam': optim.Adam(net.parameters(), lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay,
                           amsgrad=args.amsgrad)
    }
    catchsnap_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        NormalizeExcludingMask(CATCHSNAP_MEAN, CATCHSNAP_STD),
        ColorJitterExcludingMask(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    ])

    catchsnap_transform_extreme = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.GaussianBlur(5),
        transforms.RandomPerspective(),
        ColorJitterExcludingMask(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        NormalizeExcludingMask(CATCHSNAP_MEAN, CATCHSNAP_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        NormalizeExcludingMask(CATCHSNAP_MEAN, CATCHSNAP_STD)
    ])

    train_set = CatchSnap('./data/', transform=catchsnap_transform_extreme)
    train_set_unaugmented = CatchSnap('./data/', transform=transform_test)
    test_set = CatchSnap('./data/', train=False, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=3)
    train_loader_unaugmented = DataLoader(train_set_unaugmented, batch_size=args.batch_size, num_workers=3)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=3)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim_from_args[args.optimizer]

    input_tensor = torch.Tensor(1, 3, 512, 512)
    if args.gpu:
        input_tensor = input_tensor.cuda()
        net = net.cuda()
        loss_function = loss_function.cuda()

    iter_per_epoch = len(train_loader)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, 'CATCHSNAP', args.net, 'kmargin' if args.kmargin else 'standard',
        settings.TIME_NOW + args_suffix))
    writer.add_graph(net, input_tensor)
    del input_tensor

    best_acc = 0.0
    train_func = train_kmargin if args.kmargin else train

    print("begin training...")
    acc = eval_training(test_loader, 0)
    acc_train = eval_training(train_loader_unaugmented, 0, tag='Train')
    writer.add_scalar('accuracy_ratio', acc / max(acc_train, 1e-8), 0)

    for epoch in range(1, args.epochs + 1):
        if epoch > args.warm:
            pass

        train_func(epoch)
        acc = eval_training(test_loader, epoch)
        if epoch % 5 == 0:
            acc_train = eval_training(train_loader_unaugmented, epoch, tag='Train')
            writer.add_scalar('accuracy_ratio', acc / max(acc_train, 1e-8), epoch)

    writer.close()
