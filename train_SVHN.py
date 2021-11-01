
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed import group
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.datasets import SVHN

from arg_utils import make_interval_parser
from conf import settings
from qmargin_sampling import qmargin_accumulate
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, compute_mean_std


def train_qmargin(epoch):

    start = time.time()
    net.train()
    batch_index = 0
    for batch_index, (images, labels) in enumerate(qmargin_accumulate(train_loader, net, args.b, args.gpu, args.quantile, args.margin)):
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

        print('QMargin Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=(batch_index + 1) * args.b,
            total_samples=len(train_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        # if epoch <= args.warm:
            # warmup_scheduler.step()

    print('QMargin Training Epoch: {epoch}\t{percentage:.2f}% of samples used...'.format(
            epoch=epoch,
            percentage=((batch_index+1) * args.b)*100/len(train_loader.dataset)) +
          '\nDropped {dropped_samples} out of {total_samples} samples.'.format(
            dropped_samples=len(train_loader.dataset) - ((batch_index+1) * args.b),
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
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(train_loader.dataset)
        ))

        #update training loss for each iteration
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

    #add informations to tensorboard
    if tb:
        writer.add_scalar(tag+'/Average loss', test_loss / len(loader.dataset), epoch)
        writer.add_scalar(tag+'/Accuracy', correct.float() / len(loader.dataset), epoch)

    return correct.float() / len(loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('-warm', type=int, default=-1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--qmargin', action='store_true', default=False, help=argparse.SUPPRESS)


    # ================================================ QMargin Options ================================================
    parse_quantile = make_interval_parser(0, 1, lower_closed=True, upper_closed=True)
    parse_margin = make_interval_parser(0, 1, lower_closed=True, upper_closed=False)
    qmargin_parser = subparsers.add_parser('qmargin', help='use qmargin selection')
    qmargin_parser.add_argument('--qmargin', action='store_true', default=True, help=argparse.SUPPRESS)
    qmargin_parser.add_argument('-q', '--quantile', type=parse_quantile, default=1.0)
    qmargin_parser.add_argument('-m', '--margin', type=parse_margin, default=0.4)

    args = parser.parse_args()

    net = get_network(args)


    #data preprocessing:

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(settings.SVHN_TRAIN_MEAN, settings.SVHN_TRAIN_STD)
    ])

    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transform_test
    ])

    train_loader = DataLoader(SVHN('./data', "train", transform_train, download=True),
                              batch_size=args.b, shuffle=True, num_workers=4)

    train_unaugmented_loader = DataLoader(SVHN('./data', "train", transform_test, download=True),
                              batch_size=args.b, shuffle=True, num_workers=4)

    test_loader = DataLoader(SVHN('./data', "test", transform_test, download=True),
                              batch_size=args.b, shuffle=True, num_workers=4)


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    # print(optimizer.param_groups)
    # exit()
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(train_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, 'SVHN', args.net, 'qmargin' if args.qmargin else 'standard', settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(test_loader, tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    train_func = train_qmargin if args.qmargin else train

    for epoch in range(1, args.epochs + 1):
        if epoch > args.warm:
            # train_scheduler.step()
            pass

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train_func(epoch)
        acc = eval_training(test_loader, epoch)
        if epoch % 5 == 0:
            eval_training(train_unaugmented_loader, epoch, tag='Train')

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()