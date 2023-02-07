# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from calculate_metrics import calc_auroc, calc_tnr
from generate_loaders import Normalizer, GaussianLoader, UniformLoader
from godin.parameters import get_args
from nets.deconfnet import DeconfNet, CosineDeconf, InnerDeconf, EuclideanDeconf
from nets.densenet import DenseNet
from nets.resnet import ResNet34
from nets.wideresnet import WideResNet

r_mean = 125.3 / 255
g_mean = 123.0 / 255
b_mean = 113.9 / 255
r_std = 63.0 / 255
g_std = 62.1 / 255
b_std = 66.7 / 255

train_transform_cifar10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform_cifar10 = transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_transform_cifar100 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ]
)
test_transform_cifar100 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
    ])

h_dict = {
    'cosine': CosineDeconf,
    'inner': InnerDeconf,
    'euclid': EuclideanDeconf
}

losses_dict = {
    'ce': nn.CrossEntropyLoss(),
}

generating_loaders_dict = {
    'Gaussian': GaussianLoader,
    'Uniform': UniformLoader
}


def get_datasets(data_dir, data_name, batch_size, data_in, num_workers):
    if data_in == 'CIFAR10':
        num_classes = 10
        train_set_in = torchvision.datasets.CIFAR10(
            root=f'{data_dir}/cifar10',
            train=True, download=True,
            transform=train_transform_cifar10)
        test_set_in = torchvision.datasets.CIFAR10(
            root=f'{data_dir}/cifar10',
            train=False, download=True,
            transform=test_transform_cifar10)
    elif data_in == 'CIFAR100':
        num_classes = 100
        train_set_in = torchvision.datasets.CIFAR10(
            root=f'{data_dir}/cifar100',
            train=True, download=True,
            transform=train_transform_cifar100)
        test_set_in = torchvision.datasets.CIFAR10(
            root=f'{data_dir}/cifar100',
            train=False, download=True,
            transform=test_transform_cifar100)
    else:
        raise Exception(f"Unknown in-distribution dataset: {data_in}.")

    if data_name == 'Gaussian' or data_name == 'Uniform':
        normalizer = Normalizer(r_mean, g_mean, b_mean, r_std, g_std, b_std)
        outlier_loader = generating_loaders_dict[data_name](
            batch_size=batch_size, num_batches=int(10000 / batch_size),
            transformers=[normalizer])
    elif data_name == 'Imagenet':
        outlier_set = torchvision.datasets.ImageFolder(
            f'{data_dir}/{data_name}', transform=test_transform_cifar10)
        outlier_loader = DataLoader(outlier_set, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    elif data_name == 'SVHN':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        trainset = torchvision.datasets.SVHN(
            root=f'{data_dir}/{data_name}', split="train", download=True,
            transform=transform
        )
        kwargs = {"num_workers": num_workers, "pin_memory": True}
        # train_loader = torch.utils.data.DataLoader(
        #     trainset, batch_size=batch_size, shuffle=True, **kwargs
        # )
        testset = torchvision.datasets.SVHN(
            root=f'{data_dir}/{data_name}', split="test", download=True,
            transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=True, **kwargs
        )
        outlier_loader = test_loader
    else:
        raise Exception(f"Unsupported OOD dataset: {data_name}.")

    test_indices = list(range(len(test_set_in)))
    validation_set_in = Subset(test_set_in, test_indices[:1000])
    test_set_in = Subset(test_set_in, test_indices[1000:])

    train_loader_in = DataLoader(train_set_in, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)
    validation_loader_in = DataLoader(validation_set_in, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers)
    test_loader_in = DataLoader(test_set_in, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

    return train_loader_in, validation_loader_in, test_loader_in, outlier_loader, num_classes


def main():
    args = get_args()

    device = args.gpu

    load_model = args.load_model
    model_dir = args.model_dir

    architecture = args.architecture
    similarity = args.similarity
    loss_type = args.loss_type

    data_dir = args.data_dir
    data_name = args.out_dataset
    data_in = args.in_dataset
    batch_size = args.batch_size

    train = args.train
    weight_decay = args.weight_decay
    epochs = args.epochs

    test = args.test
    noise_magnitudes = args.magnitudes

    # Create necessary directories
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_data, val_data, test_data, open_data, num_classes = get_datasets(
        data_dir, data_name, batch_size, data_in, num_workers=args.num_workers)

    if architecture == 'densenet':
        underlying_net = DenseNet(num_classes=num_classes)
    elif architecture == 'resnet':
        underlying_net = ResNet34(num_classes=num_classes)
    elif architecture == 'wideresnet':
        underlying_net = WideResNet(num_classes=num_classes)
    else:
        raise Exception(f"Unsupported architecture: {architecture}.")

    underlying_net.to(device)

    # Construct g, h, and the composed deconf net
    baseline = (similarity == 'baseline')

    if baseline:
        h = InnerDeconf(in_features=underlying_net.output_size,
                        num_classes=num_classes)
    else:
        h = h_dict[similarity](in_features=underlying_net.output_size,
                               num_classes=num_classes)

    h.to(device)

    deconf_net = DeconfNet(underlying_net, underlying_net.output_size,
                           num_classes=num_classes, h=h, baseline=baseline)

    deconf_net.to(device)

    parameters = []
    h_parameters = []
    for name, parameter in deconf_net.named_parameters():
        if name == 'h.h.weight' or name == 'h.h.bias':
            h_parameters.append(parameter)
        else:
            parameters.append(parameter)

    optimizer = optim.SGD(parameters, lr=0.1, momentum=0.9,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(epochs * 0.5), int(epochs * 0.75)],
        gamma=0.1)

    h_optimizer = optim.SGD(
        h_parameters, lr=0.1, momentum=0.9, weight_decay=0)  # No weight decay
    h_scheduler = optim.lr_scheduler.MultiStepLR(
        h_optimizer, gamma=0.1,
        milestones=[int(epochs * 0.5), int(epochs * 0.75)], )

    # Load the model (capable of resuming training or inference)
    # from the checkpoint file

    suffix = f"-{similarity}-{architecture}"
    file_name = f'{model_dir}/checkpoint{suffix}.pth'
    if load_model:
        print(f'Loading model: {file_name}.')
        checkpoint = torch.load(file_name)

        epoch_start = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        h_optimizer.load_state_dict(checkpoint['h_optimizer'])
        deconf_net.load_state_dict(checkpoint['deconf_net'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        h_scheduler.load_state_dict(checkpoint['h_scheduler'])
        epoch_loss = checkpoint['epoch_loss']
    else:
        print('Starting from scratch.')
        epoch_start = 0
        epoch_loss = None

    criterion = losses_dict[loss_type]

    if train:
        print('Train the model.')
        deconf_net.train()

        num_batches = len(train_data)
        epoch_bar = tqdm(total=num_batches * epochs,
                         initial=num_batches * epoch_start)

        lowest_loss = None
        for epoch in range(epoch_start, epochs):
            total_loss = 0.0
            num_samples = 0
            correct = 0
            for batch_idx, (inputs, targets) in enumerate(train_data):
                if epoch_loss is None:
                    epoch_bar.set_description(
                        f'Training | Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{num_batches}')
                else:
                    epoch_bar.set_description(
                        f'Training | Epoch {epoch + 1}/{epochs} | Epoch loss = {epoch_loss:0.2f} | Batch {batch_idx + 1}/{num_batches}')
                inputs = inputs.to(device)
                targets = targets.to(device)
                h_optimizer.zero_grad()
                optimizer.zero_grad()

                logits, _, _ = deconf_net(inputs)
                loss = criterion(logits, targets)
                loss.backward()

                optimizer.step()
                h_optimizer.step()
                total_loss += loss.item()

                epoch_bar.update()

                predictions = torch.argmax(logits, dim=1)
                num_samples += len(logits)
                correct += (predictions == targets).int().sum()

            epoch_loss = total_loss
            h_scheduler.step()
            scheduler.step()

            accuracy = 100 * correct / num_samples
            print(f'epoch: {epoch}, accuracy: {accuracy}')

            if lowest_loss is None or epoch_loss < lowest_loss:
                lowest_loss = epoch_loss

                checkpoint = {
                    'epoch': epoch + 1,
                    'optimizer': optimizer.state_dict(),
                    'h_optimizer': h_optimizer.state_dict(),
                    'deconf_net': deconf_net.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'h_scheduler': h_scheduler.state_dict(),
                    'epoch_loss': epoch_loss,
                    'accuracy': accuracy,
                }
                # For continuing training or inference
                torch.save(checkpoint,
                           file_name + f'-epoch-{epoch}-accuracy-{accuracy}-loss-{loss}')
                # For exporting / sharing / inference only
                torch.save(deconf_net.state_dict(),
                           f'{model_dir}/model{suffix}-{epoch}-{accuracy}.pth')

        if epoch_loss is None:
            epoch_bar.set_description(
                f'Training | Epoch {epochs}/{epochs} | Batch {num_batches}/{num_batches}')
        else:
            epoch_bar.set_description(
                f'Training | Epoch {epochs}/{epochs} | Epoch loss = {epoch_loss:0.2f} | Batch {num_batches}/{num_batches}')
        epoch_bar.close()

    if test:
        print('Test the model.')
        deconf_net.eval()
        best_val_score = None
        best_auroc = None

        # score_functions = ['h', 'g', 'logit']
        score_functions = ['h']
        for score_func in score_functions:
            print(f'Score function: {score_func}')
            for noise_magnitude in noise_magnitudes:
                print(f'Noise magnitude {noise_magnitude:.5f}         ')

                id_val_results = generate_scores(
                    deconf_net, device, val_data, noise_magnitude,
                    score_func, title='Validating')
                validation_results = np.average(id_val_results)
                print('average id_val scores: ', validation_results)

                id_test_results = generate_scores(
                    deconf_net, device, test_data, noise_magnitude,
                    score_func, title='Testing ID')
                print('average id_test scores: ', np.average(id_test_results))

                ood_test_results = generate_scores(
                    deconf_net, device, open_data, noise_magnitude,
                    score_func, title='Testing OOD')
                print('average ood scores: ', np.average(ood_test_results))

                print('# of id_test_results: ', len(id_test_results))
                print('# of ood_test_results: ', len(ood_test_results))

                auroc = calc_auroc(id_test_results, ood_test_results) * 100
                tnrATtpr95 = calc_tnr(id_test_results, ood_test_results)
                print('AUROC:', auroc, 'TNR@TPR95:', tnrATtpr95)

                if best_auroc is None:
                    best_auroc = auroc
                else:
                    best_auroc = max(best_auroc, auroc)
                if best_val_score is None or validation_results > best_val_score:
                    best_val_score = validation_results
                    best_val_auroc = auroc
                    best_tnr = tnrATtpr95

        print('best auroc: ', best_val_auroc, ' and tnr@tpr95 ', best_tnr)
        print('true best auroc:', best_auroc)


def generate_scores(model, CUDA_DEVICE, data_loader, noise_magnitude,
                    score_func='h', title='Testing'):
    model.eval()
    num_batches = len(data_loader)
    results = []
    data_iter = tqdm(data_loader)
    num_examples = 0
    correct = 0
    for j, (images, targets) in enumerate(data_iter):
        data_iter.set_description(
            f'{title} | Processing image batch {j + 1}/{num_batches}')
        images = Variable(images.to(CUDA_DEVICE), requires_grad=True)
        targets = targets.to(CUDA_DEVICE)

        logits, h, g = model(images)

        num_examples += len(logits)
        predicted = torch.argmax(logits, dim=1)
        correct += (predicted == targets).int().sum()

        if score_func == 'h':
            scores = h
        elif score_func == 'g':
            scores = g
        elif score_func == 'logit':
            scores = logits

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of the numerator w.r.t. input

        max_scores, _ = torch.max(scores, dim=1)
        max_scores.backward(torch.ones(len(max_scores)).to(CUDA_DEVICE))

        # Normalizing the gradient to binary in {-1, 1}
        if images.grad is not None:
            gradient = torch.ge(images.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            gradient[::, 0] = (gradient[::, 0]) / (63.0 / 255.0)
            gradient[::, 1] = (gradient[::, 1]) / (62.1 / 255.0)
            gradient[::, 2] = (gradient[::, 2]) / (66.7 / 255.0)
            # Adding small perturbations to images
            tempInputs = torch.add(images.data, gradient, alpha=noise_magnitude)

            # Now calculate score
            logits, h, g = model(tempInputs)

            if score_func == 'h':
                scores = h
            elif score_func == 'g':
                scores = g
            elif score_func == 'logit':
                scores = logits

        results.extend(torch.max(scores, dim=1)[0].data.cpu().numpy())

    data_iter.set_description(
        f'{title} | Processing image batch {num_batches}/{num_batches}')

    accuracy = 100 * correct / num_examples
    print(f'accuracy: {accuracy}')

    data_iter.close()

    return np.array(results)


if __name__ == '__main__':
    main()
