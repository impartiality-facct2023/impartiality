from __future__ import print_function

import os
import shutil
import time
import random

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path().absolute().parent))

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import architectures.wideresnet as models1
from architectures.densenet_pre import densenetpre
import cxpert
from mix_match.utils_mix import Bar, Logger, AverageMeter, accuracy, computeauc, \
    mkdir_p
from mix_match.parameters import get_args
from mix_match.datasets_mix import mix_transforms
from datasets.xray.xray_datasets import XRayCenterCrop
from datasets.xray.xray_datasets import XRayResizer

args = get_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy
best_auc = 0  # best test auc


def create_model(ema=False):
    if args.binary:
        model = densenetpre(num_outputs=1)
    else:
        model = densenetpre()
    # Replace with loading
    model = model.cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def main():
    global best_acc, best_auc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    if args.dataset == "cxpert":
        args.num_classes = 2
        print("Loading cxpert")
        mix_transform_train = transforms.Compose([
            mix_transforms.RandomPadandCrop(224),
            mix_transforms.RandomFlip(),
            mix_transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

        transform_val = transforms.Compose([
            mix_transforms.ToTensor(),
        ])

        train_labeled_sets, train_unlabeled_set, test_set, val_set = cxpert.get_cxpert(
            mix_transform_train=mix_transform_train, args=args)
        # Note train_labeled_sets is a list and we need to iterate over it for each of the labels.
        labeled_trainloaders = []  # List of trainloaders for each of the labels (or only one if binary not used)
        for i in range(len(train_labeled_sets)):
            labeled_trainloader = data.DataLoader(train_labeled_sets[i],
                                                  batch_size=args.batch_size,
                                                  shuffle=True, num_workers=0,
                                                  drop_last=True)
            labeled_trainloaders.append(labeled_trainloader)
        # labeled_trainloader = data.DataLoader(train_labeled_set,
        #                                       batch_size=args.batch_size,
        #                                       shuffle=True, num_workers=0,
        #                                       drop_last=True)
        unlabeled_trainloader = data.DataLoader(train_unlabeled_set,
                                                batch_size=args.batch_size,
                                                shuffle=True, num_workers=0,
                                                drop_last=True)
        val_loader = data.DataLoader(val_set, batch_size=args.batch_size,
                                     shuffle=False, num_workers=0)
        test_loader = data.DataLoader(test_set, batch_size=args.batch_size,
                                      shuffle=False, num_workers=0)

        for i in range(len(train_labeled_sets)):
            # for i in range(4,5): # Only training model 4
            print(f"Traning Model {i}")
            labeled_trainloader = labeled_trainloaders[i]

            if args.dataset == "cxpert":
                print("Loading pretrained densenet")

            model = create_model()
            ema_model = create_model(ema=True)

            cudnn.benchmark = True
            print('    Total params: %.2fM' % (
                    sum(p.numel() for p in model.parameters()) / 1000000.0))

            train_criterion = SemiLoss()
            if args.binary:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
            start_epoch = 0

            # Resume
            if args.dataset == 'cxpert':
                title = 'noisy-cxpert'

            if args.resume:
                # Load checkpoint.
                print('==> Resuming from checkpoint..')
                assert os.path.isfile(
                    args.resume), 'Error: no checkpoint directory found!'
                args.out = os.path.dirname(args.resume)
                checkpoint = torch.load(args.resume)
                best_acc = checkpoint['best_acc']
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                ema_model.load_state_dict(checkpoint['ema_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger = Logger(os.path.join(args.out, 'log.txt'), title=title,
                                resume=True)
            else:
                logger = Logger(os.path.join(args.out, f'log{i}.txt'),
                                title=title)
                logger.set_names(
                    ['Train Loss', 'Train Loss X', 'Train Loss U', 'Valid Loss',
                     'Valid Acc.', 'Test Loss', 'Test Acc.', 'Test Auc.'])

            step = 0
            test_accs = []
            # Train and val
            for epoch in range(start_epoch, args.epochs):
                print(
                    '\nEpoch: [%d | %d] LR: %f' % (
                    epoch + 1, args.epochs, state['lr']))

                train_loss, train_loss_x, train_loss_u = train(
                    labeled_trainloader,
                    unlabeled_trainloader,
                    model, optimizer,
                    ema_optimizer,
                    train_criterion, epoch,
                    use_cuda)
                _, train_acc = validate(labeled_trainloader, ema_model,
                                        criterion,
                                        epoch, use_cuda, mode='Train Stats')
                if args.binary:
                    val_loss, val_acc = validate(val_loader, ema_model,
                                                 criterion, epoch,
                                                 use_cuda, mode='Valid Stats',
                                                 index=i)
                    test_loss, test_acc, test_auc = validate(test_loader,
                                                             ema_model,
                                                             criterion, epoch,
                                                             use_cuda,
                                                             mode='Test Stats',
                                                             index=i)
                else:
                    val_loss, val_acc = validate(val_loader, ema_model,
                                                 criterion,
                                                 epoch,
                                                 use_cuda, mode='Valid Stats',
                                                 )
                    test_loss, test_acc, test_auc = validate(test_loader,
                                                             ema_model,
                                                             criterion,
                                                             epoch,
                                                             use_cuda,
                                                             mode='Test Stats ',
                                                             )

                step = args.train_iteration * (epoch + 1)

                # append logger file
                logger.append(
                    [train_loss, train_loss_x, train_loss_u, val_loss, val_acc,
                     test_loss, test_acc, test_auc])

                # save model
                # is_best = val_acc > best_acc
                is_best = test_auc > best_auc
                best_acc = max(val_acc, best_acc)
                best_auc = max(test_auc, best_auc)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'acc': val_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best, filename=f'checkpoint-model({i}).pth.tar')
                try:
                    test_accs.append(test_acc.cpu())  # test_acc
                except:
                    test_accs.append(test_acc)
            logger.close()

            print('Best acc:')
            print(best_acc)

            print('Best auc:')
            print(best_auc)

            print('Mean acc:')
            print(np.mean(test_accs[-20:]))


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
          ema_optimizer, criterion, epoch, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()  # xb, pb
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()
        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()  # ubhat
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(
                non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabeled samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            if args.binary:
                p = (torch.sigmoid(outputs_u) + torch.sigmoid(outputs_u2)) / 2
                targets_u = p  # Dont need temperature sharpening for binary classification
            else:
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2,
                                                                     dim=1)) / 2  # qb (before qb.cpu called)
                pt = p ** (1 / args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)  # qb after total

            targets_u = targets_u.detach()  # removes tracking of gradients for this

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        # we have one variable which contains both inputs and targets (W)

        # Transform label to one-hot
        if args.binary == False:
            targets_x = torch.zeros(batch_size, args.num_classes).scatter_(
                1, targets_x.view(-1, 1).long(), 1)

        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1 - l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct
        # batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)
        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u,
                              mixed_target[batch_size:],
                              epoch + batch_idx / args.train_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
            batch=batch_idx + 1,
            size=args.train_iteration,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            w=ws.avg,
        )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg,)


def validate(valloader, model, criterion, epoch, use_cuda, mode, index=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    outputscomb = None
    targetscomb = None
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            if mode in ['Valid Stats',
                        'Test Stats'] and args.dataset == "cxpert":
                inputs = inputs.repeat(1, 3, 1, 1)
                targets = targets[:, index]
                targets = targets.reshape(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1 = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            if outputscomb == None:
                outputscomb = outputs
                targetscomb = targets
            else:
                outputscomb = torch.cat((outputscomb, outputs), dim=0)
                targetscomb = torch.cat((targetscomb, targets), dim=0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | acc: {top1: .4f}'.format(
                batch=batch_idx + 1,
                size=len(valloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
            )
            bar.next()
        bar.finish()
    auc = computeauc(outputscomb, targetscomb)
    if mode == 'Test Stats':
        return (losses.avg, top1.avg, auc)
    else:
        return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint=args.out,
                    filename='checkpoint.pth.tar'):
    # filepath = os.path.join(checkpoint, filename)
    # torch.save(state, filepath)
    if is_best:
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        # shutil.copyfile(filepath,
        #                 os.path.join(checkpoint, 'model_best.pth.tar'))


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        if args.binary:
            probs_u = torch.sigmoid(outputs_u)  # Not sure about this
            Lx = -torch.mean(
                torch.sum(torch.log(torch.sigmoid(outputs_x)) * targets_x,
                          dim=1) +
                torch.sum(
                    torch.log(torch.sigmoid(1 - outputs_x)) * (1 - targets_x),
                    dim=1))  # BCE
        else:
            probs_u = torch.softmax(outputs_u, dim=1)
            Lx = -torch.mean(
                torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


if __name__ == '__main__':
    main()
