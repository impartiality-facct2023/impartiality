import getpass
import os

import numpy as np
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision import datasets, transforms

from datasets.gaussian.gaussian_dataset import GaussianDataset
from queryset import QuerySet

user = getpass.getuser()


def get_gaussian_set(split, args, dataset_path=''):
    """
    Function that enables to load the set with respective split form the gaussian dataset
    """
    dataset_path = args.data_dir
    full_path = os.path.join(os.getcwd(), dataset_path)
    dataset_extractor = GaussianDataset

    all_set = dataset_extractor(
        args=args,
        root=full_path,
        split=split,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
    return all_set


def get_gaussian_train_set(args):
    return get_gaussian_set(split='train', args=args)


def get_gaussian_val_set(args):
    return get_gaussian_set(split='valid', args=args)


def get_gaussian_test_set(args):
    return get_gaussian_set(split='test', args=args)


def get_gaussian_all_set(args):
    return get_gaussian_set(split='all', args=args)


def get_gaussian_private_data(args):
    if args.dataset != 'gaussian':
        return None
    all_private_datasets = get_gaussian_train_set(args)
    private_dataset_size = len(all_private_datasets) // args.num_models
    all_private_trainloaders = []
    for i in range(args.num_models):
        begin = i * private_dataset_size
        if i == args.num_models - 1:
            end = len(all_private_datasets)
        else:
            end = (i + 1) * private_dataset_size
        indices = list(range(begin, end))
        private_dataset = Subset(all_private_datasets, indices)
        kwargs = args.kwargs
        private_trainloader = DataLoader(
            dataset=private_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)
        all_private_trainloaders.append(private_trainloader)
    return all_private_trainloaders


def main():
    # ones_per_example()
    pass


if __name__ == "__main__":
    main()
