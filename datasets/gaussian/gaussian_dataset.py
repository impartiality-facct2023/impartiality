from functools import partial
import torch
import os
import numpy as np
import PIL
import glob

from torchvision.datasets.vision import VisionDataset

base_folder = "gaussian"


class GaussianDataset(VisionDataset):
    """Synthetic Gaussian Dataset.
    Args:
    root (string): Root directory where images are downloaded to.
    split (string): One of {'train', 'valid', 'test', 'all'}.
        Accordingly dataset is selected.
    target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
        or ``landmarks``. Can also be a list to output a tuple with all specified target types.
        The targets represent:
    transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.ToTensor``
    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    """

    base_folder = "gaussian"

    file_list = [
        # File ID                                                    Filename
        ('gaussian.npy'),
    ]

    def __init__(self, args, root, split="train", transform=None,
                 target_transform=None, download=False):
        import pandas
        super(GaussianDataset, self).__init__(root, transform=transform,
                                             target_transform=target_transform)
        self.split = split
        self.target_feat = 'y'
        fn = partial(os.path.join, self.root, self.base_folder)

        self.num_all_samples = args.num_all_samples
        self.num_train_samples = args.num_train_samples

        # make validation and test have the same number of samples
        self.num_val_samples = args.num_test_samples
        self.num_test_samples = args.num_test_samples

        # define indices for different sets to start
        train_start = 0
        val_start = train_start + self.num_train_samples
        test_start = val_start + self.num_val_samples
        end = test_start + self.num_test_samples

        rng = np.random.default_rng(42)  # use a random state for reproducibility

        # store the actual arrays here
        path = os.path.join(args.data_dir,base_folder,'gaussian.npy')
        with open(path, 'rb') as f:
            x = np.load(f)
            y = np.load(f)
            z = np.load(f)
        y = np.expand_dims(y, axis = 1)
        if split == 'train':
            self.data = torch.as_tensor(x[train_start:val_start]).float()
            self.target = torch.as_tensor(y[train_start:val_start]).float()
            self.sensitive = torch.as_tensor(z[train_start:val_start]).int()
        elif split == 'valid':
            self.data = torch.as_tensor(x[val_start:test_start]).float()
            self.target = torch.as_tensor(y[val_start:test_start]).float()
            self.sensitive = torch.as_tensor(z[val_start:test_start]).int()
        elif split == 'test':
            self.data = torch.as_tensor(x[test_start:]).float().float()
            self.target = torch.as_tensor(y[test_start:]).float()
            self.sensitive = torch.as_tensor(z[test_start:]).int()
        else:
            self.data = torch.as_tensor(x).float()
            self.target = torch.as_tensor(y).float()
            self.sensitive = torch.as_tensor(z).int()

    # get one data point
    def __getitem__(self, index):
        X = self.data[index]
        target = self.target[index]
        sensitive_attribute = self.sensitive[index]

        #target = target.type(torch.LongTensor)
        #sensitive_attribute = sensitive_attribute.type(torch.LongTensor)

        return X, target, sensitive_attribute

    def __len__(self):
        return len(self.target)  # just take a random one of all the lists: all have same length

    # todo: see what it does and it I need to adapt it
    def extra_repr(self):
        lines = ["Target variable: {target_feat}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    import getpass

    user = getpass.getuser()
    root = f'~/data/'
    args = {'num_all_samples': 4000, 'num_train_samples': 2800, 'num_val_samples': 600, 'num_test_samples': 600}
    args = dotdict(args)
    gaussian = GaussianDataset(root=root, args=args, split='train')
    X, target, sens = utkface.__getitem__(0)
    breakpoint()

