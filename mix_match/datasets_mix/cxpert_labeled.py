import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class CxpertLabeledDataset(Dataset):
    def __init__(self, samples, labels, transform,
                 target_transform=None, index=None):
        """

        Args:
            samples: the labeled samples
            labels: all the multi-labels for the sample
            transform: the transformations for the image
            target_transform: the transformations of the label
        """
        super(CxpertLabeledDataset, self).__init__()
        # index to select label to use.
        self.samples = samples
        if index is not None:
            self.labels = labels[:, index]
            self.labels = self.labels.reshape(-1, 1)
        else:
            self.labels = labels
            self.labels = self.labels.reshape(-1, 1)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.samples[idx], self.labels[idx]
        # print(img.shape)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = torch.tensor(target)

        return img, target
