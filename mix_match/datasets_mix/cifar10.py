import numpy as np
import torchvision

from mix_transforms import TransformTwice
from mix_transforms import normalise
from mix_transforms import transpose


def get_cifar10(root, n_labeled,
                transform_train=None, transform_val=None,
                download=True):
    base_dataset = torchvision.datasets.CIFAR10(root, train=True,
                                                download=download)
    test_dataset = torchvision.datasets.CIFAR10(root, train=False,
                                                download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(
        test_dataset.targets, int(n_labeled / 10))  # was base_dataset.target

    train_labeled_dataset = CIFAR10_labeledmod(root, train_labeled_idxs,
                                               train=False,
                                               transform=transform_train)  # train=True
    train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs,
                                                train=False,
                                                transform=TransformTwice(
                                                    transform_train))  # train = True. possibly modify the transforms
    val_dataset = CIFAR10_labeled(root, val_idxs, train=True,
                                  transform=transform_val,
                                  download=True)  # possibly change train = False here and above
    test_dataset = CIFAR10_labeled(root, val_idxs, train=False,
                                   transform=transform_val,
                                   download=True)  # might be good to use seperate indices than the ones in train_labeled i.e use val_idxs here too
    # print("Test shape", train_labeled_dataset[0][0].shape)
    print(
        f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    # To get an equal number of samples per class.
    # for i in range(10):
    #     idxs = np.where(labels == i)[0]
    #     np.random.shuffle(idxs)
    #     train_labeled_idxs.extend(idxs[:n_labeled_per_class])
    #     train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
    #     val_idxs.extend(idxs[-500:])

    # Random selection for point:
    n_labeled = n_labeled_per_class * 10
    idxs = np.where(labels < 10)[0]  # All points
    np.random.shuffle(idxs)
    train_labeled_idxs.extend(idxs[:n_labeled])
    train_unlabeled_idxs.extend(
        idxs[n_labeled: -1000])  # -500 here and below originally
    val_idxs.extend(idxs[-1000:])
    ent = 0
    gap = 0
    temp1 = np.load("cifarent.npy")
    temp2 = np.load("cifargap.npy")
    for i in train_labeled_idxs:
        ent += temp1[i]
        gap += temp2[i]
    # pknn = 0
    total = n_labeled_per_class * 10
    file = f"cifar10@{total}new/stats.txt"
    f = open(file, "w")
    f.write("Entropy: " + str(ent) + "\n")
    f.write("Gap: " + str(gap) + "\n")
    f.close()
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


cifar10_mean = (0.4914, 0.4822,
                0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (
    0.2471, 0.2435,
    0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                                              transform=transform,
                                              target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalise(self.data, mean=cifar10_mean,
                                        std=cifar10_std))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # print(img.shape)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_labeledmod(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=False,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeledmod, self).__init__(root, train=train,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            # victim = load_private_model_by_id()
            # temp = []
            # for i in indexs:
            # self.targets = victim(self.data)
            # Use model predictions here?
            targets = np.load("cifartargets.npy")
            self.targets = np.array(targets)[indexs]
            # print(self.targets)
        self.data = transpose(normalise(self.data, mean=cifar10_mean,
                                        std=cifar10_std))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # print(img.shape)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                                                transform=transform,
                                                target_transform=target_transform,
                                                download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])
