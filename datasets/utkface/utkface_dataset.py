from functools import partial
import torch
import os
import numpy as np
import PIL
import glob

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive
from datasets.utkface.utkface_labels import infer_information_from_filename

base_folder = "utkface"


class UTKfaceDataset(VisionDataset):
    """`UTKface Dataset https://susanqq.github.io/UTKFace/`_ Dataset.
    Args:
    root (string): Root directory where images are downloaded to.
    split (string): One of {'train', 'valid', 'test', 'all'}.
        Accordingly dataset is selected.
    target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
        or ``landmarks``. Can also be a list to output a tuple with all specified target types.
        The targets represent:
            ``age``
            ``gender``
            ``race``
        Defaults to ``age``. If empty, ``None`` will be returned as target.
    transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.ToTensor``
    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "utkface"

    file_list = [
        
        ('0BxYys69jI14kU0I1YUQyY1ZDRUE', 'UTKFace.tar.gz'),
    ]

    def __init__(self, args, root, split="train", target_feat="gender", sensitive_feat="race", transform=None,
                 target_transform=None, download=False):
        import pandas
        super(UTKfaceDataset, self).__init__(root, transform=transform,
                                             target_transform=target_transform)
        self.split = split

        if download:
            self.download()

        fn = partial(os.path.join, self.root, self.base_folder)

        
        
        self.num_all_samples = args.num_all_samples
        self.num_train_samples = args.num_train_samples
        self.num_dev_samples = args.num_dev_samples
        self.num_test_samples = args.num_test_samples

        
        train_start = 0
        
        test_start = train_start + self.num_train_samples
        end = test_start + self.num_test_samples + args.num_unlabeled_samples

        rng = np.random.default_rng(42)  
        all_files = [file for file in os.listdir(fn("UTKFace/")) if file.endswith(".jpg")]  

        
        
        all_files = [file for file in all_files if file.count('_') == 3]
        rng.shuffle(all_files)  

        if split == 'train':
            self.filename = all_files[train_start:test_start]
        
            
        elif split == 'test':
            self.filename = all_files[test_start:end]
        elif split == 'all':
            self.filename = all_files
        else:
            raise ValueError("Please provide a valid split {'train', 'valid', 'test', 'all'}")

        assert len(self.filename) == len(
            set(self.filename)), "Your sampling has sampled the same datapoint more than once"

        information = [infer_information_from_filename(fn) for fn in
                       self.filename]  
        information = np.array(information, dtype=np.float32)  

        self.target_feat = target_feat
        self.sensitive_feat = sensitive_feat

        self.age = torch.as_tensor(information[:, 0])
        self.gender = torch.as_tensor(information[:, 1])
        self.race = torch.as_tensor(information[:, 2])

    def download(self):
        
        import zipfile

        
        
        for (file_id, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root,
                                                                  self.base_folder),
                                            filename)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder,
                                          "UTKFace.tar.gz"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    
    def __getitem__(self, index):
        X = PIL.Image.open(
            os.path.join(self.root, self.base_folder, "UTKFace",
                         self.filename[index])).convert('RGB')
        targets = []
        for t in [self.target_feat, self.sensitive_feat]:
            if t == "age":
                targets.append(self.age[index])
            elif t == "gender":
                targets.append(self.gender[index])
            elif t == "race":
                targets.append(self.race[index])
            else:
                
                raise ValueError(
                    "Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        target = targets[0]
        sensitive_attribute = targets[1]

        if self.target_transform is not None:
            target = self.target_transform(target)
            
            target = target.type(torch.LongTensor)

        sensitive_attribute = sensitive_attribute.type(torch.LongTensor)

        return X, target, sensitive_attribute

    def __len__(self):
        return len(self.age)  

    
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
    root = os.path.join(os.getcwd(), f'../../data/')
    args = {'num_all_samples': 23708, 'num_train_samples': 15000, 'num_dev_samples': 1500, 'num_test_samples': 1500}
    args = dotdict(args)
    utkface = UTKfaceDataset(root=root, args=args, split='train')
    X, target, sens = utkface.__getitem__(0)
    breakpoint()

