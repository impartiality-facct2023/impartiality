import numpy as np

from torch.utils.data import Subset, DataLoader, ConcatDataset
from datasets.chexpert.chexpert_dataset import ChexpertSensitive
from sklearn.model_selection import StratifiedKFold


def get_chexpert_set(args, csv, cfg, mode):
    
    dataset = ChexpertSensitive(csv=csv, config_path=cfg, mode=mode)
    # print("sensitive shape: ", sensitive.shape)
    
    return dataset

def get_chexpert_sensitive_train_set(args, cfg):
    return get_chexpert_set(args=args, csv=cfg.train_csv, cfg=cfg, mode='train')


# return the private trainloaders
def get_chexpert_sensitive_private_data(args, kwargs, cfg):
    if not args.dataset.startswith('chexpert'):
        return None
    all_private_datasets = get_chexpert_set(args=args, csv=cfg.train_csv, cfg=cfg, mode='train')
    
    skf = StratifiedKFold(n_splits=args.num_models)
    all_private_trainloaders = []

    # using this as placeholder for the training data
    X = np.zeros(len(all_private_datasets))[:-3000]
    y = all_private_datasets.race[:-3000]
    for _, indices in skf.split(X, np.array(y)):
        indices = list(indices)
        private_dataset = Subset(all_private_datasets, indices)
        kwargs = args.kwargs
        private_trainloader = DataLoader(
            dataset=private_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)
        all_private_trainloaders.append(private_trainloader)
    return all_private_trainloaders


def get_chexpert_sensitive_unlabeled_set(args, cfg):
    # take out an unlabeled set from traing for querying the ensembles
    unlabeled = get_chexpert_set(args, csv=cfg.train_csv, cfg=cfg, mode='train')
    unlabeled = Subset(unlabeled, np.arange(len(unlabeled))[-2000:])
    return unlabeled

def get_chexpert_sensitive_valid_set(args, cfg):
    # take out a validation set from training for hyperparameter tuning
    validset = get_chexpert_set(args, csv=cfg.train_csv, cfg=cfg, mode='train')
    validset = Subset(validset, np.arange(len(validset))[-3000:-2000])
    return validset

def get_chexpert_sensitive_test_set(args, cfg):
    testset = get_chexpert_set(args, csv=cfg.dev_csv, cfg=cfg, mode='dev')
    return testset
