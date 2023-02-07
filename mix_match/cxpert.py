from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path().absolute().parent))

from mix_match.parameters import get_args
from datasets_mix.mix_transforms import TransformTwice
from queryset import get_queries_filename
from queryset import get_raw_queries_filename
from queryset import get_aggregated_labels_filename
from datasets.xray.xray_utils import get_data
from datasets.xray.xray_utils import get_xray_test_data
from datasets.xray.xray_utils import get_xray_unlabeled_data
from datasets.xray.xray_utils import DataTypes
from datasets.xray.xray_datasets import SubsetDataset
from torch.utils.data import Dataset
from datasets_mix.cxpert_labeled import CxpertLabeledDataset


# TODO:
# For the mix-match unlabeled set, we select a portion of the unlabeled set
# (not in common with the answered queries) with an equal size as the labeled
# set and apply any relevant transformations.

# TODO: The unlabeled set also needs transformations (Check cifar10)

def get_cxpert_unlabeled_set(num_labeled=None, args=None, mix_transform=None):
    """Get the unlabeled dataset for cxpert."""
    if args.num_querying_parties == -1 and args.querying_party_ids == -1 and num_labeled is not None:
        unlabeled_dataset = get_xray_unlabeled_data(args=args,
                                                    mix_transform=mix_transform)
        print('initial_unlabeled_dataset: ', unlabeled_dataset)
        # IMPORTANT: the query_(big_)ensemble should have only a single querying
        # party for the MixMatch since we create the student model. If it is not
        # the case - then use the additional test set with 1000 examples.
        # The analyze functions takes votes one by one in order (no random
        # sampling) so labeled samples were simply the first samples from the
        # unlabeled set.
        # Thus, we skip the first labeled samples below.
        start = num_labeled
        end = len(unlabeled_dataset)
        return SubsetDataset(unlabeled_dataset, list(
            range(start, end)))  # Might be useful to add random selection here
    else:
        # Take the test portion of the whole original training set.
        return get_xray_test_data(args=args, mixtransform=mix_transform)


def get_cxpert(mix_transform_train=None, args=None):
    model_dir = os.path.join(
        args.capc_dir,
        f"ensemble-models/{args.dataset}{args.dataset_type}",
        f"{args.model_type}_{args.dataset}/{args.num_models}-models",
        "".join(args.xray_views),
    )
    model_name = 'model(1)'

    if args.query_set_type == 'raw':
        filename = get_raw_queries_filename(name=model_name, args=args)
    elif args.query_set_type == 'numpy':
        filename = get_queries_filename(name=model_name, args=args)
    filepath = os.path.join(model_dir, filename)
    if os.path.isfile(filepath):
        samples = np.load(filepath)
    else:
        raise Exception(
            f"Queries '{filepath}' do not exist, please generate them via "
            f"'query_ensemble_model(args)'!")
    filename = get_aggregated_labels_filename(name=model_name, args=args)
    filepath = os.path.join(model_dir, filename)
    if os.path.isfile(filepath):
        labels = np.load(filepath)
    else:
        raise Exception(
            "Answers '{}' do not exist, please generate them via "
            "'query_ensemble_model(args)'!".format(filepath))
    #print("samples", samples)
    print("labels", labels)

    num_labeled = len(samples)

    labeled_datasets = []
    if args.binary:
        for i in range(len(labels[1])):
            labeled_dataset = CxpertLabeledDataset(samples, labels,mix_transform_train,
                                               None, index=i)
            labeled_datasets.append(labeled_dataset)
    else:
        labeled_dataset = CxpertLabeledDataset(
            samples=samples,
            labels=labels,
            transform=mix_transform_train,
        )
        labeled_datasets.append(labeled_dataset)

    args.dataset_path = os.path.join(args.data_dir, 'CheXpert-v1.0-small/')

    # Load the original test data with only 202 valid samples and not the test
    # set that was sub-selected from the original training set.
    # The below get_data goes to the valid.csv file directly.
    test_small_dataset = get_data(args=args, data_type=DataTypes.test)
    print('test_small_dataset: ', test_small_dataset)
    unlabeled_dataset = get_cxpert_unlabeled_set(
        num_labeled=num_labeled, args=args,
        mix_transform=TransformTwice(mix_transform_train))
    print('unlabeled_dataset: ', unlabeled_dataset)

    # This is the portion of the original training set that was dedicated for
    # larger testing set (1000 samples).
    test_dataset = get_xray_test_data(args=args)
    print('test_dataset: ', test_dataset)

    return labeled_datasets, unlabeled_dataset, test_small_dataset, test_dataset


if __name__ == '__main__':
    args = get_args()
    get_cxpert(args=args)  # Test data loading
