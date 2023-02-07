import numpy as np
import os
import random
import torch

from datasets.utils import get_dataset_full_name
from datasets.utils import set_dataset
from parameters import get_parameters


def get_standard_args():
    args = get_parameters()
    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # CUDA support
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_dataset(args=args)
    args.architecture = args.architectures[0]
    print('architecture: ', args.architecture)
    args.end_id = args.num_models
    dataset = get_dataset_full_name(args=args)
    xray_views = ''.join(args.xray_views)
    # Folders
    args.private_model_path = os.path.join(
        args.path, 'private-models',
        dataset, args.architecture, '{:d}-models'.format(
            args.num_models), xray_views)
    print('args.private_model_path: ', args.private_model_path)
    args.save_model_path = args.private_model_path

    args.ensemble_model_path = os.path.join(
        args.path, 'ensemble-models',
        dataset, args.architecture, '{:d}-models'.format(
            args.num_models), xray_views)

    args.non_private_model_path = os.path.join(
        args.path, 'non-private-models',
        dataset, args.architecture)
    args.retrained_private_model_path = os.path.join(
        args.path,
        'retrained-private-models',
        dataset,
        args.architecture,
        '{:d}-models'.format(
            args.num_models),
        args.mode, xray_views)

    print('args.retrained_private_models_path: ',
          args.retrained_private_model_path)

    args.adaptive_model_path = os.path.join(
        args.path, 'adaptive-model',
        dataset, args.architecture, '{:d}-models'.format(
            args.num_models), xray_views)

    if args.attacker_dataset:
        args.adaptive_model_path = os.path.join(
            args.path, 'adaptive-model',
            dataset + "_" + args.attacker_dataset, args.architecture,
            '{:d}-models'.format(args.num_models), xray_views)

    for path_name in [
        'private_model',
        'ensemble_model',
        'retrained_private_model',
        'adaptive_model',
    ]:
        path_name += '_path'
        args_path = getattr(args, path_name)
        if not os.path.exists(args_path):
            os.makedirs(args_path)
        args.private_tau = args.private_taus[0]
        args.budget = args.budgets[0]

    return args
