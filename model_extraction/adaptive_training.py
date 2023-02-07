from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import time
from torch.utils.data import DataLoader, Subset

import utils
from active_learning import compute_utility_scores_entropy
from active_learning import compute_utility_scores_gap
from active_learning import compute_utility_scores_greedy
from active_learning import compute_utility_scores_random
from datasets.dataset_custom_labels import DatasetLabels
from datasets.utils import show_dataset_stats
from model_extraction.deepfool import compute_utility_scores_deepfool
from models.ensemble_model import EnsembleModel
from models.load_models import load_private_models
from models.private_model import get_private_model_by_id
from utils import eval_distributed_model
from utils import from_result_to_str
from utils import metric
from utils import train_model
from utils import update_summary


def train_model_adaptively(args):
    start_time = time.time()

    # Logs
    filename = 'logs-(num-epochs:{:d})-train-adaptively.txt'.format(
        args.num_epochs)
    file = open(os.path.join(args.adaptive_model_path, filename), 'w')
    file_raw_acc = open(
        os.path.join(args.adaptive_model_path, f'log_raw_acc_{args.mode}.txt'),
        'w')

    args.log_file = file
    args.kwargs = utils.get_kwargs(args=args)
    args.save_model_path = args.adaptive_model_path
    utils.augmented_print("##########################################", file)

    args.log_file = file
    args.kwargs = utils.get_kwargs(args=args)
    args.save_model_path = args.adaptive_model_path
    utils.augmented_print("##########################################", file)
    utils.augmented_print(
        "Training adaptive model on '{}' dataset!".format(args.dataset), file)
    utils.augmented_print(
        "Training adaptive model on '{}' architecture!".format(
            args.architecture), file)
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file)
    utils.augmented_print(f"Initial learning rate: {args.lr}.", file)
    utils.augmented_print(
        "Number of epochs for training each model: {:d}".format(
            args.num_epochs), file)

    evalloader = utils.load_evaluation_dataloader(args)
    # evalloader = utils.load_private_data(args=args)[0]
    print(f'eval dataset: ', evalloader.dataset)

    if args.debug is True:
        # Logs about the eval set
        show_dataset_stats(dataset=evalloader.dataset, args=args, file=file,
                           dataset_name='eval')

    # Training
    summary = {
        'loss': [],
        'acc': [],
        'balanced_acc': [],
        'auc': [],
    }

    utils.augmented_print("##########################################",
                          file)

    # Select the utility function.
    if args.mode == 'entropy':
        utility_function = compute_utility_scores_entropy
    elif args.mode == 'gap':
        utility_function = compute_utility_scores_gap
    elif args.mode == 'greedy':
        utility_function = compute_utility_scores_greedy
    elif args.mode == 'random':
        utility_function = compute_utility_scores_random
    elif args.mode == 'deepfool':
        utility_function = compute_utility_scores_deepfool
    else:
        raise Exception(f"Unknown query selection mode: {args.mode}.")

    # Adaptive model for training.
    id = 0
    model = get_private_model_by_id(args=args, id=id)
    if args.cuda:
        model = model.cuda()

    # Create an ensemble model to be extracted / attacked.
    private_models = load_private_models(args=args,
                                         model_path=args.private_model_path)
    ensemble_model = EnsembleModel(model_id=-1, args=args,
                                   private_models=private_models)

    # Prepare data.
    adaptive_batch_size = args.adaptive_batch_size
    if args.attacker_dataset:
        unlabeled_dataset = utils.get_attacker_dataset(
            args=args,
            dataset_name=args.attacker_dataset)
        print("attacker uses {} dataset".format(args.attacker_dataset))
    else:
        unlabeled_dataset = utils.get_unlabeled_set(args=args)

    total_data_size = len(unlabeled_dataset)

    # All labels extracted from the attacked model.
    all_labels = np.array([])

    # Initial all indices.
    unlabeled_indices = set([i for i in range(0, total_data_size)])
    labeled_indices = []

    for i, data_size in enumerate(range(adaptive_batch_size, total_data_size,
                                        adaptive_batch_size)):
        unlabeled_dataloader = DataLoader(
            unlabeled_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **args.kwargs)

        utility_scores = utility_function(
            model=model,
            dataloader=unlabeled_dataloader,
            args=args)

        # Sort unlabeled data according to their utility scores.
        all_indices_sorted = utility_scores.argsort()[::-1]
        # Take only the next adaptive batch size for labeling and this indices
        # that have not been labeled yet.
        selected_indices = []
        for index in all_indices_sorted:
            if index in unlabeled_indices:
                selected_indices.append(index)
                if len(selected_indices) == adaptive_batch_size:
                    break

        # Remove indices that we chosen for this query.
        unlabeled_indices = unlabeled_indices.difference(selected_indices)
        assert len(unlabeled_indices) == total_data_size - data_size

        unlabeled_subset = Subset(unlabeled_dataset, list(selected_indices))
        unlabeled_subloader = DataLoader(
            unlabeled_subset,
            batch_size=args.batch_size,
            shuffle=False,
            **args.kwargs)
        votes = ensemble_model.inference(unlabeled_subloader, args)
        new_labels = votes.argmax(axis=1)
        all_labels = np.concatenate([all_labels, new_labels])

        labeled_indices += list(selected_indices)
        assert len(labeled_indices) == data_size
        assert len(unlabeled_indices.union(labeled_indices)) == total_data_size

        adaptive_dataset = Subset(unlabeled_dataset, labeled_indices)
        adaptive_dataset = DatasetLabels(adaptive_dataset, all_labels)
        adaptive_loader = DataLoader(
            adaptive_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **args.kwargs)

        # Logs about the adaptive train set.
        if args.debug is True:
            show_dataset_stats(dataset=adaptive_dataset,
                               args=args,
                               file=file,
                               dataset_name='private train')
        utils.augmented_print(
            "Steps per epoch: {:d}".format(len(adaptive_loader)), file)

        model = get_private_model_by_id(args=args, id=id)
        train_model(
            args=args,
            model=model,
            trainloader=adaptive_loader,
            evalloader=evalloader)

        result = eval_distributed_model(
            model=model, dataloader=evalloader, args=args)

        result_str = from_result_to_str(result=result, sep=' | ',
                                        inner_sep=': ')
        utils.augmented_print(text=result_str, file=file, flush=True)
        summary = update_summary(summary=summary, result=result)
        utils.augmented_print(
            f'{data_size},{result[metric.acc]},{args.mode}',
            file_raw_acc,
            flush=True)
        ensemble_acc = adaptive_dataset.correct / adaptive_dataset.total
        utils.augmented_print(text=f'accuracy of ensemble: {ensemble_acc}.',
                              file=file)

        utils.augmented_print("##########################################",
                              file)

    assert len(unlabeled_indices) == 0

    for key, value in summary.items():
        if len(value) > 0:
            avg_value = np.mean(value)
            utils.augmented_print(
                f"Average {key} of private models: {avg_value}", file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    utils.augmented_print(f"elapsed time: {elapsed_time}\n", file,
                          flush=True)
    utils.augmented_print("##########################################",
                          file)
    file.close()
    file_raw_acc.close()
