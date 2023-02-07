from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

import analysis
import utils
from active_learning import PateKNN
from active_learning import compute_utility_scores_entropy
from active_learning import compute_utility_scores_gap
from active_learning import compute_utility_scores_greedy
from active_learning import compute_utility_scores_random
from datasets.dataset_custom_labels import DatasetLabels
from datasets.utils import show_dataset_stats
from model_extraction.deepfool import compute_utility_scores_deepfool
from models.ensemble_model import EnsembleModel
from models.load_models import load_private_models
from models.load_models import load_victim_model
from models.private_model import get_private_model_by_id
from utils import eval_distributed_model
from utils import from_result_to_str
from utils import metric
from utils import update_summary


def get_utility_function(args):
    """
    Select the utility function.

    :param args: the arguments for the program.
    :return: the utility function (handler).
    """
    if args.mode == 'entropy':
        utility_function = compute_utility_scores_entropy
    elif args.mode == 'gap':
        utility_function = compute_utility_scores_gap
    elif args.mode == 'greedy':
        utility_function = compute_utility_scores_greedy
    elif args.mode == 'deepfool':
        utility_function = compute_utility_scores_deepfool
    elif args.mode == 'random':
        utility_function = compute_utility_scores_random
    else:
        raise Exception(f"Unknown query selection mode: {args.mode}.")
    return utility_function


def set_victim_model_path(args):
    if args.target_model in ["victim", "victim_only"]:
        args.victim_model_path = os.path.join(
            args.path, 'private-models',
            args.dataset, args.architecture, '1-models')
    elif args.target_model in ["pate", "another_pate"]:
        args.victim_model_path = os.path.join(
            args.path, 'private-models',
            args.dataset, args.architecture,
            '{}-models'.format(args.num_models))
    else:
        raise Exception(
            f"Target unspecified or unknown target type: {args.target_model}.")
    if os.path.exists(args.victim_model_path):
        print('args.victim_model_path: ', args.victim_model_path)
    else:
        raise Exception(
            "Victim Model does not exist at {}".format(args.victim_model_path))


def get_log_files(args, create_files=False):
    log_file_name = f"logs-num-epochs-{args.num_epochs}-{args.dataset}-{args.mode}-model-stealing.txt"
    log_file = os.path.join(args.path, log_file_name)
    file_raw_acc_name = f"log_raw_acc_PATE_cost_{args.mode}.txt"
    file_raw_acc = os.path.join(args.adaptive_model_path,
                                file_raw_acc_name)
    file_raw_entropy_name = f"log_raw_entropy_{args.mode}.txt"
    file_raw_entropy = os.path.join(args.adaptive_model_path,
                                    file_raw_entropy_name)
    file_privacy_cost = os.path.join(args.adaptive_model_path,
                                     f'log_raw_pkNN_cost_{args.mode}.txt')
    file_raw_gap = os.path.join(args.adaptive_model_path,
                                f'log_raw_gap_{args.mode}.txt')

    files = {
        'log_file': log_file,
        'file_raw_acc': file_raw_acc,
        'file_raw_entropy': file_raw_entropy,
        'file_privacy_cost': file_privacy_cost,
        'file_raw_gap': file_raw_gap,
    }

    if create_files:
        for name in files:
            file_path = files[name]
            openfile = open(file_path, 'w+')
            openfile.close()

    return files


def close_log_files(files: dict):
    for file in files.values():
        file.close()


def print_initial_logs(args, evalloader=None):
    utils.augmented_print(
        "Training adaptive model on '{}' dataset!".format(args.dataset),
        args.log_file)
    utils.augmented_print(
        "Training adaptive model on '{}' architecture!".format(
            args.architecture), args.log_file)
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), args.log_file)
    utils.augmented_print(f"Initial learning rate: {args.lr}.", args.log_file)
    utils.augmented_print(
        "Number of epochs for training each model: {:d}".format(
            args.num_epochs), args.log_file)
    # Logs about the eval set
    if evalloader is not None:
        print(f'eval dataset: ', evalloader.dataset)
        show_dataset_stats(dataset=evalloader.dataset, args=args,
                           file=args.log_file,
                           dataset_name='eval')


def retrain(args, model, adaptive_loader, adaptive_dataset, evalloader,
            dp_eps, data_size, file_raw_acc):
    summary = {
        'loss': [],
        'acc': [],
        'balanced_acc': [],
        'auc': [],
    }
    utils.augmented_print(
        f"Steps per epoch: {len(adaptive_loader)}.", args.log_file)
    if args.num_optimization_loop > 0:
        best_parameters = utils.bayesian_optimization_training_loop(
            args, model, adaptive_dataset, evalloader,
            patience=args.patience,
            num_optimization_loop=args.num_optimization_loop)

    else:
        model = get_private_model_by_id(args=args, id=0)
        best_parameters = {"lr": args.lr, "batch_size": args.batch_size}

    model = utils.train_with_bayesian_optimization_with_best_hyperparameter(
        args,
        model,
        adaptive_dataset,
        evalloader,
        parameters=best_parameters,
        patience=args.patience)

    result = eval_distributed_model(
        model=model, dataloader=evalloader, args=args)

    result_str = from_result_to_str(result=result, sep=' | ',
                                    inner_sep=': ')
    utils.augmented_print(text=result_str, file=args.log_file, flush=True)
    summary = update_summary(summary=summary, result=result)
    utils.augmented_print(
        f'{data_size},{result[metric.acc]},{args.mode},{dp_eps}',
        file_raw_acc,
        flush=True)
    utils.augmented_print(
        text=f'best hyperparameters : '
             f'lr {best_parameters["lr"]}, '
             f'batch size {best_parameters["batch_size"]}',
        file=args.log_file)
    # the number reported here is not correct, what is ensemble accuracy exactly?
    # ensemble_acc = adaptive_dataset.correct / len(adaptive_dataset)
    # utils.augmented_print(text=f'accuracy of ensemble: {ensemble_acc}.',
    #                       file=args.log_file)

    for key, value in summary.items():
        if len(value) > 0:
            avg_value = np.mean(value)
            utils.augmented_print(
                f"Average {key} of private models: {avg_value}", args.log_file)

    return result, model


def select_query_indices_based_on_utility(args, unlabeled_indices,
                                          unlabeled_dataset, utility_function,
                                          model,
                                          adaptive_batch_size):
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **args.kwargs)

    utility_scores = utility_function(
        model=model,
        dataloader=unlabeled_dataloader,
        args=args)

    # Sort unlabeled data according to their utility scores in
    # the descending order.
    all_indices_sorted = utility_scores.argsort()[::-1]
    # Take only the next adaptive batch size for labeling and this indices
    # that have not been labeled yet.
    selected_indices = []
    for index in all_indices_sorted:
        if index in unlabeled_indices:
            selected_indices.append(index)
            if len(selected_indices) == adaptive_batch_size:
                break
    return selected_indices


def get_victim_model_and_estimator(args):
    if not args.target_model == "victim_only":
        cost_estimator = load_private_models(args=args,
                                             model_path=args.private_model_path)
        #
        cost_estimator_model = EnsembleModel(model_id=-1, args=args,
                                             private_models=cost_estimator)
    else:
        cost_estimator_model = None

    if args.target_model in ["victim", "victim_only"]:
        victim_model = load_victim_model(args=args)
    elif args.target_model == "another_pate":
        # in this case, we are comparing privacy cost output from 2 pate with same architecture,
        # one as the victim and one as the cost estimator
        # we load the victim from a different folder victim_model_path
        private_models = load_private_models(args=args,
                                             model_path=args.victim_model_path)
        victim_model = EnsembleModel(model_id=-1, args=args,
                                     private_models=private_models)
    elif args.target_model == "pate":
        # Create an ensemble model to be extracted / attacked.
        private_models = load_private_models(args=args,
                                             model_path=args.private_model_path)
        victim_model = EnsembleModel(model_id=-1, args=args,
                                     private_models=private_models)
    else:
        raise Exception("target model not defined")

    return victim_model, cost_estimator_model


def run_model_extraction(args, no_model_extraction=False):
    start_time = time.time()

    files = get_log_files(args=args, create_files=True)
    log_file = files['log_file']
    file_raw_acc = files['file_raw_acc']
    file_raw_entropy = files['file_raw_entropy']
    file_privacy_cost = files['file_privacy_cost']
    file_raw_gap = files['file_raw_gap']

    evalloader = utils.load_evaluation_dataloader(args=args)

    args.log_file = log_file
    args.kwargs = utils.get_kwargs(args=args)
    args.save_model_path = args.adaptive_model_path

    set_victim_model_path(args=args)
    print_initial_logs(args=args, evalloader=evalloader)

    utility_function = get_utility_function(args=args)
    entropy_cost = 0
    pate_cost = 0
    gap_cost = 0
    stolen_model = get_private_model_by_id(args=args, id=0)
    # Adaptive model for training.
    if args.cuda:
        stolen_model = stolen_model.cuda()

    victim_model, cost_estimator_model = get_victim_model_and_estimator(args)

    if args.target_model == 'victim':
        victim_acc = eval_distributed_model(
            model=victim_model, dataloader=evalloader, args=args)
        utils.augmented_print(
            text=f'accuracy of victim: {victim_acc[metric.acc]}.',
            file=log_file)
        trainloader = utils.load_training_data(args=args)

        # for the Private kNN
        pate_knn = PateKNN(model=victim_model, trainloader=trainloader,
                           args=args)

    else:
        trainloader = None
        train_represent = None
        train_labels = None

    # Prepare data.
    # How many queries do we answer in a single request?
    adaptive_batch_size = args.adaptive_batch_size

    # we are using a different dataset to steal this model
    if args.attacker_dataset:
        unlabeled_dataset = utils.get_attacker_dataset(
            args=args,
            dataset_name=args.attacker_dataset)
        print(f"attacker uses {args.attacker_dataset} dataset.")
    else:
        unlabeled_dataset = utils.get_unlabeled_set(args=args)

    if args.attacker_dataset == "tinyimages":
        unlabeled_dataset_cifar = utils.get_unlabeled_set(args=args)
        unlabeled_dataset = ConcatDataset(
            [unlabeled_dataset_cifar, unlabeled_dataset])

    total_data_size = len(unlabeled_dataset)
    print(f"There are {total_data_size} unlabeled points in total.")

    # Initially all indices are unlabeled.
    unlabeled_indices = set([i for i in range(0, total_data_size)])
    # We will progressively add more labeled indices.
    labeled_indices = []
    # All labels extracted from the attacked model.
    all_labels = np.array([])

    data_iterator = range(
        adaptive_batch_size, total_data_size + 1, adaptive_batch_size)
    retrain_extracted_model = args.retrain_extracted_model  # save the parameter retrain_extracted_model
    for i, data_size in enumerate(data_iterator):
        if no_model_extraction:
            # cannot select query with stolen model's output, use victim instead
            selected_indices = select_query_indices_based_on_utility(
                args=args,
                unlabeled_indices=unlabeled_indices,
                unlabeled_dataset=unlabeled_dataset,
                utility_function=utility_function,
                model=victim_model,
                adaptive_batch_size=adaptive_batch_size)
        else:
            if i == 0:
                print("First iteration, no retraining")
                args.retrain_extracted_model = False
            else:
                args.retrain_extracted_model = retrain_extracted_model
            selected_indices = select_query_indices_based_on_utility(
                args=args,
                unlabeled_indices=unlabeled_indices,
                unlabeled_dataset=unlabeled_dataset,
                utility_function=utility_function,
                model=stolen_model,
                adaptive_batch_size=adaptive_batch_size)

        # Remove indices that we choose for this query.
        unlabeled_indices = unlabeled_indices.difference(selected_indices)
        assert len(unlabeled_indices) == total_data_size - data_size

        unlabeled_subset = Subset(unlabeled_dataset, list(selected_indices))
        unlabeled_subloader = DataLoader(
            unlabeled_subset,
            batch_size=args.batch_size,
            shuffle=False,
            **args.kwargs)

        new_labels = []
        if args.target_model in ["victim", "victim_only"]:
            predicted_logits = utils.get_prediction(
                args=args, model=victim_model,
                unlabeled_dataloader=unlabeled_subloader)
            new_labels = predicted_logits.argmax(axis=1).cpu()

            gap_scores = compute_utility_scores_gap(
                model=victim_model, dataloader=unlabeled_subloader, args=args)
            gap_cost += gap_scores.sum()

            entropy_scores = compute_utility_scores_entropy(
                model=victim_model, dataloader=unlabeled_subloader, args=args)
            entropy_cost += entropy_scores.sum()

            # utils.augmented_print(
            #     f"{data_size},'entropy',{entropy_cost}", file_raw_entropy,
            #     flush=True)
            # utils.augmented_print(
            #     f"{data_size},'gap',{gap_cost}", file_raw_gap,
            #     flush=True)

            if trainloader:
                # record privacy cost
                pate_cost = pate_knn.compute_privacy_cost(
                    unlabeled_loader=unlabeled_subloader)

                msg = f"{data_size},{args.mode},private-knn,{pate_cost}"
                utils.augmented_print(msg, file_privacy_cost, flush=True)

        elif args.target_model in ["pate", "another_pate"]:
            # victim model is ensemble model
            votes = victim_model.inference(unlabeled_subloader, args)
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

        if args.target_model == "another_pate":
            # record the victim's pate privacy estimation (this will be different from the pate privacy estimator)
            # so we can compare the two output to see how much difference there is
            votes_victim = victim_model.inference(adaptive_loader, args)
            # Analyze how the pre-defined privacy budget will be exhausted when
            # answering queries.
            if args.threshold == 0:
                # Disable the thresholding mechanism.
                assert args.sigma_threshold == 0
                filename_victim_privacy_cost = os.path.join(
                    args.adaptive_model_path,
                    f'log_raw_dps_{args.mode}_with_BO_victim.txt')
                max_num_query_victim, dp_eps_victim, partition_victim, answered_victim, order_opt_victim = analysis.analyze_multiclass_gnmax(
                    votes=votes_victim,
                    threshold=0,
                    sigma_threshold=0,
                    sigma_gnmax=args.sigma_gnmax,
                    budget=args.budget,
                    delta=args.delta,
                    file=filename_victim_privacy_cost,
                    show_dp_budget=args.show_dp_budget,
                    args=args
                )

        # Logs about the adaptive train set.
        if args.debug is True:
            show_dataset_stats(dataset=adaptive_dataset,
                               args=args,
                               file=log_file,
                               dataset_name='private train')

        if no_model_extraction:
            # skip all the model saving/training logic
            continue
        else:
            result, model = retrain(args=args, model=stolen_model,
                                    file_raw_acc=file_raw_acc,
                                    evalloader=evalloader,
                                    adaptive_dataset=adaptive_dataset,
                                    adaptive_loader=adaptive_loader,
                                    dp_eps=pate_cost,
                                    data_size=data_size)
            # save checkpoint
            state = result
            state['epoch'] = i
            state['state_dict'] = model.module.state_dict()
            filename = "checkpoint-1.pth.tar"
            filepath = os.path.join(args.save_model_path, filename)
            torch.save(state, filepath)
            print("trained model for iteration {} mode {} saved at {}".format(
                i, args.mode, filepath))
    assert len(unlabeled_indices) == 0
    if args.target_model in ["victim", "victim_only"]:
        victim_acc = eval_distributed_model(
            model=victim_model, dataloader=evalloader, args=args)
        utils.augmented_print(text=f'accuracy of victim: {victim_acc["acc"]}.',
                              file=log_file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    utils.augmented_print(f"elapsed time: {elapsed_time}\n", log_file,
                          flush=True)
    close_log_files(files=files)
