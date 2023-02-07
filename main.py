from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataclasses import dataclass
from operator import truediv

import os
from pickle import FALSE
import random
import time

import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader, TensorDataset, Subset

import analysis
import utils

from active_learning import compute_utility_scores_entropy
from active_learning import compute_utility_scores_gap
from active_learning import compute_utility_scores_greedy
from architectures.densenet_pre import densenetpre
from architectures.resnet_pre import resnetpre
from architectures.utils_architectures import pytorch2pickle



from datasets.utils import get_dataset_full_name
from datasets.utils import set_dataset
from datasets.utils import show_dataset_stats
from errors import check_perfect_balance_type
from model_extraction.adaptive_training import train_model_adaptively
from model_extraction.deepfool import compute_utility_scores_deepfool
from model_extraction.main_model_extraction import run_model_extraction
from models.add_tau_per_model import set_taus
from models.big_ensemble_model import BigEnsembleModel
from models.ensemble_model import EnsembleModel, FairEnsembleModel, BigEnsembleFairModel
from models.load_models import load_private_model_by_id
from models.load_models import load_private_models
from models.private_model import get_private_model_by_id
from models.utils_models import get_model_name_by_id
from models.utils_models import model_size
from parameters import get_parameters
from utils import eval_distributed_model
from utils import eval_model
from utils import from_result_to_str
from utils import get_unlabeled_indices
from utils import get_train_dataset
from utils import get_unlabeled_set
from utils import metric
from utils import pick_labels_general
from utils import result
from utils import train_model
from utils import update_summary
from virtual_parties import query_ensemble_model_with_virtual_parties
from fairness.sampler import StratifiedBatchSampler

from csv import DictWriter

from ax.service.managed_loop import optimize
from sklearn.model_selection import train_test_split
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from torch.optim import SGD





def train_private_models(args):
    """Train N = num-models private models."""
    start_time = time.time()
    
    assert 0 <= args.begin_id
    assert args.begin_id < args.end_id
    assert args.end_id <= args.num_models
    
    filename = "logs-(id:{:d}-{:d})-(num-epochs:{:d}).txt".format(
        args.begin_id + 1, args.end_id, args.num_epochs
    )
    if os.name == "nt":
        filename = "logs-(id_{:d}-{:d})-(num-epochs_{:d}).txt".format(
            args.begin_id + 1, args.end_id, args.num_epochs
        )
    file = open(os.path.join(args.private_model_path, filename), "w+")
    args.log_file = file
    args.save_model_path = args.private_model_path
    
    utils.augmented_print(
        "Training private models on '{}' dataset!".format(args.dataset), file
    )
    utils.augmented_print(
        "Training private models on '{}' architecture!".format(
            args.architecture), file
    )
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file
    )
    utils.augmented_print(f"Initial learning rate: {args.lr}.", file)
    utils.augmented_print(
        "Number of epochs for training each model: {:d}".format(
            args.num_epochs), file
    )

    
    
    if args.dataset_type == "imbalanced":
        all_private_trainloaders = utils.load_private_data_imbalanced(args)
    elif args.dataset_type == "balanced":
        if args.balance_type == "standard":
            all_private_trainloaders = utils.load_private_data(args=args)
        elif args.balance_type == "perfect":
            check_perfect_balance_type(args=args)
            all_private_trainloaders = utils.load_private_data_imbalanced(args)
        else:
            raise Exception(f"Unknown args.balance_type: {args.balance_type}.")
    else:
        raise Exception(f"Unknown dataset type: {args.dataset_type}.")

    evalloader = utils.load_evaluation_dataloader(args)
    
    print(f"eval dataset: ", evalloader.dataset)

    if args.debug is True:
        
        show_dataset_stats(
            dataset=evalloader.dataset, args=args, file=file,
            dataset_name="eval"
        )

    
    summary = {
        "loss": [],
        "acc": [],
        "balanced_acc": [],
        "auc": [],
    }
    for id in range(args.begin_id, args.end_id):
        
        if args.dataset == "cxpert":
            model = densenetpre()
            print("Loaded densenet121")
        else:
            model = get_private_model_by_id(args=args, id=id)
        
        if args.dataset == "pascal":
            model_state_dict = model.state_dict()
            pretrained_dict34 = torch.load(
                "./architectures/resnet50-19c8e357.pth")
            pretrained_dict_1 = {
                k: v for k, v in pretrained_dict34.items() if
                k in model_state_dict
            }
            model_state_dict.update(pretrained_dict_1)
            model.load_state_dict(model_state_dict)

        trainloader = all_private_trainloaders[id]

        print(f"train dataset for model id: {id}", trainloader.dataset)

        
        if args.debug is True:
            show_dataset_stats(
                dataset=trainloader.dataset,
                args=args,
                file=file,
                dataset_name="private train",
            )
        utils.augmented_print("Steps per epoch: {:d}".format(len(trainloader)),
                              file)

        if args.dataset.startswith(
                "chexpert") and not args.architecture.startswith(
            "densenet"
        ):
            devloader = get_chexpert_dev_loader(args=args)
            result, best_model = train_chexpert.run(
                args=args,
                model=model,
                dataloader_train=trainloader,
                dataloader_dev=devloader,
                dataloader_eval=evalloader,
            )
        
        
        
        else:
            train_model(
                args=args, model=model, trainloader=trainloader,
                evalloader=evalloader
            )
            result, _ = eval_distributed_model(
                model=model, dataloader=evalloader, args=args
            )

        model_name = get_model_name_by_id(id=id)
        result["model_name"] = model_name
        result_str = from_result_to_str(result=result, sep=" | ",
                                        inner_sep=": ")
        utils.augmented_print(text=result_str, file=file, flush=True)
        summary = update_summary(summary=summary, result=result)

        
        state = result
        state["state_dict"] = model.state_dict()
        filename = "checkpoint-{}.pth.tar".format(model_name)
        filepath = os.path.join(args.private_model_path, filename)
        torch.save(state, filepath)



    for key, value in summary.items():
        if len(value) > 0:
            avg_value = np.mean(value)
            utils.augmented_print(
                f"Average {key} of private models: {avg_value}", file)

    end_time = time.time()
    elapsed_time = end_time - start_time
    utils.augmented_print(f"elapsed time: {elapsed_time}\n", file, flush=True)

    file.close()





def evaluate_ensemble_model(args):
    """Evaluate the accuracy of noisy ensemble model under varying noise scales."""
    
    file = open(
        os.path.join(args.ensemble_model_path, "logs-ensemble(all).txt"), "w")

    utils.augmented_print(
        "Evaluating ensemble model 'ensemble(all)' on '{}' dataset!".format(
            args.dataset
        ),
        file,
    )
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file
    )

    
    private_models = load_private_models(args=args)

    if args.dataset in ['colormnist', 'celeba', 'celebasensitive']:
        ensemble_model_class = FairEnsembleModel
    else:
        ensemble_model_class = EnsembleModel
    
    ensemble_model = ensemble_model_class(
        model_id=-1, args=args, private_models=private_models
    )
    
    evalloader = utils.load_evaluation_dataloader(args)
    
    error_msg = (
        f"Unknown number of models: {args.num_models} for dataset {args.dataset}."
    )
    sigma_list = [args.sigma_gnmax]
    accs = []
    gaps = []
    for sigma in sigma_list:
        args.sigma_gnmax = sigma
        acc, acc_detailed, gap, gap_detailed = ensemble_model.evaluate(
            evalloader, args)
        accs.append(acc)
        gaps.append(gap)
        utils.augmented_print("sigma_gnmax: {:.4f}".format(args.sigma_gnmax),
                              file)
        utils.augmented_print("Accuracy on evalset: {:.2f}%".format(acc), file)
        utils.augmented_print(
            "Detailed accuracy on evalset: {}".format(
                np.array2string(acc_detailed, precision=2, separator=", ")
            ),
            file,
        )
        utils.augmented_print(
            "Gap on evalset: {:.2f}% ({:.2f}|{:d})".format(
                100.0 * gap / args.num_models, gap, args.num_models
            ),
            file,
        )
        utils.augmented_print(
            "Detailed gap on evalset: {}".format(
                np.array2string(gap_detailed, precision=2, separator=", ")
            ),
            file,
            flush=True,
        )

    utils.augmented_print(f"Sigma list on evalset: {sigma_list}", file,
                          flush=True)
    utils.augmented_print(f"Accuracies on evalset: {accs}", file, flush=True)
    utils.augmented_print(f"Gaps on evalset: {gaps}", file, flush=True)

    file.close()

    if hasattr(private_models[0], "first_time"):
        model0 = private_models[0]
        print("first time: ", model0.first_time)
        print("middle time: ", model0.middle_time)
        print("last time: ", model0.last_time)


def evaluate_big_ensemble_model(args):
    """Query-answer process where each constituent model in the ensemble is
    big in the sense that we cannot load all the models to the GPUs at once."""
    
    file_name = "logs-evaluate-big-ensemble-(num-models:{})-(num-query-parties:{})-(query-mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f}).txt".format(
        args.num_models,
        args.num_querying_parties,
        args.mode,
        args.threshold,
        args.sigma_gnmax,
        args.sigma_threshold,
        args.budget,
    )
    print("ensemble_model_path: ", args.ensemble_model_path)
    print("file_name: ", file_name)
    file = open(os.path.join(args.ensemble_model_path, file_name), "w")
    args.log_file = file
    
    args.save_model_path = args.private_model_path
    
    utils.augmented_print(
        "Query-answer process on '{}' dataset!".format(args.dataset), file
    )
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file
    )
    utils.augmented_print(
        "Number of querying parties: {:d}".format(args.num_querying_parties),
        file
    )
    utils.augmented_print("Querying mode: {}".format(args.mode), file)
    utils.augmented_print("Confidence threshold: {:.1f}".format(args.threshold),
                          file)
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the GNMax mechanism: {:.1f}".format(
            args.sigma_gnmax
        ),
        file,
    )
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the threshold mechanism: {:.1f}".format(
            args.sigma_threshold
        ),
        file,
    )
    utils.augmented_print(
        "Pre-defined privacy budget: ({:.2f}, {:.0e})-DP".format(
            args.budget, args.delta
        ),
        file,
    )

    all_models_id = -1
    big_ensemble = BigEnsembleFairModel(model_id=all_models_id, args=args)



    dataset_type = "test"
    if dataset_type == "dev":
        dataloader = utils.load_dev_dataloader(args=args)
    elif dataset_type == "test":
        dataloader = utils.load_evaluation_dataloader(
            args=args)  
        print("Loaded test set")
    else:
        raise Exception(f"Unsupported dataset_type: {dataset_type}.")
    print(f"dataset: ", dataloader.dataset)

    
    
    votes = big_ensemble.get_votes_cached(
        dataloader=dataloader, args=args, dataset_type=dataset_type
    )
    if args.class_type in ['multilabel_powerset', 'multilabel_powerset_tau']:
        axis = 2
    else:
        axis = 1
    votes = pick_labels_general(labels=votes, args=args, axis=axis)

    if args.class_type in ['multilabel_powerset_tau']:
        votes = utils.pick_votes_from_probabilities(
            probs=votes, powerset_tau=args.powerset_tau,
            threshold=args.multilabel_prob_threshold)

    
    

    if args.command == 'evaluate_big_ensemble_model' and (
            args.class_type != 'multilabel_powerset'):
        sigma_gnmaxs = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 55, 60,
            
        ]
    else:
        sigma_gnmaxs = [args.sigma_gnmax]
    thresholds = [args.threshold]
    sigma_thresholds = [args.sigma_threshold]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    is_header = False
    for sigma_gnmax in sigma_gnmaxs:
        for threshold in thresholds:
            pass
            
            
            
            for sigma_threshold in sigma_thresholds:
                if sigma_threshold > threshold:
                    
                    continue
                args.threshold = threshold
                args.sigma_threshold = sigma_threshold
                args.sigma_gnmax = sigma_gnmax

                indices_queried = np.arange(0, len(dataloader.dataset))
                results = big_ensemble.query(
                    queryloader=dataloader,
                    args=args,
                    indices_queried=indices_queried,
                    votes_queried=votes,
                )

                msg = {
                    "private_tau": args.private_tau,
                    "sigma-gnmax": sigma_gnmax,
                    "acc": results[metric.acc],
                    "balanced_accuracy": results[metric.balanced_acc],
                    "auc": results[metric.auc],
                    "map": results[metric.map],
                }
                msg_str = ";".join(
                    [f"{str(key)};{str(value)}" for key, value in msg.items()]
                )
                print(msg_str)

                num_labels = args.num_classes
                if args.pick_labels is not None and args.pick_labels != [-1]:
                    num_labels = len(args.pick_labels)

                file_name = (
                    f"evaluate_big_ensemble_{args.dataset}_{args.class_type}_"
                    f"summary_private_tau_{args.private_tau}_"
                    f"dataset_{args.dataset}_"
                    f"_private_tau_{args.private_tau}_"
                    f"labels_{num_labels}_"
                    f".txt"
                )
                with open(file_name, "a") as writer:
                    writer.write(msg_str + "\n")

                
                
                
                
                
                
                
                name = args.class_type
                dataset = args.dataset
                if dataset == 'celeba':
                    dataset = 'CelebA'
                file_name = f'labels_{name}_{dataset}_{num_labels}_labels.csv'
                if args.class_type != 'multilabel_powerset':
                    with open(file_name, "a") as writer:
                        if is_header is False:
                            is_header = True
                            writer.write('sigma,metric,value\n')

                        writer.write(
                            f"{args.sigma_gnmax},ACC,{results[metric.acc]}\n")
                        writer.write(
                            f"{args.sigma_gnmax},AUC,{results[metric.auc]}\n")
                        writer.write(
                            f"{args.sigma_gnmax},MAP,{results[metric.map]}\n")

                print(
                    "Note: we have the same balanced accuracy and auc because"
                    " we operate on votes and not the probability outputs."
                )
                results_str = utils.from_result_to_str(
                    result=utils.extract_metrics(results)
                )
                utils.augmented_print(results_str, file, flush=True)
                utils.print_metrics_detailed(results=results)

    file.close()








def write_to_file(dict, filename):
    if filename != '':
        field_names = ['threshold', 'max fairness violation', 'sigma gnmax', 'sigma threshold', 'budget', 'achieved budget',
                    'accuracy', 'fairness disparity gaps', 'expected number answered', 'number answered', 'accuracy by sensitive']
        
        with open(filename, 'a') as myfile:
            dictwriter_object = DictWriter(myfile, fieldnames=field_names)
            dictwriter_object.writerow(dict)
            
            myfile.close()


def query_ensemble_model(args):
    """Query-answer process"""
    
    file_name = "logs-(num-models:{})-(num-query-parties:{})-(query-mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f}).txt".format(
        args.num_models,
        args.num_querying_parties,
        args.mode,
        args.threshold,
        args.sigma_gnmax,
        args.sigma_threshold,
        args.budget,
    )
    print("ensemble_model_path: ", args.ensemble_model_path)
    print("file_name: ", file_name)

    
    mydict = {}
    mydict['threshold'] = args.threshold
    mydict['max fairness violation'] = args.max_fairness_violation
    mydict['sigma gnmax'] = args.sigma_gnmax
    mydict['sigma threshold'] = args.sigma_threshold
    mydict['budget'] = args.budget

    file = open(os.path.join(args.ensemble_model_path, file_name), "w")
    args.save_model_path = args.ensemble_model_path
     
    utils.augmented_print(
        "Query-answer process on '{}' dataset!".format(args.dataset), file
    )
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), file
    )
    utils.augmented_print(
        "Number of querying parties: {:d}".format(args.num_querying_parties),
        file
    )
    utils.augmented_print("Querying mode: {}".format(args.mode), file)
    utils.augmented_print("Confidence threshold: {:.1f}".format(args.threshold),
                          file)
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the GNMax mechanism: {:.1f}".format(
            args.sigma_gnmax
        ),
        file,
    )
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the threshold mechanism: {:.1f}".format(
            args.sigma_threshold
        ),
        file,
    )
    utils.augmented_print(
        "Pre-defined privacy budget: ({:.2f}, {:.0e})-DP".format(
            args.budget, args.delta
        ),
        file,
    )

    model_path = args.private_model_path
    private_models = load_private_models(args=args, model_path=model_path)
    
    prev_num_models = args.num_models

    if args.test_virtual is True:
        query_ensemble_model_with_virtual_parties(args=args, file=file)

    parties_q = private_models[: args.num_querying_parties]
    args.querying_parties = parties_q

    if args.dataset in ['colormnist', 'celeba', 'celebasensitive', 'fairface', 'utkface', 'gaussian', 'chexpert-sensitive']:
        ensemble_model_class = FairEnsembleModel
    else:
        ensemble_model_class = EnsembleModel

    
    parties_a = []
    for i in range(args.num_querying_parties):
        
        
        if args.test_virtual is True:
            num_private = len(private_models) // args.num_querying_parties
            start = i * num_private
            end = start + (i + 1) * num_private
            private_subset = private_models[0:start] + private_models[end:]
        else:
            private_subset = private_models[:i] + private_models[i + 1:]

        ensemble_model = ensemble_model_class(
            model_id=i, private_models=private_subset, args=args
        )
        parties_a.append(ensemble_model)

    

    if args.attacker_dataset:
        unlabeled_dataset = utils.get_attacker_dataset(
            args=args, dataset_name=args.attacker_dataset
        )
        print("attacker uses {} dataset".format(args.attacker_dataset))
    else:
        unlabeled_dataset = utils.get_unlabeled_set(args=args)

    if args.mode == "random":
        all_indices = get_unlabeled_indices(args=args,
                                            dataset=unlabeled_dataset)
    else:
        unlabeled_dataloaders = utils.load_unlabeled_dataloaders(
            args=args, unlabeled_dataset=unlabeled_dataset
        )
        utility_scores = []

        
        if args.mode == "entropy":
            utility_function = compute_utility_scores_entropy
        elif args.mode == "gap":
            utility_function = compute_utility_scores_gap
        elif args.mode == "greedy":
            utility_function = compute_utility_scores_greedy
        elif args.mode == "deepfool":
            utility_function = compute_utility_scores_deepfool
        else:
            raise Exception(f"Unknown query selection mode: {args.mode}.")

        for i in range(args.num_querying_parties):
            filename = "{}-utility-scores-(mode-{})-dataset-{}.npy".format(
                parties_q[i].name, args.mode, args.dataset
            )
            filepath = os.path.join(args.ensemble_model_path, filename)
            if os.path.isfile(filepath) and args.debug is True:
                utils.augmented_print(
                    "Loading utility scores for '{}' in '{}' mode!".format(
                        parties_q[i].name, args.mode
                    ),
                    file,
                )
                utility = np.load(filepath)
            else:
                utils.augmented_print(
                    "Computing utility scores for '{}' in '{}' mode!".format(
                        parties_q[i].name, args.mode
                    ),
                    file,
                )
                utility = utility_function(
                    model=parties_q[i], dataloader=unlabeled_dataloaders[i],
                    args=args
                )
            utility_scores.append(utility)

        
        all_indices = []
        for i in range(args.num_querying_parties):
            offset = i * (
                    args.num_unlabeled_samples // args.num_querying_parties)
            indices = utility_scores[i].argsort()[::-1] + offset
            all_indices.append(indices)
            assert len(set(indices)) == len(indices)
        if not args.attacker_dataset:
            
            assert (
                    len(set(np.concatenate(all_indices, axis=0)))
                    == args.num_unlabeled_samples
            )


    utils.augmented_print(
        "Select queries according to their utility scores subject to the pre-defined privacy budget",
        file,
        flush=True,
    )

    for i in range(args.num_querying_parties):
        
        if args.attacker_dataset is None:
            attacker_dataset = ""
        else:
            attacker_dataset = args.attacker_dataset
        filename = "{}-raw-votes-(mode-{})-dataset-{}.npy".format(
            parties_a[i].name, args.mode, args.dataset
        )
        filepath = os.path.join(args.ensemble_model_path, filename)
        utils.augmented_print(f"filepath: {filepath}", file=file)
        if os.path.isfile(filepath) and args.debug is True:
            utils.augmented_print(
                "Loading raw ensemble votes for '{}' in '{}' mode!".format(
                    parties_a[i].name, args.mode
                ),
                file,
            )
            votes = np.load(filepath)
        else:
            utils.augmented_print(
                "Generating raw ensemble votes for '{}' in '{}' mode!".format(
                    parties_a[i].name, args.mode
                ),
                file,
            )
            
            
            unlabeled_dataloader_ordered = utils.load_ordered_unlabeled_data(
                args, all_indices[i], unlabeled_dataset=unlabeled_dataset
            )
            if args.vote_type == "confidence_scores":
                votes = parties_a[i].inference_confidence_scores(
                    unlabeled_dataloader_ordered, args
                )
            else:
                votes = parties_a[i].inference(unlabeled_dataloader_ordered,
                                               args)
            np.save(file=filepath, arr=votes)
            
            
            raw_labels_sensitives_loader = DataLoader(unlabeled_dataset, batch_size=len(unlabeled_dataset))
            
            
            targets = next(iter(raw_labels_sensitives_loader))[1].numpy()
            filename_targets = "{}-targets-(mode-{})-dataset-{}.npy".format(
                    parties_a[i].name, args.mode, args.dataset)
            filepath_targets = os.path.join(args.ensemble_model_path, filename_targets)
            np.save(file=filepath_targets, arr=targets)
            
            sensitives = next(iter(raw_labels_sensitives_loader))[2].numpy()
            filename_sensitives = "{}-sensitives-(mode-{})-dataset-{}.npy".format(
                    parties_a[i].name, args.mode, args.dataset)
            filepath_sensitives = os.path.join(args.ensemble_model_path, filename_sensitives)
            np.save(file=filepath_sensitives, arr=sensitives)
            

        
        
        (
            max_num_query, dp_eps, partition, answered, 
            order_opt, sensitive_group_count, per_class_pos_classified_group_count, 
            answered_curr, gaps, pr_answered_per_query
        ) = analysis.analyze_multiclass_confident_fair_gnmax(votes=votes, sensitives=sensitives, \
                        threshold=args.threshold, fair_threshold=args.max_fairness_violation,\
                        sigma_threshold=args.sigma_threshold, sigma_fair_threshold=0.0, sigma_gnmax=args.sigma_gnmax,\
                        budget=args.budget, delta=args.delta, file=file, show_dp_budget='disable', \
                        args=None, num_sensitive_attributes=len(args.sensitive_group_list), num_classes=args.num_classes, 
                        minimum_group_count=args.min_group_count)
        
        utils.augmented_print(
            "Maximum number of queries: {}".format(max_num_query), file
        )
        utils.augmented_print(
            "Privacy guarantee achieved: ({:.4f}, {:.0e})-DP".format(
                dp_eps[max_num_query - 1], args.delta
            ),
            file,
        )
        mydict['achieved budget'] = dp_eps[max_num_query - 1]
        utils.augmented_print(
            "Expected number of queries answered: {:.3f}".format(
                answered[max_num_query - 1]
            ),
            file,
        )
        
        mydict['expected number answered'] = answered[max_num_query - 1]
        utils.augmented_print(
            "Partition of privacy cost: {}".format(
                np.array2string(
                    partition[max_num_query - 1], precision=3, separator=", "
                )
            ),
            file,
        )


        utils.augmented_print("Generate query-answer pairs.", file)
        indices_queried = all_indices[i][:max_num_query]
        queryloader = utils.load_ordered_unlabeled_data(
            args=args, indices=indices_queried,
            unlabeled_dataset=unlabeled_dataset
        )
        
        ans, indices_answered, acc, acc_detailed, gap, gap_detailed, fairness_disparity_gaps, acc_sens = parties_a[
            i].query(
            queryloader, args, indices_queried
        )
        mydict['number answered'] = sum(ans)
        mydict['accuracy by sensitive'] = acc_sens
        
        utils.save_raw_queries_targets(
            args=args,
            indices=indices_answered,
            dataset=unlabeled_dataset,
            name=parties_q[i].name,
        )
        
        utils.augmented_print("Accuracy on queries: {:.2f}%".format(acc), file)
        mydict['accuracy'] = acc

        utils.augmented_print(
            "Detailed accuracy on queries: {}".format(
                np.array2string(acc_detailed, precision=2, separator=", ")
            ),
            file,
        )
        utils.augmented_print(
            "Gap on queries: {:.2f}% ({:.2f}|{:d})".format(
                100.0 * gap / len(parties_a[i].ensemble),
                gap,
                len(parties_a[i].ensemble),
            ),
            file,
        )
        utils.augmented_print(
            "Detailed gap on queries: {}".format(
                np.array2string(gap_detailed, precision=2, separator=", ")
            ),
            file,
        )

        utils.augmented_print(
            "Fairness Disparity Gaps: {}".format(fairness_disparity_gaps),
            file,
        )
        mydict['fairness disparity gaps'] = fairness_disparity_gaps


        utils.augmented_print("Check query-answer pairs.", file)
        queryloader = utils.load_ordered_unlabeled_data(
            args=args, indices=indices_answered,
            unlabeled_dataset=unlabeled_dataset
        )
        counts, ratios = utils.class_ratio(queryloader.dataset, args)
        utils.augmented_print(
            "Label counts: {}".format(np.array2string(counts, separator=", ")),
            file
        )
        utils.augmented_print(
            "Class ratios: {}".format(
                np.array2string(ratios, precision=2, separator=", ")
            ),
            file,
        )
        utils.augmented_print(
            "Number of samples: {:d}".format(len(queryloader.dataset)), file
        )

    
    file.close()
    args.num_models = prev_num_models
    write_to_file(mydict, args.file_name)


def query_big_ensemble_model(args):
    """Query-answer process where each constituent model in the ensemble is
    big in the sense that we cannot load all the models to the GPUs at once."""
    
    file_name = "logs-(num-models:{})-(num-query-parties:{})-(query-mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f}).txt".format(
        args.num_models,
        args.num_querying_parties,
        args.mode,
        args.threshold,
        args.sigma_gnmax,
        args.sigma_threshold,
        args.budget,
    )
    print("ensemble_model_path: ", args.ensemble_model_path)
    print("file_name: ", file_name)
    log_file = open(os.path.join(args.ensemble_model_path, file_name), "w")
    args.log_file = log_file
    
    args.save_model_path = args.private_model_path
    

    utils.augmented_print(
        "Query-answer process on '{}' dataset!".format(args.dataset), log_file
    )
    utils.augmented_print(
        "Number of private models: {:d}".format(args.num_models), log_file
    )
    utils.augmented_print(
        "Number of querying parties: {:d}".format(args.num_querying_parties),
        log_file
    )
    utils.augmented_print("Querying mode: {}".format(args.mode), log_file)
    utils.augmented_print(
        "Confidence threshold: {:.1f}".format(args.threshold), log_file
    )
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the GNMax mechanism: {:.1f}".format(
            args.sigma_gnmax
        ),
        log_file,
    )
    utils.augmented_print(
        "Standard deviation of the Gaussian noise in the threshold mechanism: {:.1f}".format(
            args.sigma_threshold
        ),
        log_file,
    )
    utils.augmented_print(
        "Pre-defined privacy budget: ({:.2f}, {:.0e})-DP".format(
            args.budget, args.delta
        ),
        log_file,
    )

    
    parties_a = {}
    if args.num_querying_parties > 0:
        for i in range(args.num_querying_parties):
            
            
            ensemble_model = BigEnsembleFairModel(model_id=i, args=args, private_models=None)
            parties_a[i] = ensemble_model
            args.querying_parties = range(args.num_querying_parties)
            
    else:
        
        
        
        other_querying_party = -1
        assert args.num_querying_parties == other_querying_party
        ensemble_model = BigEnsembleFairModel(model_id=other_querying_party,
                                          args=args)
        querying_party_ids = args.querying_party_ids
        for querying_party_id in querying_party_ids:
            parties_a[querying_party_id] = ensemble_model
        args.querying_parties = querying_party_ids


    utils.augmented_print(
        "Compute utility scores and sort available queries.", file=log_file
    )
    
    if args.mode == "entropy":
        utility_function = compute_utility_scores_entropy
    elif args.mode == "gap":
        utility_function = compute_utility_scores_gap
    elif args.mode == "greedy":
        utility_function = compute_utility_scores_greedy
    elif args.mode == "deepfool":
        utility_function = compute_utility_scores_deepfool
    else:
        assert args.mode == "random"
        utility_function = None

    unlabeled_dataset = get_unlabeled_set(args=args)

    if args.mode != "random":
        
        unlabeled_dataloaders = utils.load_unlabeled_dataloaders(args=args)
        
        utility_scores = []
        for i in range(args.num_querying_parties):
            query_party_name = get_model_name_by_id(id=i)
            filename = "{}-utility-scores-(mode:{}).npy".format(
                query_party_name, args.mode
            )
            if os.name == "nt":
                filename = "{}-utility-scores-(mode_{}).npy".format(
                    query_party_name, args.mode
                )
            filepath = os.path.join(args.ensemble_model_path, filename)
            if os.path.isfile(filepath):
                utils.augmented_print(
                    "Loading utility scores for '{}' in '{}' mode!".format(
                        query_party_name, args.mode
                    ),
                    log_file,
                )
                utility = np.load(filepath)
            else:
                utils.augmented_print(
                    "Computing utility scores for '{}' in '{}' mode!".format(
                        query_party_name, args.mode
                    ),
                    log_file,
                )
                query_party_model = load_private_model_by_id(args=args, id=i)
                utility = utility_function(
                    model=query_party_model,
                    dataloader=unlabeled_dataloaders[i],
                    args=args,
                )
            utility_scores.append(utility)
        
        unlabeled_indices = []
        for i in range(args.num_querying_parties):
            offset = i * (
                    args.num_unlabeled_samples // args.num_querying_parties)
            indices = utility_scores[i].argsort()[::-1] + offset
            unlabeled_indices.append(indices)
            assert len(set(indices)) == len(indices)
        if not args.attacker_dataset:
            
            
            assert (
                    len(set(np.concatenate(unlabeled_indices, axis=0)))
                    == args.num_unlabeled_samples
            )
    else:
        
        unlabeled_indices = get_unlabeled_indices(args=args,
                                                  dataset=unlabeled_dataset)


    utils.augmented_print(
        "Select queries according to their utility scores subject to the "
        "pre-defined privacy budget.",
        log_file,
        flush=True,
    )
    utils.augmented_print(
        "Analyze how the pre-defined privacy budget will be exhausted when "
        "answering queries.",
        log_file,
        flush=True,
    )

    for i in range(args.num_querying_parties):
        
        if args.attacker_dataset is None:
            attacker_dataset = ""
        else:
            attacker_dataset = args.attacker_dataset
        filename = "{}-raw-votes-(mode-{})-dataset-{}.npy".format(
            parties_a[i].name, args.mode, args.dataset
        )
        filepath = os.path.join(args.ensemble_model_path, filename)
        utils.augmented_print(f"filepath: {filepath}", file=log_file)
        if os.path.isfile(filepath):
            utils.augmented_print(
                "Loading raw ensemble votes for '{}' in '{}' mode!".format(
                    parties_a[i].name, args.mode
                ),
                log_file,
            )
            votes = np.load(filepath)
        else:
            utils.augmented_print(
                "Generating raw ensemble votes for '{}' in '{}' mode!".format(
                    parties_a[i].name, args.mode
                ),
                log_file,
            )
            
            
            unlabeled_dataloader_ordered = utils.load_ordered_unlabeled_data(
                args, unlabeled_indices[i], unlabeled_dataset=unlabeled_dataset
            )
            if args.vote_type == "confidence_scores":
                votes = parties_a[i].inference_confidence_scores(
                    unlabeled_dataloader_ordered, args
                )
            else:
                print("saving discrete votes!!!!!!")
                votes = parties_a[i].inference(unlabeled_dataloader_ordered,
                                               args)
            np.save(file=filepath, arr=votes)
            
            
            raw_labels_sensitives_loader = DataLoader(unlabeled_dataset, batch_size=len(unlabeled_dataset))
            
            
            targets = next(iter(raw_labels_sensitives_loader))[1].numpy()
            filename_targets = "{}-targets-(mode-{})-dataset-{}.npy".format(
                    parties_a[i].name, args.mode, args.dataset)
            filepath_targets = os.path.join(args.ensemble_model_path, filename_targets)
            np.save(file=filepath_targets, arr=targets)
            
            sensitives = next(iter(raw_labels_sensitives_loader))[2].numpy()
            filename_sensitives = "{}-sensitives-(mode-{})-dataset-{}.npy".format(
                    parties_a[i].name, args.mode, args.dataset)
            filepath_sensitives = os.path.join(args.ensemble_model_path, filename_sensitives)
            np.save(file=filepath_sensitives, arr=sensitives)
            
            
    if args.class_type == "multiclass":
        if args.threshold == 0:
            assert args.sigma_threshold == 0
            analyze = analysis.analyze_multiclass_gnmax
        else:
            analyze = analysis.analyze_multiclass_confident_fair_gnmax
    else:
        raise Exception(f"Unknown args.class_type: {args.class_type}.")

    for party_nr, party_id in enumerate(args.querying_parties):
        big_ensemble = parties_a[party_id]
        party_unlabeled_indices = unlabeled_indices[party_nr]
        query_party_name = get_model_name_by_id(id=party_id)
        utils.augmented_print(f"Querying party: {query_party_name}", log_file)

        
        unlabeled_dataloader_ordered = utils.load_ordered_unlabeled_data(
            args, party_unlabeled_indices, unlabeled_dataset=unlabeled_dataset
        )

        dataset_type = "unlabeled"
        
        all_votes = big_ensemble.get_votes_cached(
            dataloader=unlabeled_dataloader_ordered,
            args=args,
            dataset_type=dataset_type,
            party_id=party_id,
        )
        axis = 1
        if args.dataset == 'gaussian':
            votes = all_votes
        else:
            votes = pick_labels_general(labels=all_votes, args=args, axis=axis)
        
        
        filename_sensitives = "{}-sensitives-(mode-{})-dataset-{}.npy".format(
                parties_a[i].name, args.mode, args.dataset)
        filepath_sensitives = os.path.join(args.ensemble_model_path, filename_sensitives)
        sensitives = np.load(filepath_sensitives)

        utils.augmented_print(
            text=f"shape of votes: {votes.shape}", file=log_file, flush=True
        )

        

        sigma_gnmaxs = [args.sigma_gnmax]
        thresholds = [args.threshold]
        sigma_thresholds = [args.sigma_threshold]

        for sigma_gnmax in sigma_gnmaxs:
            
            
            
            
            for threshold in thresholds:
                pass
                
                
                
                for sigma_threshold in sigma_thresholds:
                    if sigma_threshold > threshold:
                        
                        
                        continue
                    args.threshold = threshold
                    args.sigma_threshold = sigma_threshold
                    args.sigma_gnmax = sigma_gnmax
                    
                    
                    (max_num_query, dp_eps, partition, answered, order_opt, sensitive_group_count, \
                    per_class_pos_classified_group_count, answered_curr, gaps, pr_answered_per_query) \
                    = analyze(
                        votes=votes, sensitives=sensitives, \
                        threshold=args.threshold, fair_threshold=args.max_fairness_violation,\
                        sigma_threshold=args.sigma_threshold, sigma_fair_threshold=0.0, sigma_gnmax=args.sigma_gnmax,\
                        budget=args.budget, delta=args.delta, file=log_file,show_dp_budget='disable', \
                        args=None, num_sensitive_attributes=len(args.sensitive_group_list), num_classes=args.num_classes, 
                        minimum_group_count=args.min_group_count)
                    
                    print('sigma_gnmax: ', sigma_gnmax,
                          ', max_num_query: ', max_num_query)
                    if max_num_query == 0:
                        
                        raise Exception(
                            f"No queries answered. The privacy "
                            f"budget is too low: {args.budget}.")

                    indices_queried = party_unlabeled_indices[:max_num_query]

                    if args.class_type in ["multiclass_confidence"]:
                        
                        
                        
                        votes_queried = votes[:, :max_num_query, :]
                    else:
                        
                        
                        votes_queried = votes[:max_num_query]

                    queryloader = utils.load_ordered_unlabeled_data(
                        args=args,
                        indices=indices_queried,
                        unlabeled_dataset=unlabeled_dataset,
                    )
                    results, fairness_gap, acc_sens = big_ensemble.query(
                        queryloader=queryloader,
                        args=args,
                        indices_queried=indices_queried,
                        votes_queried=votes_queried,
                    )
                    if args.num_classes > 2:
                        fairness_gap = np.amax(fairness_gap)
                    else:
                        fairness_gap = max(fairness_gap[1])
                    
                    if isinstance(dp_eps, np.ndarray):
                        if max_num_query > 0:
                            dp_eps = dp_eps[max_num_query - 1]
                        else:
                            dp_eps = 0

                    
                    mydict = {'threshold': args.threshold, 'sigma gnmax': args.sigma_gnmax, 
                            'max fairness violation': args.max_fairness_violation, 'sigma threshold': args.sigma_threshold, 
                            'budget': args.budget, 'achieved budget': dp_eps,
                            'accuracy': results[metric.acc], 'fairness disparity gaps': fairness_gap,
                            'expected number answered': answered[max_num_query-1], 'number answered': results[result.count_answered], 
                            'accuracy by sensitive': acc_sens}
                    msg = {
                        "private_tau": args.private_tau,
                        "privacy_budget": args.budget,
                        "max_num_query": max_num_query,
                        "dp_eps": dp_eps,
                        "sigma-gnmax": sigma_gnmax,
                        "acc": results[metric.acc],
                        "balanced_accuracy": results[metric.balanced_acc],
                        "auc": results[metric.auc],
                        "map": results[metric.map],
                        "num_answered_queries": len(
                            results[result.indices_answered]),
                        "num_labels_answered": results[result.count_answered],
                    }
                    msg_str = ";".join(
                        [f"{str(key)};{str(value)}" for key, value in
                         msg.items()]
                    )
                    print(msg_str)
                    with open(
                            "query_big_ensemble_summary_private_tau_all.txt",
                            "a"
                    ) as writer:
                        writer.write(msg_str + "\n")
                    with open(
                            f"query_big_ensemble_{args.dataset}_"
                            f"summary_private_tau_{args.private_tau}_"
                            f"{args.class_type}.txt",
                            "a",
                    ) as writer:
                        writer.write(
                            f"{args.private_tau},ACC,{results[metric.acc]}\n")
                        writer.write(
                            f"{args.private_tau},AUC,{results[metric.auc]}\n")
                        writer.write(
                            f"{args.private_tau},MAP,{results[metric.map]}\n")

                    with open(
                            f"query_big_ensemble_{args.dataset}_"
                            f"{args.private_tau}_answered_epsilon_method.txt",
                            "a",
                    ) as writer:
                        if args.class_type == "multilabel":
                            method = "PATE"
                        elif args.class_type in [
                            "multilabel_tau",
                            "multilabel_tau_pate",
                        ]:
                            method = f"L{args.private_tau_norm}"
                        else:
                            method = args.class_type

                        writer.write(f"{max_num_query},{dp_eps},{method}\n")
                    aggregated_labels = results[result.predictions]
                    indices_answered = results[result.indices_answered]

        print("aggregated labels size: ", aggregated_labels.shape)
        print("indices answered: ", len(indices_answered))
        utils.save_labels(name=query_party_name, args=args,
                          labels=aggregated_labels)
        if args.query_set_type == "raw":
            utils.save_raw_queries_targets(
                args=args,
                indices=indices_answered,
                dataset=unlabeled_dataset,
                name=query_party_name,
            )
        elif args.query_set_type == "numpy":
            utils.save_queries(
                args=args,
                indices=indices_answered,
                dataset=unlabeled_dataset,
                name=query_party_name,
            )
        else:
            raise Exception(
                f"Unknown type of the query dataset for retraining: "
                f"{args.query_set_type}."
            )


        utils.augmented_print("Check query-answer pairs.", log_file)

        utils.augmented_print(
            utils.from_result_to_str(result=utils.extract_metrics(results)),
            log_file,
            flush=True,
        )

        if args.debug is True:
            queryloader = utils.load_ordered_unlabeled_data(
                args=args, indices=indices_answered,
                unlabeled_dataset=unlabeled_dataset
            )
            counts, ratios = utils.class_ratio(queryloader.dataset, args)
            utils.augmented_print(
                "Label counts: {}".format(
                    np.array2string(counts, separator=", ")),
                log_file,
            )
            utils.augmented_print(
                "Class ratios: {}".format(
                    np.array2string(ratios, precision=2, separator=", ")
                ),
                log_file,
            )
            utils.augmented_print(
                "Number of samples: {:d}".format(len(queryloader.dataset)),
                log_file
            )

    write_to_file(mydict, args.file_name)
    log_file.close()





def retrain_private_models(args):
    """
    Retrain N = num-querying-parties private models.

    :arg args: program parameters
    """
    assert 0 <= args.begin_id and args.begin_id < args.end_id and args.end_id

    if args.num_querying_parties > 0:
        args.querying_parties = range(args.begin_id, args.end_id, 1)
    else:
        other_querying_party = -1
        assert args.num_querying_parties == other_querying_party
        args.querying_parties = args.querying_party_ids

    
    filename = 'logs-(num_models:{:d})-(id:{:d}-{:d})-(num-epochs:{:d})-(budget:{:f})-(dataset:{})-(architecture:{}).txt'.format(
        args.num_models,
        args.begin_id + 1, args.end_id,
        args.num_epochs,
        args.budget,
        args.dataset,
        args.architecture,
    )
    print('filename: ', filename)
    file = open(os.path.join(args.retrained_private_model_path, filename), 'w')
    args.save_model_path = args.retrained_private_model_path

    utils.augmented_print(
        "Retraining the private models of all querying parties on '{}' dataset!".format(
            args.dataset), file)
    utils.augmented_print(
        "Number of querying parties: {:d}".format(len(args.querying_parties)),
        file)
    utils.augmented_print("Initial learning rate: {:.2f}".format(args.lr), file)
    utils.augmented_print(
        "Number of epochs for retraining each model: {:d}".format(
            args.num_epochs), file)
    if args.test_virtual:
        assert args.num_querying_parties > 0
        prev_num_models = args.num_models
        args.num_models = args.num_querying_parties
        if args.dataset_type == 'imbalanced':
            all_private_trainloaders = utils.load_private_data_imbalanced(args)
        elif args.dataset_type == 'balanced':
            all_private_trainloaders = utils.load_private_data(args)
        else:
            raise Exception(
                'Unknown dataset type: {}'.format(args.dataset_type))
        evalloader = utils.load_evaluation_dataloader(args)
    
    if args.dataset_type == 'imbalanced':
        all_augmented_dataloaders = utils.load_private_data_and_qap_imbalanced(
            args=args)
    elif args.dataset_type == 'balanced':
        if args.balance_type == 'standard':
            all_augmented_dataloaders = utils.load_private_data_and_qap(
                args=args)
        elif args.balance_type == 'perfect':
            check_perfect_balance_type(args=args)
            all_augmented_dataloaders = utils.load_private_data_and_qap_imbalanced(
                args=args)
        else:
            raise Exception(f'Unknown args.balance_type: {args.balance_type}.')
    else:
        raise Exception(f'Unknown dataset type: {args.dataset_type}.')
    evalloader = utils.load_evaluation_dataloader(args)
    
    for party_nr, party_id in enumerate(args.querying_parties):
        utils.augmented_print("##########################################",
                              file)
        
        
        
        seed_list = [args.seed]
        model_name = get_model_name_by_id(id=party_id)
        summary = {
            metric.loss: [],
            metric.acc: [],
            metric.balanced_acc: [],
            metric.auc: [],
            metric.acc_detailed: [],
            metric.balanced_acc_detailed: [],
        }

        trainloader = all_augmented_dataloaders[party_nr]
        show_dataset_stats(
            dataset=trainloader.dataset,
            args=args,
            dataset_name='retrain data',
            file=file)

        model = None
        for seed in seed_list:
            args.seed = seed
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.cuda:
                torch.cuda.manual_seed(args.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            if args.retrain_model_type == 'load':
                
                model = load_private_model_by_id(
                    args=args, id=party_id, model_path=args.private_model_path)
            elif args.retrain_model_type == 'raw':
                model = get_private_model_by_id(args=args, id=party_id)
                model.name = model_name
            else:
                raise Exception(f"Unknown args.retrain_model_type: "
                                f"{args.retrain_model_type}")

            
            if args.dataset == 'pascal':
                model_name = f'mutillabel_net_params_{party_id}.pkl'
                model_path = args.private_model_path
                filepath = os.path.join(model_path, model_name)

                checkpoint = torch.load(filepath)
                model.load_state_dict(checkpoint)

            
            train_model(args=args, model=model, trainloader=trainloader,
                        evalloader=evalloader)

            result = eval_model(model=model, dataloader=evalloader, args=args)
            summary = update_summary(summary=summary, result=result)

        
        summary['model_name'] = model_name
        from_args = ['dataset', 'num_models', 'budget', 'architecture']
        for arg in from_args:
            summary[arg] = getattr(args, arg)

        
        for metric_key in [metric.loss, metric.acc, metric.balanced_acc,
                           metric.auc]:
            value = summary[metric_key]
            if len(value) > 0:
                avg_value = np.mean(value)
                summary[metric_key] = avg_value
            else:
                summary[metric_key] = 'N/A'

        for metric_key in [metric.acc_detailed, metric.balanced_acc_detailed]:
            detailed_value = summary[metric_key]
            if len(detailed_value) > 0:
                detailed_value = np.array(detailed_value)
                summary[metric_key] = detailed_value.mean(axis=0)
                summary[metric_key.name + '_std'] = detailed_value.std(axis=0)
            else:
                summary[metric_key] = 'N/A'

        summary_str = from_result_to_str(result=summary, sep=' | ',
                                         inner_sep=': ')
        utils.augmented_print(text=summary_str, file=file, flush=True)

        if model is not None:
            utils.save_model(args=args, model=model, result_test=summary)

        utils.augmented_print("##########################################",
                              file)


    file.close()

    if args.test_virtual:
        args.num_models = prev_num_models


def train_student_model(args):
    """
    Retrain N = num-querying-parties private models.

    :arg args: program parameters
    """
    assert 0 <= args.begin_id and args.begin_id < args.end_id and args.end_id

    if args.num_querying_parties > 0:
        args.querying_parties = range(args.begin_id, args.end_id, 1)
    else:
        other_querying_party = -1
        assert args.num_querying_parties == other_querying_party
        args.querying_parties = args.querying_party_ids

    
    filename = "logs-(num_models:{:d})-(id:{:d}-{:d})-(num-epochs:{:d})-(budget:{:f})-(dataset:{})-(architecture:{}).txt".format(
        args.num_models,
        args.begin_id + 1,
        args.end_id,
        args.num_epochs,
        args.budget,
        args.dataset,
        args.architecture,
    )
    print("filename: ", filename)
    file = open(os.path.join(args.retrained_private_model_path, filename), "w")
    args.save_model_path = args.retrained_private_model_path

    utils.augmented_print(
        "Retraining the private models of all querying parties on '{}' dataset!".format(
            args.dataset
        ),
        file,
    )
    utils.augmented_print(
        "Number of querying parties: {:d}".format(len(args.querying_parties)),
        file
    )
    utils.augmented_print("Initial learning rate: {:.2f}".format(args.lr), file)
    utils.augmented_print(
        "Number of epochs for retraining each model: {:d}".format(
            args.num_epochs), file
    )
    if args.test_virtual:
        assert args.num_querying_parties > 0
        prev_num_models = args.num_models
        args.num_models = args.num_querying_parties
        if args.dataset_type == "imbalanced":
            all_private_trainloaders = utils.load_private_data_imbalanced(args)
        elif args.dataset_type == "balanced":
            all_private_trainloaders = utils.load_private_data(args)
        else:
            raise Exception(
                "Unknown dataset type: {}".format(args.dataset_type))
        evalloader = utils.load_evaluation_dataloader(args)
    
    if args.dataset_type == "imbalanced":
        all_augmented_dataloaders = utils.load_private_data_and_qap_imbalanced(
            args=args
        )
    elif args.dataset_type == "balanced":
        if args.balance_type == "standard":
            all_augmented_dataloaders = utils.load_private_data_and_qap(
                args=args)
        elif args.balance_type == "perfect":
            check_perfect_balance_type(args=args)
            all_augmented_dataloaders = utils.load_private_data_and_qap_imbalanced(
                args=args
            )
        else:
            raise Exception(f"Unknown args.balance_type: {args.balance_type}.")
    else:
        raise Exception(f"Unknown dataset type: {args.dataset_type}.")
    evalloader = utils.load_evaluation_dataloader(args)
    

    
    
    
    seed_list = [args.seed]
    model_name = get_model_name_by_id(id=0)
    summary = {
        metric.loss: [],
        metric.acc: [],
        metric.balanced_acc: [],
        metric.auc: [],
        metric.map: [],
        metric.acc_detailed: [],
        metric.balanced_acc_detailed: [],
        metric.auc_detailed: [],
        metric.map_detailed: []
    }
    trainloader = all_augmented_dataloaders[0]
    
    
    
    
    
    if args.dataset == "pascal" and args.retrain_fine_tune:
        model = resnetpre()
        print("Loaded pretrained resnet50")
    elif args.dataset == "cxpert" and args.retrain_fine_tune:
        model = densenetpre()
        print("Loaded pretrained densenet")
    else:
        if args.retrain_model_type == 'load':
            model = load_private_model_by_id(
                args=args, id=0, model_path=args.private_model_path)
        elif args.retrain_model_type == 'raw':
            model = get_private_model_by_id(args=args, id=0)
            model.name = model_name
        else:
            raise Exception(f"Unknown args.retrain_model_type: "
                            f"{args.retrain_model_type}")

    args.seed = seed_list[0]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    train_model(args=args, model=model, trainloader=trainloader,
                evalloader=evalloader)

    result = eval_model(model=model, dataloader=evalloader, args=args)
    summary = update_summary(summary=summary, result=result)
    summary["model_name"] = model_name
    from_args = ["dataset", "num_models", "budget", "architecture"]
    for arg in from_args:
        summary[arg] = getattr(args, arg)

    
    for metric_key in [metric.loss, metric.acc, metric.balanced_acc, metric.auc,
                       metric.map]:
        value = summary[metric_key]
        if len(value) > 0:
            avg_value = np.mean(value)
            summary[metric_key] = avg_value
        else:
            summary[metric_key] = "N/A"

    for metric_key in [metric.acc_detailed, metric.balanced_acc_detailed,
                       metric.auc_detailed, metric.map_detailed]:
        detailed_value = summary[metric_key]
        if len(detailed_value) > 0:
            detailed_value = np.array(detailed_value)
            summary[metric_key] = detailed_value.mean(axis=0)
            summary[metric_key.name + "_std"] = detailed_value.std(axis=0)
        else:
            summary[metric_key] = "N/A"

    summary_str = from_result_to_str(result=summary, sep=" | ", inner_sep=": ")
    utils.augmented_print(text=summary_str, file=file, flush=True)

    if model is not None:
        utils.save_model(args=args, model=model, result_test=summary)


    file.close()

def train_student_fair(args):
    
    
    file_name = "logs-(num-models:{})-(num-query-parties:{})-(query-mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f}).txt".format(
        args.num_models,
        args.num_querying_parties,
        args.mode,
        args.threshold,
        args.sigma_gnmax,
        args.sigma_threshold,
        args.budget,
    )
    file = open(os.path.join(args.ensemble_model_path, file_name), "w")
    args.save_model_path = args.ensemble_model_path

    
    unlabeled_dataset = utils.get_unlabeled_set(args=args)

    
    filename = "model(1)-raw-votes-(mode-{})-dataset-{}.npy".format(
            args.mode, args.dataset
        )
    filepath = os.path.join(args.ensemble_model_path, filename)
    votes = np.load(filepath)
    filename = "model(1)-sensitives-(mode-{})-dataset-{}.npy".format(
        args.mode, args.dataset
    )
    filepath = os.path.join(args.ensemble_model_path, filename)
    sensitive = np.load(filepath)
    
    
    (
        max_num_query, dp_eps, _, answered, _, _, 
        _, _, _, _
        ) = analysis.analyze_multiclass_confident_fair_gnmax(votes=votes, sensitives=sensitive, \
                        threshold=args.threshold, fair_threshold=args.max_fairness_violation,\
                        sigma_threshold=args.sigma_threshold, sigma_fair_threshold=0.0, sigma_gnmax=args.sigma_gnmax,\
                        budget=args.budget, delta=args.delta, file=file,show_dp_budget='disable', \
                        args=None, num_sensitive_attributes=len(args.sensitive_group_list), num_classes=args.num_classes, 
                        minimum_group_count=args.min_group_count)
        
    ensemble_model = FairEnsembleModel(
            model_id=0, private_models=[], args=args
        )
    all_indices = list(range(0, args.num_unlabeled_samples))
    indices_queried_num = all_indices[:max_num_query]

    unlabeled_dataset = Subset(unlabeled_dataset, indices_queried_num)
    queryloader = DataLoader(
        unlabeled_dataset, batch_size=len(unlabeled_dataset), shuffle=False
    )
    
    votes = votes[:len(indices_queried_num)]
    sensitive = sensitive[:len(indices_queried_num)]
    noise_threshold = np.random.normal(0., args.sigma_threshold,
                                                       votes.shape[0])
    vote_counts = votes.max(axis=1)
    answered = (vote_counts + noise_threshold) > args.threshold              
    noise_gnmax = np.random.normal(0., args.sigma_gnmax, (
                    votes.shape[0], votes.shape[1]))
    noisy_votes = (votes + noise_gnmax)
    preds = (noisy_votes).argmax(axis=1)

    answered = ensemble_model.apply_fairness_constraint(preds, answered, sensitive, args)
    print("Number answered: ", sum(answered))
    print(ensemble_model.per_class_pos_classified_group_count)
    print(ensemble_model.fairness_disparity_gaps)
    
    X = None
    z = None
    for data, target, sensitive in queryloader:
        X = data
        z = sensitive
    indices = np.where(answered == 1)[0]
    X = X[indices].to(torch.float32)
    if args.dataset == "gaussian":
        y =  torch.from_numpy(np.expand_dims(preds[indices], axis=1)).to(torch.float32)
    else:
        y =  torch.from_numpy(preds[indices]).to(torch.float32)
    z = z[indices]

    dataset = TensorDataset(X,y,z)
    trainloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=FALSE)
    
    evalloader = utils.load_evaluation_dataloader(args)
    
    model_name = get_model_name_by_id(id=0)
    model = get_private_model_by_id(args=args, id=0)
    model.name = model_name
    train_model(args=args, model=model, trainloader=trainloader,
                evalloader=evalloader)
    
    result, fairness_gaps = eval_model(args=args, model=model, dataloader=evalloader, sensitives=True, preprocessor=True)
    print(fairness_gaps)
    field_names = ['achieved epsilon', 'fairness gaps', 'query fairness gaps', 'number answered', 'student accuracy', 
                   'student auc', 'coverage']
    mydict = {'achieved epsilon':dp_eps[max_num_query - 1], 
              'fairness gaps': np.amax(fairness_gaps), 
              'query fairness gaps': np.amax(ensemble_model.fairness_disparity_gaps), 
              'number answered': sum(answered), 
              'student accuracy':result[metric.acc],
              'student auc': result[metric.auc],
              'coverage': result[metric.coverage]}
    
    
    with open(args.file_name, 'a') as myfile:
        dictwriter_object = DictWriter(myfile, fieldnames=field_names)
        dictwriter_object.writerow(mydict)
        
        myfile.close()
    
    
    filename = "model-{:.2f}-{:.2f}-{}-pate-pre.pth.tar".format(dp_eps[max_num_query - 1], np.amax(fairness_gaps), sum(answered))
    save_model_path = os.path.join(args.path, args.dataset)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
        
    filepath = os.path.join(save_model_path, filename)
    torch.save(model.state_dict(), filepath)
    
    

def test_student_model(args):
    
    filename = "model-{}-{}-{}-{}-{}.pth.tar".format(args.threshold, args.sigma_threshold, args.sigma_gnmax, 
                                               args.budget, args.max_fairness_violation)
    save_model_path = os.path.join(args.path, args.dataset)
    filepath = os.path.join(save_model_path, filename)

    model_name = get_model_name_by_id(id=0)
    model = get_private_model_by_id(args=args, id=0)
    model.name = model_name
    model.load_state_dict(torch.load(filepath))
    device, device_ids = utils.get_device(args=args)
    model = DataParallel(model, device_ids=device_ids).to(device)
    model.eval()
    
    
    evalloader = utils.load_evaluation_dataloader(args)
    result, fairness_gaps = eval_model(model=model, dataloader=evalloader, args=args, sensitives=True, preprocessor=True)
    print(fairness_gaps)
    field_names = ['achieved epsilon', 'fairness gaps', 'query fairness gaps', 'number answered', 'student accuracy', 
                   'auc','coverage']
    mydict = {'achieved epsilon':"N/A", 
              'fairness gaps': np.amax(fairness_gaps),
              'query fairness gaps': "N/A", 
              'number answered': "N/A", 
              'student accuracy':result[metric.acc],
              'auc':result[metric.auc],
              'coverage':result[metric.coverage]}
    
    
    with open(args.file_name, 'a') as myfile:
        dictwriter_object = DictWriter(myfile, fieldnames=field_names)
        dictwriter_object.writerow(mydict)
        
        myfile.close()
           
def test_models(args):
    start_time = time.time()

    if args.num_querying_parties > 0:
        
        assert 0 <= args.begin_id
        assert args.begin_id < args.end_id
        assert args.end_id <= args.num_models
        args.querying_parties = range(args.begin_id, args.end_id, 1)
    else:
        other_querying_party = -1
        assert args.num_querying_parties == other_querying_party
        args.querying_parties = args.querying_party_ids

    
    filename = "logs-testing-(id:{:d}-{:d})-(num-epochs:{:d}).txt".format(
        args.begin_id + 1, args.end_id, args.num_epochs
    )
    file = open(os.path.join(args.private_model_path, filename), "w")
    args.log_file = file

    test_type = args.test_models_type
    
    
    if test_type == "private":
        args.save_model_path = args.private_model_path
    elif test_type == "retrained":
        args.save_model_path = args.retrained_private_model_path
    else:
        raise Exception(f"Unknown test_type: {test_type}")


    utils.augmented_print("Test models on '{}' dataset!".format(args.dataset),
                          file)
    utils.augmented_print(
        "Test models on '{}' architecture!".format(args.architecture), file
    )
    utils.augmented_print(
        "Number test models: {:d}".format(args.end_id - args.begin_id), file
    )
    if args.dataset == "pascal":
        evalloader = utils.load_evaluation_dataloader(args=args)
    else:
        evalloader = utils.load_unlabeled_dataloader(args=args)
    
    print(f"eval dataset: ", evalloader.dataset)

    if args.debug is True:
        
        show_dataset_stats(
            dataset=evalloader.dataset, args=args, file=file,
            dataset_name="eval"
        )

    
    summary = {
        metric.loss: [],
        metric.acc: [],
        metric.balanced_acc: [],
        metric.auc: [],
        metric.map: [],
    }
    for id in args.querying_parties:
        utils.augmented_print("##########################################",
                              file)

        model = load_private_model_by_id(
            args=args, id=id, model_path=args.save_model_path
        )

        result = eval_distributed_model(
            model=model, dataloader=evalloader, args=args)

        model_name = get_model_name_by_id(id=id)
        result["model_name"] = model_name
        result_str = from_result_to_str(result=result, sep="\n",
                                        inner_sep=args.sep)
        utils.print_metrics_detailed(results=result)
        utils.augmented_print(text=result_str, file=file, flush=True)
        summary = update_summary(summary=summary, result=result)



    for key, value in summary.items():
        if len(value) > 0:
            avg_value = np.mean(value)
            std_value = np.std(value)
            min_value = np.min(value)
            max_value = np.max(value)
            med_value = np.median(value)
            str_value = utils.get_value_str(value=np.array(value))
            utils.augmented_print(
                f"{key} of private models;average;{avg_value};std;{std_value};"
                f"min;{min_value};max;{max_value};median;{med_value};"
                f"value;{str_value}",
                file,
            )

    end_time = time.time()
    elapsed_time = end_time - start_time
    utils.augmented_print(f"elapsed time: {elapsed_time}\n", file, flush=True)

    file.close()


def train_eval(args, parameters):
    
    unlabeled_dataset = utils.get_unlabeled_set(args=args)

    
    filename = "model(1)-raw-votes-(mode-{})-dataset-{}.npy".format(
            args.mode, args.dataset
        )
    filepath = os.path.join(args.ensemble_model_path, filename)
    votes = np.load(filepath)
    filename = "model(1)-sensitives-(mode-{})-dataset-{}.npy".format(
        args.mode, args.dataset
    )
    filepath = os.path.join(args.ensemble_model_path, filename)
    sensitive = np.load(filepath)
    
    
    (
        max_num_query, dp_eps, _, answered, _, _, 
        _, _, _, _
        ) = analysis.analyze_multiclass_confident_fair_gnmax(votes=votes, sensitives=sensitive, \
                        threshold=parameters.get('threshold'), fair_threshold=0.5,\
                        sigma_threshold=parameters.get('sigma_threshold'), sigma_fair_threshold=0.0, sigma_gnmax=parameters.get('sigma_gnmax'),\
                        budget=args.budget, delta=args.delta, file=None,show_dp_budget='disable', \
                        args=None, num_sensitive_attributes=len(args.sensitive_group_list), num_classes=args.num_classes, 
                        minimum_group_count=args.min_group_count)
        
    ensemble_model = FairEnsembleModel(
            model_id=0, private_models=[], args=args
        )
    all_indices = list(range(0, args.num_unlabeled_samples))
    indices_queried_num = all_indices[:max_num_query]

    unlabeled_dataset = Subset(unlabeled_dataset, indices_queried_num)
    queryloader = DataLoader(
        unlabeled_dataset, batch_size=len(unlabeled_dataset), shuffle=False
    )
    
    votes = votes[:len(indices_queried_num)]
    sensitive = sensitive[:len(indices_queried_num)]
    noise_threshold = np.random.normal(0., args.sigma_threshold,
                                                       votes.shape[0])
    vote_counts = votes.max(axis=1)
    answered = (vote_counts + noise_threshold) > args.threshold              
    noise_gnmax = np.random.normal(0., args.sigma_gnmax, (
                    votes.shape[0], votes.shape[1]))
    noisy_votes = (votes + noise_gnmax)
    preds = (noisy_votes).argmax(axis=1)

    answered = ensemble_model.apply_fairness_constraint(preds, answered, sensitive, args)
    print("Number answered: ", sum(answered))
    print(ensemble_model.per_class_pos_classified_group_count)
    print(ensemble_model.fairness_disparity_gaps)
    
    X = None
    z = None
    for data, target, sensitive in queryloader:
        X = data
        z = sensitive
    indices = np.where(answered == 1)[0]
    X = X[indices].to(torch.float32)
    if args.dataset == "gaussian":
        y =  torch.from_numpy(np.expand_dims(preds[indices], axis=1)).to(torch.float32)
    else:
        y =  torch.from_numpy(preds[indices]).to(torch.float32)
    z = z[indices]

    dataset = TensorDataset(X,y,z)
    trainloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=FALSE)
    
    evalloader = utils.load_evaluation_dataloader(args)
    
    model_name = get_model_name_by_id(id=0)
    model = get_private_model_by_id(args=args, id=0)
    model.name = model_name
    train_model(args=args, model=model, trainloader=trainloader,
                evalloader=evalloader)
    
    result, fairness_gaps = eval_model(args=args, model=model, dataloader=evalloader, sensitives=True, preprocessor=True)
    return result[metric.acc]
    
    
def tune_hyperparameters(args):
    best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "threshold", "type": "range", "bounds": [100, 200]},
        {"name": "sigma_threshold", "type": "range", "bounds": [0.1, 100]},
        {"name": "sigma_gnmax", "type": "range", "bounds": [0.1, 100]}
    ],
    evaluation_function=train_eval,
    objective_name='accuracy',
)

def pretrain_dpsgd(args):
    """
    This function was used to pretrain model using the "unlabeled" public data.
    We are not using this anymore.
    """
    
    train_set = get_unlabeled_set(args)
    trainloader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=FALSE)
    
    evalloader = utils.load_evaluation_dataloader(args)
    
    
    model_name = get_model_name_by_id(id=0)
    model = get_private_model_by_id(args=args, id=0)
    model.name = model_name
    train_model(args=args, model=model, trainloader=trainloader,
                evalloader=evalloader, private = False)
    
    
    filename = "model-pretrained.pth.tar".format(args.budget, args.max_fairness_violation)
    save_model_path = os.path.join(args.path, args.dataset)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
        
    filepath = os.path.join(save_model_path, filename)
    torch.save(model.state_dict(), filepath)
    
       
def train_dpsgd(args):
    
    train_set = get_train_dataset(args)
    trainloader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=FALSE)
    
    evalloader = utils.load_evaluation_dataloader(args)

    
    model_name = get_model_name_by_id(id=0)
    model = get_private_model_by_id(args=args, id=0)
    model.name = model_name
    
    if args.loss_type == "CEWithDemParityLossPub":
        unlabeled_dataset = utils.get_unlabeled_set(args=args)
        
        
        X = np.arange(args.num_unlabeled_samples)
        y = unlabeled_dataset.dataset.gender[0:args.num_unlabeled_samples]
        _, indices, _, _ = train_test_split(X, y, test_size=100/len(y),stratify=y)
        unlabeled_dataset = Subset(dataset=unlabeled_dataset, indices=indices)
        public_dataloader = DataLoader(
            unlabeled_dataset, batch_size=len(indices), shuffle=False
        )
    else:
        public_dataloader = None
    model, epsilon, DPLoss = train_model(args=args, model=model, trainloader=trainloader,
                evalloader=evalloader, private = True, public_dataloader=public_dataloader)
    
    
    if args.inprocessing_fairness: 
        filename = "model-{}-{}.pth.tar".format(args.budget, args.inprocessing_fairness_penalty_scaler)
        if args.loss_type == 'CEWithDemParityLossPub':
            name = 'dpsgd-inprocessing-public'
        else:
            name = 'dpsgd-inprocessing'
    else:
        filename = "model-{}-{}.pth.tar".format(args.budget, args.max_fairness_violation)
        name = 'dpsgd-preprocessing'

    save_model_path = os.path.join(args.path, args.dataset, name)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
        
    filepath = os.path.join(save_model_path, filename)
    torch.save(model.state_dict(), filepath)
    
    if args.inprocessing_fairness:
        result, fairness_gaps = eval_model(args=args, model=model, dataloader=evalloader, sensitives=True, preprocessor=False, DPLoss=DPLoss)
        field_names = ['specified_epsilon', 'specified_fairness_factor', 'achieved epsilon', 'fairness gaps', 
                    'student accuracy', 'coverage']
        mydict = {'specified_epsilon':args.budget,
                'specified_fairness_factor':args.inprocessing_fairness_penalty_scaler,
                'achieved epsilon':epsilon, 
                'fairness gaps': np.amax(fairness_gaps),
                'student accuracy':result[metric.acc],
                'coverage':result[metric.coverage]}
    else:
        result, fairness_gaps = eval_model(args=args, model=model, dataloader=evalloader, sensitives=True, preprocessor=False)
        field_names = ['specified_epsilon', 'specified_fairness_gap', 'achieved epsilon', 'fairness gaps', 'number answered', 'accuracy', 'coverage']
        mydict = {'specified_epsilon':args.budget,
                'specified_fairness_gap':args.max_fairness_violation,
                'achieved epsilon': epsilon,
                'fairness gaps': np.amax(fairness_gaps),
                'number answered': len(train_set), 
                'accuracy':result[metric.acc],
                'coverage':result[metric.coverage]}

    
    with open(args.file_name, 'a') as myfile:
        dictwriter_object = DictWriter(myfile, fieldnames=field_names)
        dictwriter_object.writerow(mydict)
        
        myfile.close()
    
    

def train_student_vanilla_pate(args):
    
    
    file_name = "logs-(num-models:{})-(num-query-parties:{})-(query-mode:{})-(threshold:{:.1f})-(sigma-gnmax:{:.1f})-(sigma-threshold:{:.1f})-(budget:{:.2f}).txt".format(
        args.num_models,
        args.num_querying_parties,
        args.mode,
        args.threshold,
        args.sigma_gnmax,
        args.sigma_threshold,
        args.budget,
    )
    file = open(os.path.join(args.ensemble_model_path, file_name), "w")
    args.save_model_path = args.ensemble_model_path

    
    unlabeled_dataset = utils.get_unlabeled_set(args=args)

    
    filename = "model(1)-raw-votes-(mode-{})-dataset-{}.npy".format(
            args.mode, args.dataset
        )
    filepath = os.path.join(args.ensemble_model_path, filename)
    votes = np.load(filepath)
    filename = "model(1)-sensitives-(mode-{})-dataset-{}.npy".format(
        args.mode, args.dataset
    )
    filepath = os.path.join(args.ensemble_model_path, filename)
    sensitive = np.load(filepath)
    
    
    (
        max_num_query, dp_eps, _, answered, _
        ) = analysis.analyze_multiclass_confident_gnmax(
        votes, threshold=args.threshold, sigma_threshold=args.sigma_threshold, sigma_gnmax=args.sigma_gnmax, 
        budget=args.budget, delta=args.delta, file=file,
        show_dp_budget='disable', args=args)
        
    ensemble_model = FairEnsembleModel(
            model_id=0, private_models=[], args=args
        )
    all_indices = list(range(0, args.num_unlabeled_samples))
    indices_queried_num = all_indices[:max_num_query]

    unlabeled_dataset = Subset(unlabeled_dataset, indices_queried_num)
    queryloader = DataLoader(
        unlabeled_dataset, batch_size=len(unlabeled_dataset), shuffle=False
    )
    
    votes = votes[:len(indices_queried_num)]
    sensitive = sensitive[:len(indices_queried_num)]
    noise_threshold = np.random.normal(0., args.sigma_threshold,
                                                       votes.shape[0])
    vote_counts = votes.max(axis=1)
    answered = (vote_counts + noise_threshold) > args.threshold              
    noise_gnmax = np.random.normal(0., args.sigma_gnmax, (
                    votes.shape[0], votes.shape[1]))
    noisy_votes = (votes + noise_gnmax)
    preds = (noisy_votes).argmax(axis=1)
    print("Number answered before fairness constraint: ", sum(answered))
    if args.preprocess:
        answered = ensemble_model.apply_fairness_constraint(preds, answered, sensitive, args)
    print("Number answered: ", sum(answered))

    
    X = None
    z = None
    for data, target, sensitive in queryloader:
        X = data
        z = sensitive
    indices = np.where(answered == 1)[0]
    X = X[indices].to(torch.float32)
    if args.dataset == "gaussian":
        y =  torch.from_numpy(np.expand_dims(preds[indices], axis=1)).to(torch.float32)
    else:
        y =  torch.from_numpy(preds[indices]).to(torch.float32)
    z = z[indices]

    if args.dataset == 'chexpert-sensitive' and args.inprocessing_fairness:
        trainloader = DataLoader(
        dataset=TensorDataset(X,y,z),
        batch_sampler=StratifiedBatchSampler(torch.squeeze(z), batch_size=args.batch_size)
)
    else:
        dataset = TensorDataset(X,y,z)
        trainloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=FALSE)
    
    evalloader = utils.load_evaluation_dataloader(args)
    
    model_name = get_model_name_by_id(id=0)
    model = get_private_model_by_id(args=args, id=0)
    model.name = model_name
    train_model(args=args, model=model, trainloader=trainloader,
                evalloader=evalloader)
    
    if args.inprocessing_fairness:
        result, fairness_gaps = eval_model(args=args, model=model, dataloader=evalloader, sensitives=True, preprocessor=False)
        field_names = ['specified_epsilon', 'specified_fairness_factor', 'achieved epsilon', 'fairness gaps', 
                    'query fairness gaps', 'number answered', 'student accuracy', 
                    'student auc','coverage']
        mydict = {'specified_epsilon':args.budget,
                'specified_fairness_factor':args.inprocessing_fairness_penalty_scaler,
                'achieved epsilon':dp_eps[max_num_query - 1], 
                'fairness gaps': np.amax(fairness_gaps), 
                'query fairness gaps': np.amax(ensemble_model.fairness_disparity_gaps), 
                'number answered': sum(answered), 
                'student accuracy':result[metric.acc],
                'student auc':result[metric.auc],
                'coverage':result[metric.coverage]}
    else:
        result, fairness_gaps = eval_model(args=args, model=model, dataloader=evalloader, sensitives=True, preprocessor=True)
        field_names = ['specified_epsilon', 'specified_fairness_gap', 'achieved epsilon', 'fairness gaps', 
                    'query fairness gaps', 'number answered', 'student accuracy', 
                    'student auc','coverage']
        mydict = {'specified_epsilon':args.budget,
                'specified_fairness_gap':args.max_fairness_violation,
                'achieved epsilon':dp_eps[max_num_query - 1], 
                'fairness gaps': np.amax(fairness_gaps), 
                'query fairness gaps': np.amax(ensemble_model.fairness_disparity_gaps), 
                'number answered': sum(answered), 
                'student accuracy':result[metric.acc],
                'student auc':result[metric.auc],
                'coverage':result[metric.coverage]}
    
    
    with open(args.file_name, 'a') as myfile:
        dictwriter_object = DictWriter(myfile, fieldnames=field_names)
        dictwriter_object.writerow(mydict)
        
        myfile.close()
    
    
    if args.inprocessing_fairness: 
        filename = "model-{}-{}.pth.tar".format(args.budget, args.inprocessing_fairness_penalty_scaler)
    else:
        filename = "model-{}-{}.pth.tar".format(args.budget, args.max_fairness_violation)
    
    if args.inprocessing_fairness: 
        name = 'vanilla-inprocessing'
    else:
        name = 'vanilla-preprocessing'
    save_model_path = os.path.join(args.path, args.dataset, name)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
        
    filepath = os.path.join(save_model_path, filename)
    torch.save(model.state_dict(), filepath)

def query_DPSGD_vanilla(args):
    
    filename = "model-{}.pth.tar".format(args.budget)
    save_model_path = os.path.join(args.path, args.dataset, "DPSGD-Vanilla")
    filepath = os.path.join(save_model_path, filename)

    model_name = get_model_name_by_id(id=0)
    model = get_private_model_by_id(args=args, id=0)
    model.name = model_name
    model.load_state_dict(torch.load(filepath))
    device, device_ids = utils.get_device(args=args)
    model = DataParallel(model, device_ids=device_ids).to(device)
    model.eval()
    
    unlabeled_dataset = utils.get_unlabeled_set(args=args)
  
def query_DPSGD_inprocess(args):
    
    filename = "model-{}-{}.pth.tar".format(args.budget, args.inprocessing_fairness_penalty_scaler)
    if args.loss_type == "CEWithDemParityLossPub":
        save_model_path = os.path.join(args.path, args.dataset, 'dpsgd-inprocessing-public')
    else:
        save_model_path = os.path.join(args.path, args.dataset, 'dpsgd-inprocessing')
    filepath = os.path.join(save_model_path, filename)
            
    model_name = get_model_name_by_id(id=0)
    model = get_private_model_by_id(args=args, id=0)
    model.name = model_name
    
    evalloader = utils.load_evaluation_dataloader(args)
    
    
    model = ModuleValidator.fix(model)
    privacy_engine = PrivacyEngine()
    model, _, _ = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=SGD(model.parameters(), lr=0.1, momentum=args.momentum,weight_decay=args.weight_decay),
            data_loader=evalloader,
            epochs = args.num_epochs,
            target_epsilon = args.budget,
            target_delta=args.delta,
            max_grad_norm=1.0,
        )
    model.load_state_dict(torch.load(filepath))
    
    model.eval()
    
    
    result, fairness_gaps = eval_model(args=args, model=model, dataloader=evalloader, sensitives=True, preprocessor=True, DPLoss=0)
    field_names = ['specified_epsilon', 'specified_fairness_factor','specified_fairness_gap','fairness gaps', 'accuracy', 'coverage']
    mydict = {'specified_epsilon':args.budget,
              'specified_fairness_factor':args.inprocessing_fairness_penalty_scaler,
             'specified_fairness_gap':args.max_fairness_violation,
              'fairness gaps': np.amax(fairness_gaps),
              'accuracy':result[metric.acc],
              'coverage':result[metric.coverage]}

    
    with open(args.file_name, 'a') as myfile:
        dictwriter_object = DictWriter(myfile, fieldnames=field_names)
        dictwriter_object.writerow(mydict)
        
        myfile.close()
    
            
def main(args):
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_dataset(args=args)

    for model in args.architectures:
        args.architecture = model
        print("architecture: ", args.architecture)
        
        
        
        
        
        
        
        num_models_list = [args.num_models]
        for num_models in num_models_list:
            print("num_models: ", num_models)
            args.num_models = num_models
            if len(num_models_list) > 1:
                
                args.end_id = num_models

            architecture = args.architecture
            dataset = get_dataset_full_name(args=args)
            xray_views = "".join(args.xray_views)
            
            if args.use_pretrained_models:
                args.private_model_path = os.path.join(
                    args.path,
                    "private-models",
                    dataset + "pre",
                    architecture,
                    "{:d}-models".format(args.num_models),
                    xray_views,
                )
            else:
                args.private_model_path = os.path.join(
                    args.path,
                    "private-models",
                    dataset,
                    architecture,
                    "{:d}-models".format(args.num_models),
                    xray_views,
                )
            print("args.private_model_path: ", args.private_model_path)
            args.save_model_path = args.private_model_path
            if args.use_pretrained_models:
                args.ensemble_model_path = os.path.join(
                    args.path,
                    "ensemble-models",
                    dataset + "pre",
                    architecture,
                    "{:d}-models".format(args.num_models),
                    xray_views,
                )
            else:
                args.ensemble_model_path = os.path.join(
                    args.path,
                    "ensemble-models",
                    dataset,
                    architecture,
                    "{:d}-models".format(args.num_models),
                    xray_views,
                )

            args.non_private_model_path = os.path.join(
                args.path, "non-private-models", dataset, architecture
            )
            
            
            args.retrained_private_model_path = os.path.join(
                args.path,
                "retrained-private-models",
                dataset,
                architecture,
                "{:d}-models".format(args.num_models),
                args.mode,
                xray_views,
            )

            print(
                "args.retrained_private_models_path: ",
                args.retrained_private_model_path,
            )

            args.adaptive_model_path = os.path.join(
                args.path,
                "adaptive-model",
                dataset,
                architecture,
                "{:d}-models".format(args.num_models),
                args.mode,
                xray_views,
            )

            if args.attacker_dataset:
                args.adaptive_model_path = os.path.join(
                    args.path,
                    "adaptive-model",
                    dataset + "_" + args.attacker_dataset,
                    architecture,
                    "{:d}-models".format(args.num_models),
                    args.mode,
                    xray_views,
                )

            for path_name in [
                "private_model",
                "ensemble_model",
                "retrained_private_model",
                "adaptive_model",
            ]:
                path_name += "_path"
                args_path = getattr(args, path_name)
                
                
                
                
                
                if not os.path.exists(args_path):
                    os.makedirs(args_path)

            
            
            
            
            for private_tau in args.private_taus:
                
                args.private_tau = private_tau
                
                
                
                print("main budget: ", args.budget)
                for command in args.commands:
                    args.command = command
                    if command == "train_private_models":
                        train_private_models(args=args)
                    elif command == "evaluate_ensemble_model":
                        evaluate_ensemble_model(args=args)
                    elif command == "evaluate_big_ensemble_model":
                        evaluate_big_ensemble_model(args=args)
                    elif command == "query_ensemble_model":
                        if args.model_size == model_size.small:
                            query_ensemble_model(args=args)
                        elif args.model_size == model_size.big:
                            print("check")
                            query_big_ensemble_model(args=args)
                        else:
                            raise Exception(
                                f"Unknown args.model_size: {args.model_size}."
                            )
                    elif command == "retrain_private_models":
                        retrain_private_models(args=args)
                    elif command == "train_student_model":
                        train_student_fair(args=args)
                    elif command == "test_student_model":
                        test_student_model(args=args)
                    elif command == "pretrain_dpsgd":
                        pretrain_dpsgd(args)
                    elif command == "train_dpsgd":
                        train_dpsgd(args)
                    elif command == "train_student_vanilla_pate":
                        train_student_vanilla_pate(args)
                    elif command == "query_dpsgd":
                        query_DPSGD_inprocess(args)
                    elif command == "pytorch2pickle":
                        pytorch2pickle(args=args)
                    elif command == "test_models":
                        test_models(args=args)
                    elif command == "set_taus":
                        set_taus(args=args)
                    elif command == "train_model_adaptively":
                        train_model_adaptively(args=args)
                    elif command in [
                        "basic_model_stealing_attack",
                        "basic_model_stealing_attack_with_BO",
                    ]:
                        run_model_extraction(args=args)
                    elif command == "adaptive_queries_only":
                        run_model_extraction(args=args,
                                             no_model_extraction=True)
                    else:
                        raise Exception("Unknown command: {}".format(command))


if __name__ == "__main__":
    args = get_parameters()
    main(args)
