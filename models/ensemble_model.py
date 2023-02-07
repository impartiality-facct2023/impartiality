import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn as nn
import math
import time
from sklearn import metrics
from torch.nn.functional import softmax
from tqdm import tqdm

import utils
from utils import augmented_print
from models.utils_models import get_model_name_by_id
from copy import deepcopy

from analysis.multiple_counting import sample_bounded_noise
from analysis.multiple_counting import sample_gaussian_noise
from autodp.utils import clip_votes_tensor
from models.load_models import load_private_model_by_id
from utils import augmented_print
from utils import compute_metrics_multilabel
from utils import compute_metrics_multilabel_from_preds_targets
from utils import distribute_model
from utils import from_confidence_scores_to_votes
from utils import from_str
from utils import get_all_targets_numpy
# from utils import get_indexes
from utils import get_one_hot_confidence_bins
from utils import get_value_str
from utils import metric
from utils import one_hot
from utils import pick_labels_cols
from utils import pick_labels_general
from utils import result
from utils import generate_histogram_powerset
from utils import get_class_labels_and_map_powerset
from utils import get_vote_count_and_map_powerset

from analysis.pate import calculate_fairness_gaps

class EnsembleModel(nn.Module):
    """
    Noisy ensemble of private models.
    All the models for the ensemble are pre-cached in memory.
    """

    def __init__(self, model_id: int, private_models, args):
        """

        :param model_id: id of the model (-1 denotes all private models).
        :param private_models: list of private models
        :param args: program parameters
        """
        super(EnsembleModel, self).__init__()
        self.id = model_id
        if self.id == -1:
            self.name = f"ensemble(all)"
        else:
            # This is ensemble for private model_id.
            self.name = get_model_name_by_id(id=model_id)
        self.num_classes = args.num_classes
        print("Building ensemble model '{}'!".format(self.name))
        self.ensemble = private_models

    def __len__(self):
        return len(self.ensemble)

    def evaluate(self, evalloader, args):
        """Evaluate the accuracy of noisy ensemble model."""
        gap_list = np.zeros(args.num_classes, dtype=np.float64)
        correct = np.zeros(args.num_classes, dtype=np.int64)
        wrong = np.zeros(args.num_classes, dtype=np.int64)
        with torch.no_grad():
            for data, target in evalloader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                # Generate raw ensemble votes
                votes = torch.zeros((data.shape[0], self.num_classes))
                for model in self.ensemble:
                    output = model(data)
                    onehot = utils.one_hot(output.data.max(dim=1)[1].cpu(),
                                           self.num_classes)
                    votes += onehot
                # Add Gaussian noise
                assert args.sigma_gnmax >= 0
                if args.sigma_gnmax > 0:
                    noise = torch.from_numpy(
                        np.random.normal(0., args.sigma_gnmax, (
                            data.shape[0], self.num_classes))).float()
                    votes += noise
                sorted_votes = votes.sort(dim=-1, descending=True)[0]
                gaps = (sorted_votes[:, 0] - sorted_votes[:, 1]).numpy()
                preds = votes.max(dim=1)[1].numpy().astype(np.int64)
                target = target.data.cpu().numpy().astype(np.int64)
                # print("TARGET", target)
                for label, pred, gap in zip(target, preds, gaps):
                    gap_list[label] += gap
                    if label == pred:
                        correct[label] += 1
                    else:
                        wrong[label] += 1
        total = correct.sum() + wrong.sum()
        assert total == len(evalloader.dataset)
        return 100. * correct.sum() / total, 100. * correct / (
                correct + wrong), gap_list.sum() / total, gap_list / (
                       correct + wrong)

    def inference(self, unlabeled_dataloader, args):
        """Generate raw ensemble votes for RDP analysis_test."""
        all_votes = []
        end = 0
        with torch.no_grad():
            for data, *_ in unlabeled_dataloader:
                if args.cuda:
                    data = data.cuda()
                # Generate raw ensemble votes.
                batch_size = data.shape[0]
                begin = end
                end = begin + batch_size
                votes = torch.zeros((batch_size, self.num_classes))
                for model in self.ensemble:
                    output = model(data)
                    if args.dataset == "gaussian":
                        output = utils.one_hot(output.round().type(torch.LongTensor).cpu(),
                                            self.num_classes)
                        
                    if args.vote_type == 'discrete':
                        label = output.argmax(dim=1).cpu()
                        model_votes = utils.one_hot(label, self.num_classes)
                    elif args.vote_type == 'probability':
                        model_votes = F.softmax(output, dim=1).cpu()
                    else:
                        raise Exception(
                            f"Unknown args.vote_type: {args.vote_type}.")
                    votes += model_votes
                all_votes.append(votes.numpy())
    
        all_votes = np.concatenate(all_votes, axis=0)
        assert all_votes.shape == (
            len(unlabeled_dataloader.dataset), self.num_classes)
        if args.vote_type == 'discrete':
            assert np.all(all_votes.sum(axis=-1) == len(self.ensemble))
        filename = '{}-raw-votes-mode-{}-vote-type-{}'.format(
            self.name, args.mode, args.vote_type)
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, all_votes)
        return all_votes

    def inference_confidence_scores(self, unlabeled_dataloader, args):
        """Generate raw softmax confidence scores for RDP analysis_test."""
        dataset = unlabeled_dataloader.dataset
        dataset_len = len(dataset)
        num_models = len(self.ensemble)
        confidence_scores = torch.zeros(
            (num_models, dataset_len, self.num_classes))
        end = 0
        with torch.no_grad():
            for data, _ in unlabeled_dataloader:
                if args.cuda:
                    data = data.cuda()
                # Generate raw ensemble votes.
                batch_size = data.shape[0]
                begin = end
                end = begin + batch_size
                for model_idx, model in enumerate(self.ensemble):
                    output = model(data)
                    softmax_scores = F.softmax(output, dim=1).cpu()
                    confidence_scores[model_idx, begin:end, :] = softmax_scores

        filename = '{}-raw-votes-mode-{}-vote-type-{}'.format(
            self.name, args.mode, args.vote_type)
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, confidence_scores)
        return confidence_scores

    def query(self, queryloader, args, indices_queried, targets=None):
        """Query a noisy ensemble model."""
        indices_queried = np.array(indices_queried)
        indices_answered = []
        all_preds = []
        all_labels = []
        gaps_detailed = np.zeros(args.num_classes, dtype=np.float64)
        correct = np.zeros(args.num_classes, dtype=np.int64)
        wrong = np.zeros(args.num_classes, dtype=np.int64)
        with torch.no_grad():
            begin = 0
            end = 0
            for data, *target in queryloader:
                if args.has_sensitive_attribute:
                    target, sensitive = target
                else:
                    target = target[0]
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                num_samples = data.shape[0]
                end += num_samples
                # Generate raw ensemble votes
                votes = torch.zeros((num_samples, self.num_classes))
                for model in self.ensemble:
                    output = model(data)
                    if args.vote_type == 'discrete':
                        label = output.argmax(dim=1).cpu()
                        model_votes = utils.one_hot(label, self.num_classes)
                    elif args.vote_type == 'probability':
                        model_votes = F.softmax(output, dim=1).cpu()
                    else:
                        raise Exception(
                            f"Unknown args.votes_type: {args.votes_type}.")
                    votes += model_votes

                # Threshold mechanism
                if args.sigma_threshold > 0:
                    noise_threshold = np.random.normal(0., args.sigma_threshold,
                                                       num_samples)
                    vote_counts = votes.data.max(dim=1)[0].numpy()
                    answered = (vote_counts + noise_threshold) > args.threshold
                    indices_answered.append(
                        indices_queried[begin:end][answered])
                else:
                    answered = [True for _ in range(num_samples)]
                    indices_answered.append(indices_queried[begin:end])

                # GNMax mechanism
                assert args.sigma_gnmax > 0
                noise_gnmax = np.random.normal(0., args.sigma_gnmax, (
                    data.shape[0], self.num_classes))
                preds = \
                    (votes + torch.from_numpy(noise_gnmax).float()).max(dim=1)[
                        1].numpy().astype(np.int64)[answered]
                all_preds.append(preds)
                # Gap between the ensemble votes of the two most probable
                # classes.
                sorted_votes = votes.sort(dim=-1, descending=True)[0]
                gaps = (sorted_votes[:, 0] - sorted_votes[:, 1]).numpy()[
                    answered]
                # Target labels
                target = target.data.cpu().numpy().astype(np.int64)[answered]
                all_labels.append(target)
                assert len(target) == len(preds) == len(gaps)
                for label, pred, gap in zip(target, preds, gaps):
                    gaps_detailed[label] += gap
                    if label == pred:
                        correct[label] += 1
                    else:
                        wrong[label] += 1
                begin += data.shape[0]
        indices_answered = np.concatenate(indices_answered, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        total = correct.sum() + wrong.sum()
        assert len(indices_answered) == len(all_preds) == len(
            all_labels) == total
        filename = utils.get_aggregated_labels_filename(
            args=args, name=self.name)
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, all_preds)
        return indices_answered, 100. * correct.sum() / total, 100. * correct / (
                correct + wrong), gaps_detailed.sum() / total, gaps_detailed / (
                       correct + wrong)


class FairEnsembleModel(EnsembleModel):

    def __init__(self, model_id: int, private_models, args):
        super().__init__(model_id, private_models, args)
        self.subgorups = args.sensitive_group_list
        self.min_group_count = args.min_group_count
        self.max_fairness_violation = args.max_fairness_violation
        self.sensitive_group_count = np.zeros(shape=(len(args.sensitive_group_list)))
        self.per_class_pos_classified_group_count =  np.zeros(shape=(args.num_classes, len(args.sensitive_group_list)))
        
        self.fairness_disparity_gaps = None
        

    def apply_fairness_constraint(self, preds, answered, sensitive, args):
        """Ensure demographic parity fairness is within margin. Has side-effects."""         

        _answered = deepcopy(answered)

        # This is a pass-through filter. It should only block answers if they increase the fairness gap beyond `max_fairness_violation`.
        for s_id, z in enumerate(sensitive):
            z = int(z.item())
            answered = _answered[s_id]
            # Selector one-hot vectors for the sensitive feature and the predicted class 
            sensitive_one_hot_over_sensitives = (np.arange(len(args.sensitive_group_list)) == z).astype(int)
            prediction_one_hot_over_classes = (np.arange(args.num_classes) == preds[s_id]).astype(int)
            
            # Calculate of the new (tentative) gaps if the answered
            _per_class_pos_classified_group_count = self.per_class_pos_classified_group_count+ \
                prediction_one_hot_over_classes[:, None].dot(sensitive_one_hot_over_sensitives[:, None].T)
            _sensitive_group_count = self.sensitive_group_count + 1 * sensitive_one_hot_over_sensitives
            # get all the fairness gaps
            all_gaps = calculate_fairness_gaps(_sensitive_group_count, _per_class_pos_classified_group_count, rule_over_classes="all")
            _new_gaps = all_gaps[preds[s_id], :]
            # Neg. decisions maintain the gap; don;t block
            '''
            elif preds[s_id] == 0:
                _answered[s_id] = answered
            '''
            # Too few data points to estimate fairness; don't block.
            '''
            if self.sensitive_group_count[z] <  self.min_group_count:
                _answered[s_id] = answered
            '''
            if self.per_class_pos_classified_group_count[preds[s_id], z] < self.min_group_count:
                _answered[s_id] = answered
            # Pos. decisions may widen the gap, check the gap to ensure we are not over budget. If we are not, don't block.
            elif _new_gaps[z] + np.random.normal(0., 0.0) < self.max_fairness_violation:
                _answered[s_id] = answered
            else:
            # We are over budget; block.
                _answered[s_id] = False
            
            # update counters for measuring fairness 
            one_if_answered = 1 if _answered[s_id] else 0
            self.sensitive_group_count[z] += one_if_answered
            self.per_class_pos_classified_group_count[preds[s_id], z] += one_if_answered

            # update the disparity gaps 
            if one_if_answered:
                self.fairness_disparity_gaps = all_gaps

        return _answered


    def evaluate(self, evalloader, args):
        """Evaluate the accuracy of noisy ensemble model."""
        # Quesstion: Does evaluate not consider that certin queries will not be answerd? 
        gap_list = np.zeros(args.num_classes, dtype=np.float64)
        # get the number of sensitive groups
        correct = np.zeros(args.num_classes, dtype=np.int64)
        wrong = np.zeros(args.num_classes, dtype=np.int64)
        with torch.no_grad():
            for data, *target in evalloader:
                if args.has_sensitive_attribute:
                    target, sensitive = target
                else:
                    target = target[0]
                if args.cuda:
                    data, target, sensitive = data.cuda(), target.cuda(), sensitive.cuda()
                # Generate raw ensemble votes
                votes = torch.zeros((data.shape[0], self.num_classes))
                for model in self.ensemble:
                    output = model(data)
                    if args.dataset == "gaussian":
                        onehot = utils.one_hot(output.round().type(torch.LongTensor).cpu(),
                                            self.num_classes)
                    else:
                        onehot = utils.one_hot(output.data.max(dim=1)[1].cpu(),
                                            self.num_classes)
                    votes += onehot
                # Add Gaussian noise
                assert args.sigma_gnmax >= 0
                if args.sigma_gnmax > 0:
                    noise = torch.from_numpy(
                        np.random.normal(0., args.sigma_gnmax, (
                            data.shape[0], self.num_classes))).float()
                    votes += noise
                sorted_votes = votes.sort(dim=-1, descending=True)[0]
                gaps = (sorted_votes[:, 0] - sorted_votes[:, 1]).numpy()
                preds = votes.max(dim=1)[1].numpy().astype(np.int64)
                target = target.data.cpu().numpy().astype(np.int64)
                print("TARGET", target)
                for label, pred, gap in zip(target, preds, gaps):
                    gap_list[label] += gap
                    if label == pred:
                        correct[label] += 1
                    else:
                        wrong[label] += 1
        total = correct.sum() + wrong.sum()
        assert total == len(evalloader.dataset)
        return 100. * correct.sum() / total, 100. * correct / (
                correct + wrong), gap_list.sum() / total, gap_list / (
                       correct + wrong)


    def query(self, queryloader, args, indices_queried, targets=None, votes = None, preds = False):
        """Query a noisy ensemble model."""
        indices_queried = np.array(indices_queried)
        indices_answered = []
        answered_all = []
        all_preds = []
        all_labels = []
        gaps_detailed = np.zeros(args.num_classes, dtype=np.float64)
        correct = np.zeros(args.num_classes, dtype=np.int64)
        wrong = np.zeros(args.num_classes, dtype=np.int64)
        # get the number of sensitive groups
        num_sensitive = len(args.sensitive_group_list)
        correct_sens = np.zeros(num_sensitive, dtype=np.int64)
        wrong_sens = np.zeros(num_sensitive, dtype=np.int64)
        with torch.no_grad():
            begin = 0
            end = 0
            for data, *target in queryloader:
                if args.has_sensitive_attribute:
                    target, sensitive = target
                else:
                    target = target[0]
                    sensitive = None

                if args.cuda:
                    data, target, sensitive = data.cuda(), target.cuda(), sensitive.cuda()
                num_samples = data.shape[0]
                end += num_samples
                # Generate raw ensemble votes
                if not preds:
                    votes = torch.zeros((num_samples, self.num_classes))
                    for model in self.ensemble:
                        output = model(data)
                        if args.vote_type == 'discrete':
                            if args.dataset == "gaussian":
                                label = output.round().type(torch.LongTensor).cpu()
                            else:
                                label = output.argmax(dim=1).cpu()
                            model_votes = utils.one_hot(label, self.num_classes)
                        elif args.vote_type == 'probability':
                            model_votes = F.softmax(output, dim=1).cpu()
                        else:
                            raise Exception(
                                f"Unknown args.votes_type: {args.votes_type}.")
                        votes += model_votes
                else:
                    votes = votes
                # Threshold mechanism
                if args.sigma_threshold > 0:
                    noise_threshold = np.random.normal(0., args.sigma_threshold,
                                                       num_samples)
                    if not preds:
                        vote_counts = votes.data.max(dim=1)[0].numpy()
                    else:
                        vote_counts = votes.max(axis=1)
                    
                    answered = (vote_counts + noise_threshold) > args.threshold
                    # breakpoint()
                    # indices_answered.append(
                    #     indices_queried[begin:end][answered])
                else:
                    answered = [True for _ in range(num_samples)]
                    indices_answered.append(indices_queried[begin:end])

                # GNMax mechanism
                assert args.sigma_gnmax > 0
                noise_gnmax = np.random.normal(0., args.sigma_gnmax, (
                    data.shape[0], self.num_classes))
                preds = \
                    (votes + torch.from_numpy(noise_gnmax).float()).max(dim=1)[
                        1].numpy().astype(np.int64)
                # apply fairness constraint and update the answered list 
                answered = self.apply_fairness_constraint(preds, answered, sensitive, args)

                if args.sigma_threshold > 0:
                    indices_answered.append(indices_queried[begin:end][answered])
                else:
                    indices_answered.append(indices_queried[begin:end])

                preds = preds[answered]
                all_preds.append(preds)


                # Gap between the ensemble votes of the two most probable
                # classes.
                sorted_votes = votes.sort(dim=-1, descending=True)[0]
                gaps = (sorted_votes[:, 0] - sorted_votes[:, 1]).numpy()[
                    answered]
                # Target labels
                target = target.data.cpu().numpy().astype(np.int64)[answered]
                all_labels.append(target)
                assert len(target) == len(preds) == len(gaps)
                for label, pred, gap, sens in zip(target, preds, gaps, sensitive):
                    gaps_detailed[label] += gap
                    if label == pred:
                        correct[label] += 1
                        correct_sens[sens] += 1
                    else:
                        wrong[label] += 1
                        wrong_sens[sens] += 1
                begin += data.shape[0]
                if answered_all == []:
                    answered_all = answered
                else:
                    answered_all = np.concatenate([answered_all, answered])

        indices_answered = np.concatenate(indices_answered, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        total = correct.sum() + wrong.sum()
        # breakpoint()
        assert len(indices_answered) == len(all_preds) == len(
            all_labels) == total
        filename = utils.get_aggregated_labels_filename(
            args=args, name=self.name)
        filepath = os.path.join(args.ensemble_model_path, filename)
        np.save(filepath, all_preds)
        if preds:
            return answered_all, indices_answered, 100. * correct.sum() / total, 100. * correct / (
                correct + wrong), gaps_detailed.sum() / total, gaps_detailed / (
                       correct + wrong), self.fairness_disparity_gaps, 100. * correct_sens / (
                correct_sens + wrong_sens), all_preds
        else:
            return answered_all, indices_answered, 100. * correct.sum() / total, 100. * correct / (
                    correct + wrong), gaps_detailed.sum() / total, gaps_detailed / (
                        correct + wrong), self.fairness_disparity_gaps, 100. * correct_sens / (
                    correct_sens + wrong_sens)


class BigEnsembleFairModel(FairEnsembleModel):
    def __init__(self, model_id: int, private_models, args):
        super().__init__(model_id, private_models, args)
        self.num_classes = args.num_classes
        self.num_labels = args.num_classes

        # Skip the private model for the answering party id that
        # built this ensemble.
        self.model_ids = [i for i in
                          range(args.num_models) if i != model_id]

    def __len__(self):
        return len(self.model_ids)

    def get_votes_confidence_scores(self, dataloader, args) -> (
            np.ndarray, np.ndarray):
        """

        Args:
            dataloader: torch data loader
            args: program arguments

        Returns:
            votes and softmax confidence scores for each data point

        """
        dataset = dataloader.dataset
        dataset_len = len(dataset)
        votes = torch.zeros((dataset_len, self.num_classes))
        num_models = len(self.model_ids)
        confidence_scores = torch.zeros(
            (num_models, dataset_len, self.num_classes))
        with torch.no_grad():
            for model_nr, id in enumerate(self.model_ids):
                model = load_private_model_by_id(
                    args=args, id=id, model_path=args.private_model_path)
                model = distribute_model(args=args, model=model)
                correct = 0
                end = 0
                for data, *target in dataloader:
                    batch_size = data.shape[0]
                    if args.has_sensitive_attribute:
                        target, sensitive = target
                    else:
                        target = target[0] 
                    begin = end
                    end = begin + batch_size
                    if args.cuda:
                        data = data.cuda()
                    # Generate raw ensemble votes
                    output = model(data)

                    output = output.detach().cpu()
                    preds = output.argmax(dim=1)
                    labels = target.view_as(preds)
                    correct += preds.eq(labels).sum().item()
                    batch_votes = one_hot(preds, self.num_classes)
                    votes[begin:end] += batch_votes
                    softmax_scores = softmax(output, dim=1)
                    confidence_scores[model_nr, begin:end, :] = softmax_scores
                    # print(end)
                acc = correct / dataset_len
                print(f'model id {id} with acc: {acc}')
            votes = votes.numpy()
            assert np.all(votes.sum(axis=-1) == len(self.model_ids))
        return votes, confidence_scores

    def get_votes_multiclass(self, dataloader, args) -> np.ndarray:
        """

        Args:
            dataloader: torch data loader
            args: program arguments

        Returns:
            votes for each data point

        """
        votes, _ = self.get_votes_confidence_scores(
            dataloader=dataloader, args=args)
        return votes

    def get_preds(self, votes: np.ndarray, class_type: str, threshold: float):
        """
        Transform votes into predictions.

        :param votes: the votes - either counts of positive and negative votes
        for each label or the probability of a label being positive.
        :param class_type: the type of the classification task
        :param threshold: the probability threshold for predictions from the
        probabilities
        :return: the predictions
        """
        if class_type == 'multiclass':
            preds = votes.argmax(axis=-1)
        else:
            raise Exception(f"Unknown class_type: {class_type}.")
        return preds

    def get_votes(self, dataloader, args) -> np.ndarray:
        if args.class_type == 'multiclass':
            get_votes_method = self.get_votes_multiclass
        else:
            raise Exception(f'Unknown args.class_type: {args.class_type}.')
        votes = get_votes_method(dataloader=dataloader, args=args)
        return votes

    def get_votes_cached(self, dataloader, args, dataset_type='',
                         party_id=None) -> np.ndarray:
        """
        The votes for the multilabel contain the positive and negative counts
        whereas the votes for the multilabel_counting contain the probability of
        a given label being present.

        The votes are also different for ensemble models that extract the votes
        from different teacher models, thus we add the self.name to the filename
        for the votes.

        :param dataset_type: is it test, train, validation, or unlabeled.
        """
        if party_id is None:
            party_id = 'no-id'

        class_type = f"{args.class_type}"

        filename = f'votes_{args.dataset}_{args.architecture}_' \
                   f'num-models_{args.num_models}_{class_type}_{party_id}_' \
                   f'data-type_{dataset_type}_'

        filename += '.npy'
        filename = filename.replace('(', '_').replace(')', '_')

        print('cached votes filename: ', filename, flush=True)
        filepath = os.path.join(args.ensemble_model_path, filename)
        if args.load_votes is True:
            augmented_print(f'filepath: {filepath}', file=args.log_file)
            if os.path.isfile(filepath):
                augmented_print(
                    f"Loading ensemble (query {args.class_type}) votes "
                    f"for {self.name} in {args.mode} mode!", args.log_file)
                votes = np.load(filepath, allow_pickle=True)
                print("LOADED VOTES")
            else:
                augmented_print(
                    f"Generating ensemble (query {args.class_type}) votes "
                    f"for {self.name} in {args.mode} mode!", args.log_file)
                votes = self.get_votes(args=args, dataloader=dataloader)
                np.save(file=filepath, arr=votes)
        else:
            votes = self.get_votes(args=args, dataloader=dataloader)
            np.save(file=filepath, arr=votes)

        print('votes shape: ', votes.shape, flush=True)

        return votes

    def inference(self, unlabeled_dataloader, args):
        """Generate raw ensemble votes for RDP analysis_test."""
        votes = self.get_votes(dataloader=unlabeled_dataloader, args=args)
        return votes

    def query_multiclass(self, queryloader, args, indices_queried, votes):
        """Query a noisy ensemble model."""
        indices_queried = np.array(indices_queried)
        data_size = len(indices_queried)
        gaps_detailed = np.zeros(args.num_classes, dtype=np.float64)
        correct = np.zeros(args.num_classes, dtype=np.int64)
        wrong = np.zeros(args.num_classes, dtype=np.int64)
        # get the number of sensitive groups
        num_sensitive = len(args.sensitive_group_list)
        correct_sens = np.zeros(num_sensitive, dtype=np.int64)
        wrong_sens = np.zeros(num_sensitive, dtype=np.int64)
        # Thresholding mechanism (GNMax)
        if args.sigma_threshold > 0:
            noise_threshold = np.random.normal(
                loc=0.0, scale=args.sigma_threshold, size=data_size)
            vote_counts = votes.max(axis=-1)
            answered = (vote_counts + noise_threshold) > args.threshold
        else:
            answered = [True for _ in indices_queried]
        
        # Gaussian mechanism
        assert args.sigma_gnmax > 0
        noise_gnmax = np.random.normal(0., args.sigma_gnmax, (
            data_size, self.num_classes))
        noisy_votes = votes + noise_gnmax
        preds = noisy_votes.argmax(axis=1).astype(np.int64)

        # Target labels
        targets, sensitive = get_all_targets_numpy(dataloader=queryloader, args=args)
        targets = targets.astype(np.int64)
        sensitive = sensitive.numpy().astype(np.int64)
        # sensitive = np.array(sensitive).astype(np.int64)
        # apply fairness constraint and update the answered list 
        answered = self.apply_fairness_constraint(preds, answered, sensitive, args)
        count_answered = answered.sum()
        indices_answered = indices_queried[answered]
        preds = preds[answered]
        # Gap between the ensemble votes of the two most probable classes.
        # Sort the votes in descending order.
        sorted_votes = np.flip(np.sort(votes, axis=1), axis=1)
        # Compute the gap between 2 votes with the largest counts.
        gaps = (sorted_votes[:, 0] - sorted_votes[:, 1])[answered]

        targets = targets[answered]
    
        assert len(targets) == len(preds) == len(gaps) == len(indices_answered)
        for label, pred, gap, sens in zip(targets, preds, gaps, sensitive):
            gaps_detailed[label] += gap
            if label == pred:
                correct[label] += 1
                correct_sens[sens] += 1
            else:
                wrong[label] += 1
                wrong_sens[sens] += 1
        total = correct.sum() + wrong.sum()
        assert len(indices_answered) == total
        acc = 100. * correct.sum() / total
        acc_detailed = 100. * correct / (correct + wrong)
        gaps_mean = gaps_detailed.sum() / total
        gaps_detailed = gaps_detailed / (correct + wrong)

        results = {
            result.predictions: preds,
            result.indices_answered: indices_answered,
            metric.gaps_mean: gaps_mean,
            result.count_answered: count_answered,
            metric.gaps_detailed: gaps_detailed,
            metric.acc: acc,
            metric.acc_detailed: acc_detailed,
            metric.balanced_acc: 'N/A',
            metric.auc: 'N/A',
            metric.map: 'N/A',
        }

        return results, self.fairness_disparity_gaps, 100. * correct_sens / (
                correct_sens + wrong_sens)


    def query(self, queryloader, args, indices_queried, votes_queried,
              targets=None):
        # only multiclass is implemented
        if args.class_type in ['multiclass']:
            return self.query_multiclass(
                queryloader=queryloader, args=args,
                indices_queried=indices_queried, votes=votes_queried)
        else:
            raise Exception(f'Unknown args.class_type: {args.class_type}.')