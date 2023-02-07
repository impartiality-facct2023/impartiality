import torch
import torch.nn.functional as F

import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional

class CEWithDemParityLoss(nn.CrossEntropyLoss):

    def __init__(self, temperatrue=0.01, _gamma=0.5, _lambda = 1.0, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        self._lambda = _lambda
        self._gamma = _gamma
        self.temperatrue = temperatrue

    @staticmethod
    def calculate_fairness_gaps(sensitive_group_count, per_class_pos_classified_group_count, zero_dev_small_number=1e-5, rule_over_classes=None, rule_over_subgroup=None):
        """_summary_

        Args:
            sensitive_group_count (_type_): _description_
            per_class_pos_classified_group_count (_type_): _description_
            zero_dev_small_number (_type_, optional): _description_. Defaults to 1e-5.
            rule_over_classes (str, optional): _description_. Defaults to "avg".

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """

        def reduce_over(t: Tensor, dim: int, aggr: Optional[str] = None):
            if aggr == None:
                return t
            elif aggr == "avg":
                return t.amean(dim=dim)
            elif aggr == "max":
                return t.amax(dim=dim)
            else:
                raise NotImplementedError

        # breakpoint()
        every_other_sensitive_group_count = sensitive_group_count.sum() - sensitive_group_count
        every_other_per_class_pos_classified_group_count = per_class_pos_classified_group_count.sum(axis=1)[:, None] - per_class_pos_classified_group_count
        
        # breakpoint()
        self_dem_parity = torch.div(per_class_pos_classified_group_count, sensitive_group_count + zero_dev_small_number)
        every_other_dem_parity = torch.div(every_other_per_class_pos_classified_group_count, every_other_sensitive_group_count + zero_dev_small_number)

        disparity = self_dem_parity - every_other_dem_parity
        # breakpoint()
        aggregated_disparity_over_classes = reduce_over(disparity, dim=1, aggr=rule_over_classes)
        aggregated_disparity_over_subgroup = reduce_over(aggregated_disparity_over_classes, dim=0, aggr=rule_over_subgroup)
        if aggregated_disparity_over_subgroup < 0:
            breakpoint()
        return aggregated_disparity_over_subgroup

    def dem_parity_loss(self, inputs, sensitives):
        # Does not use the ground truth label (targets)
        sensitive_group_count = sensitives.sum(axis=0)
        # breakpoint()
        prediction_one_hot_over_classes = F.softmax(inputs/self.temperatrue, dim=1)
        per_class_pos_classified_group_count  = (sensitives.T.matmul(prediction_one_hot_over_classes)).T
        dem_parity_loss = CEWithDemParityLoss.calculate_fairness_gaps(sensitive_group_count, per_class_pos_classified_group_count, rule_over_classes="max", rule_over_subgroup="max")
        return dem_parity_loss


    def forward(self, input: Tensor, target: Tensor, sensitive: Tensor) -> Tensor:
        CE_loss = super().forward(input, target)
        DP_loss = self.dem_parity_loss(input, sensitive)
        # breakpoint()
        return CE_loss + self._lambda * (DP_loss - self._gamma)


class CEWithDemParityLossPub(CEWithDemParityLoss):
    # this is the same as CEWithDemParityLoss but using the public dataset to calculate DemParityLoss

    def __init__(self, _gamma, _lambda=1, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0) -> None:
        super().__init__(_gamma=_gamma, _lambda=_lambda, weight=weight)

    def forward(self, input: Tensor, target: Tensor, input_pub: Tensor, sensitive_pub: Tensor, DP_loss=None) -> Tensor:
        CE_loss = super(CEWithDemParityLoss, self).forward(input, target)
        if DP_loss == None:
            DP_loss = self.dem_parity_loss(input_pub, sensitive_pub)
        # breakpoint()
        return CE_loss + self._lambda * (DP_loss - self._gamma), DP_loss

    