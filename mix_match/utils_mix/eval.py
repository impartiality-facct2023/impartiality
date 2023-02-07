from __future__ import print_function, absolute_import
from sklearn import metrics
__all__ = ['accuracy']

# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         # print("size", correct.size())
#         # print("correct", correct)
#         # print("correctk", correct[:k])
#         correct_k = correct[:k].view(-1).float().sum(0)   # .view(-1)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

def accuracy(output, target, topk=None):
    """Computes the accuracy"""
    batch_size = target.size(0)
    #print("target", target)
    output = (output > 0.5).int()  # Replace to work with other labels as well.
    #print("output", output)
    target = target.cpu()
    output = output.cpu()
    acc = []
    acc.append(metrics.accuracy_score(y_true=target, y_pred=output))
    # _, pred = output.topk(maxk, 1, True, True)
    # pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))

    # res = []
    # for k in topk:
    #     # print("size", correct.size())
    #     # print("correct", correct)
    #     # print("correctk", correct[:k])
    #     correct_k = correct[:k].view(-1).float().sum(0)   # .view(-1)
    #     res.append(correct_k.mul_(100.0 / batch_size))
    return acc

def computeauc(outputs, targets):
    targets = targets.cpu()
    outputs = outputs.cpu()
    auc = metrics.roc_auc_score(y_true=targets, y_score=outputs)
    return auc