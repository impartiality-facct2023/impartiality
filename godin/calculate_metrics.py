import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def calc_auroc(id_test_results, ood_test_results):
    # calculate the AUROC
    scores = np.concatenate((id_test_results, ood_test_results))
    print(scores)
    trues = np.array(
        ([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    result = roc_auc_score(trues, scores)

    return result


def calc_tnr(id_test_results, ood_test_results):
    scores = np.concatenate((id_test_results, ood_test_results))
    trues = np.array(
        ([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    fpr, tpr, thresholds = roc_curve(trues, scores)
    return 1 - fpr[np.argmax(tpr >= .95)]