import numpy as np
import os
from sklearn import metrics

from general_utils.functions import sigmoid
from general_utils.save_load import load_obj


def main():
    print(os.getcwd())
    # dataset = 'celeba'
    dataset = 'cxpert'
    task_outputs = load_obj(file=f'./outputs_raw_{dataset}.npy')
    task_targets = load_obj(file=f'./targets_raw_{dataset}.npy')

    print('threshold,accuracy')
    for threshold in np.linspace(0.0, 1.0, 100):
        accs = []
        for task in range(len(task_targets)):
            targets = task_targets[task]
            outputs = task_outputs[task]
            assert not np.any(np.isnan(targets))
            if len(targets) > 0:
                preds = sigmoid(outputs) > threshold
                acc = metrics.accuracy_score(y_pred=preds, y_true=targets)
                accs.append(acc)
        print(threshold, ',', np.mean(accs))


if __name__ == "__main__":
    main()
