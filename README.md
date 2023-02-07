## Description of the code

The `main.py` file contains the starting point. The parameters for the program
are in `parameters.py` file.

### Examples of the pipeline:

The bash code for running the scripts can be found in the `experiments` folder.

1. Train private models:
See experiments/utkface_train_100.sh

2. Generate query-answer pairs for training PATE student models:
See experiments/utkface_query_search_100.sh

3. Train FairPATE student models:
See experiments/utkface_train_fairPATE.sh

4. Train FairDP-SGD models:
See experiments/utkface_train_fairDPSGD.sh

5. Train PATE-Pre models:
See experiments/utkface_vanilla_preprocess.sh

6. Train PATE-In models:
See experiments/utkface_vanilla_inprocess.sh


## The main implementation parts

The privacy analysis estimation function with fairness in PATE can be found in: `analysis/rdp_cumulative.py`

The demographic parity loss (DPL) implementation can be found in: `fairness/losses.py`



