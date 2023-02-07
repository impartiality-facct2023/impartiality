from getpass import getuser

import argparse
import os
from argparse import ArgumentParser

import utils
from models.utils_models import model_size
from models.utils_models import set_model_size
import numpy as np



from datasets.xray.dataset_pathologies import \
    get_padchest_intersect_chexpert_indexes
from datasets.xray.dataset_pathologies import get_chexpert_indexes





def get_parameters():
    user = getuser()

    bool_params = []
    bool_choices = ["True", "False"]

    timestamp = utils.get_timestamp()

    
    

    commands = ['query_ensemble_model']
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    dataset = 'pascal'
    
    
    
    pick_labels = [0, 1, 2, 3, 4]
    
    num_querying_parties = 3
    taskweights = False
    xray_views = [""]
    xray_datasets = ["cxpert", "padchest", "mimic", "vin"]
    adam_amsgrad = False
    dataset_type = "balanced"
    
    
    balance_type = "standard"
    vote_type = "probability"
    optimizer = "SGD"
    log_every_epoch = 0
    
    debug = False
    if debug:
        num_workers = 0
    else:
        num_workers = 4
    begin_id = 0
    momentum = 0.9
    scheduler_type = "ReduceLROnPlateau"
    scheduler_milestones = None
    loss_type = "CE"

    num_models = 10
    default_model_size = None
    if num_workers > 0:
        device_ids = [0, 1, 2, 3]
        
    else:
        device_ids = [0]
        

    querying_party_ids = [0, 1, 2]

    if num_models == 1:
        threshold, sigma_threshold, sigma_gnmax = [0.01, 0.01, 0.01]
    elif num_models == 5:
        threshold, sigma_threshold, sigma_gnmax = [5.0, 3.0, 1.0]
    elif num_models == 10:
        threshold, sigma_threshold, sigma_gnmax = [10.0, 6.0, 2.0]
    elif num_models == 50:
        threshold, sigma_threshold, sigma_gnmax = [50.0, 30.0, 7.0]
    elif num_models == 100:
        threshold, sigma_threshold, sigma_gnmax = [135.0, 65.0, 25.0]
    elif num_models == 150:
        threshold, sigma_threshold, sigma_gnmax = [190.0, 110.0, 30.0]
    elif num_models == 200:
        threshold, sigma_threshold, sigma_gnmax = [245.0, 155.0, 35.0]
    elif num_models == 250:
        threshold, sigma_threshold, sigma_gnmax = [300.0, 200.0, 40.0]
    elif num_models == 300:
        threshold, sigma_threshold, sigma_gnmax = [355.0, 245.0, 50.0]
    elif num_models == 400:
        threshold, sigma_threshold, sigma_gnmax = [450.0, 300.0, 60.0]
    else:
        raise Exception(f"Unsupported number of models: {num_models}.")

    multilabel_prob_threshold = 0.5
    sigma_gnmax_private_knn = 28.0
    selection_mode = "random"
    
    private_tau = 0
    private_query_count = None
    private_tau_norm = "2"
    num_teachers_private_knn = 300

    
    
    
    
    
    sigma_gnmax_confidence = None
    bins_confidence = None

    if dataset == "colormnist":
        min_group_count = 50
        max_fairness_violation = 0.2
        sensitive_group_list = [0, 1, 2]

        momentum = 0.5
        lr = 0.1
        weight_decay = 1e-4
        batch_size = 64
        eval_batch_size = 1000
        end_id = 1
        num_epochs = 20
        num_models = 250
        
        
        num_querying_parties = 1

        selection_mode = "random"
        
        
        
        

        
        

        
        threshold = 200
        sigma_threshold = 150
        sigma_gnmax = 40.0
        sigma_gnmax_private_knn = 28.0

        
        bins_confidence = 10
        sigma_gnmax_confidence = 40.0
        

        
        

        
        
        
        
        
        

        
        
        budget = 2.5
        
        
        
        budgets = [budget]
        architecture = "MnistNetPate"
        
        weak_classes = ""
        class_type = "multiclass"
        
        dataset_type = "balanced"
        balance_type = "standard"
        vote_type = "probability"
        
        default_model_size = model_size.big
        
        num_workers = 4
        
        device_ids = [0]
  
    elif dataset == "mnist":
        momentum = 0.5
        lr = 0.1
        weight_decay = 1e-4
        batch_size = 64
        eval_batch_size = 1000
        end_id = 1
        num_epochs = 20
        num_models = 250
        
        
        num_querying_parties = 1

        selection_mode = "random"
        
        
        
        

        
        

        
        threshold = 200
        sigma_threshold = 150
        sigma_gnmax = 40.0
        sigma_gnmax_private_knn = 28.0

        
        bins_confidence = 10
        sigma_gnmax_confidence = 40.0
        

        
        

        
        
        
        
        
        

        
        
        budget = 2.5
        
        
        
        budgets = [budget]
        architecture = "MnistNetPate"
        
        weak_classes = ""
        class_type = "multiclass"
        
        dataset_type = "balanced"
        balance_type = "standard"
        vote_type = "probability"
        
        default_model_size = model_size.big
        
        num_workers = 4
        
        device_ids = [0]
    elif dataset == "fashion-mnist":
        optimizer = "Adam"
        
        if optimizer == "Adam":
            lr = 0.001
        elif optimizer == "SGD":
            lr = 0.01
            momentum = 0.5

        weight_decay = 1e-4
        batch_size = 64
        eval_batch_size = 1000
        end_id = 1
        num_epochs = 100
        num_models = 250
        threshold = 200.0
        sigma_gnmax = 40.0
        sigma_threshold = 150.0
        budget = 2.5
        num_teachers_private_knn = 300
        
        
        
        budgets = [budget]
        architecture = "FashionMnistNet"
        weak_classes = ""
        class_type = "multiclass"
        
        dataset_type = "balanced"
        
        balance_type = "standard"
        log_every_epoch = 0

        bins_confidence = 10
        sigma_gnmax_confidence = 40.0

        default_model_size = model_size.big
        
        num_workers = 6
        
        device_ids = [0, 1, 2]
        selection_mode = "random"

    elif dataset == "cifar10":
        lr = 0.01
        weight_decay = 1e-5
        batch_size = 128
        eval_batch_size = batch_size
        end_id = 50
        num_epochs = 500
        num_models = 50
        threshold = 50.0
        sigma_gnmax = 7.0
        sigma_gnmax_private_knn = 28.0
        
        
        sigma_threshold = 30.0
        budget = 20.0
        budgets = [budget]
        architecture = "ResNet12"
        
        
        weak_classes = ""
        class_type = "multiclass"
        

        selection_mode = "random"
        
        
        
        

        bins_confidence = 10
        sigma_gnmax_confidence = 7.0

        default_model_size = model_size.big
        
        num_workers = 6
        
        device_ids = [0, 1, 2]

    elif dataset == "cifar100":
        lr = 0.01
        weight_decay = 1e-4
        batch_size = 128
        eval_batch_size = batch_size
        end_id = 1
        num_epochs = 500
        num_models = 50
        threshold = 50.0
        sigma_gnmax = 7.0
        sigma_threshold = 30.0
        budget = 20.0
        budgets = [budget]
        num_teachers_private_knn = 300
        
        architecture = "VGG5"
        weak_classes = ""
        class_type = "mutliclass"
        dataset_type = "balanced"
        balance_type = "perfect"

    elif dataset == "svhn":
        lr = 0.1
        weight_decay = 1e-4
        batch_size = 128
        eval_batch_size = batch_size
        end_id = 1
        num_epochs = 200
        num_models = 250
        
        
        
        threshold = 0
        sigma_threshold = 0
        sigma_gnmax = 35.0
        sigma_gnmax_private_knn = 100
        
        budget = 3.0
        
        
        
        
        budgets = [budget]
        num_teachers_private_knn = 300
        
        architecture = "VGG7"
        
        if architecture.startswith("ResNet"):
            lr = 0.01
            weight_decay = 1e-5
            num_epochs = 300
        
        weak_classes = ""
        class_type = "multiclass"
    elif dataset == "chexpert":
        optimizer = "Adam"
        lr = 0.0001
        weight_decay = 0.0001
        batch_size = 32
        eval_batch_size = 32
        end_id = 1
        num_models = 100
        num_epochs = 300
        budget = 20.0
        
        
        
        budgets = [budget]
        
        architecture = "densenet121"
        weak_classes = ""
        if debug:
            num_workers = 0
        else:
            num_workers = 8
        chexpert_dataset_type = "pos"
        if chexpert_dataset_type in ["multi"]:
            class_type = "multilabel"
        elif chexpert_dataset_type in ["pos", "single"]:
            class_type = "multiclass"
        else:
            raise Exception(
                f"Unknown chexpert_dataset_type: {chexpert_dataset_type}.")
    elif dataset == "retinopathy":
        optimizer = "SGD"  
        lr = 3e-3  
        end_id = 1
        weight_decay = 5e-4  
        momentum = 0.9
        batch_size = 64
        eval_batch_size = 64
        num_epochs = 200
        num_models = 50  
        threshold = 50.0
        sigma_gnmax = 7.0
        sigma_threshold = 30.0
        budget = 20.0
        budgets = [budget]
        architecture = (
            "RetinoNet"
            
        )
        if architecture == "RetinoNet":
            loss_type = "MSE"

        if architecture.startswith("ResNet"):
            lr = 0.01
            weight_decay = 1e-5
            num_epochs = 300

        weak_classes = "0"
        scheduler_type = "MultiStepLR"
        scheduler_milestones = [150, 220]
        class_type = "multiclass"
    elif dataset == "celeba":
        use_tau: bool = False
        querying_party_ids = [-1]
        num_querying_parties = -1
        pick_labels = [x for x in range(20)]

        if use_tau:
            
            
            class_type = "multilabel_tau_pate"
            if class_type in ["multilabel_tau", "multilabel_tau_pate"]:
                
                private_tau_norm = "2"
                if private_tau_norm == "1":
                    private_tau = 10.0
                    threshold, sigma_threshold, sigma_gnmax = [0.0, 0.0, 12.0]
                elif private_tau_norm == "2":
                    private_tau = np.sqrt(10)
                    threshold, sigma_threshold, sigma_gnmax = [0.0, 0.0, 20.0]
                else:
                    raise Exception(
                        f"Unsupported private tau norm: {private_tau_norm}")

                
                private_query_count = None
                budget = 0.0
            elif class_type == "multilabel_tau_dep":
                
                
                private_tau = np.sqrt(10)
                
                
                private_query_count = None
                
                threshold, sigma_threshold, sigma_gnmax = [0.0, 0.0, 25.0]
                
                
                
                budget = 1000.0
            else:
                raise Exception(f"Unknown class_type: {class_type}.")
        else:
            
            class_type = 'multilabel_powerset'
            
            private_tau = None
            private_query_count = None
            
            
            
            
            threshold, sigma_threshold, sigma_gnmax = [0, 0, 1]
            budget = 20.0
            
        taskweights = False
        optimizer = "SGD"
        lr = 0.001
        weight_decay = 0.00001
        momentum = 0.9
        batch_size = 64
        eval_batch_size = 64
        num_models = 50
        
        begin_id = 0
        
        end_id = 1
        num_epochs = 100
        
        
        
        
        
        budgets = [budget]
        architecture = "CelebaNet"
        loss_type = "BCEWithLogits"
        weak_classes = ""
        if debug:
            num_workers = 0
        else:
            num_workers = 4
        
        
        log_every_epoch = 1
        
        
        
        
        multilabel_prob_threshold = [
            0.08080808080808081,
            0.26262626262626265,
            0.5555555555555556,
            0.16161616161616163,
            0.030303030303030304,
            0.10101010101010102,
            0.24242424242424243,
            0.21212121212121213,
            0.21212121212121213,
            0.18181818181818182,
            0.04040404040404041,
            0.20202020202020204,
            0.15151515151515152,
            0.05050505050505051,
            0.04040404040404041,
            0.05050505050505051,
            0.08080808080808081,
            0.06060606060606061,
            0.5151515151515152,
            0.4747474747474748,
            0.393939393939394,
            0.5151515151515152,
            0.030303030303030304,
            0.12121212121212122,
            0.8282828282828284,
            0.30303030303030304,
            0.05050505050505051,
            0.23232323232323235,
            0.05050505050505051,
            0.11111111111111112,
            0.04040404040404041,
            0.5353535353535354,
            0.20202020202020204,
            0.2828282828282829,
            0.18181818181818182,
            0.10101010101010102,
            0.3434343434343435,
            0.08080808080808081,
            0.07070707070707072,
            0.8181818181818182,
        ]
        labels_order = [
            21,
            37,
            32,
            19,
            6,
            20,
            10,
            25,
            22,
            36,
            16,
            18,
            9,
            3,
            39,
            17,
            40,
            31,
            35,
            30,
            34,
            13,
            2,
            1,
            12,
            8,
            5,
            14,
            4,
            23,
            29,
            15,
            26,
            28,
            33,
            7,
            38,
            24,
            27,
            11,
        ]
    elif dataset == "pascal":
        
        

        querying_party_ids = [-1]
        num_querying_parties = -1

        optimizer = "SGD"
        lr = 0.001
        weight_decay = 0.0001
        momentum = 0.9
        batch_size = 32
        eval_batch_size = 64
        num_models = 50
        
        begin_id = 0
        end_id = num_models
        
        
        num_epochs = 500
        
        
        
        budget = 16
        
        
        
        budgets = [budget]
        architecture = "resnet50"
        loss_type = "MultiLabelSoftMarginLoss"
        weak_classes = ""
        if debug:
            num_workers = 0
        else:
            num_workers = 8

        
        

        class_type = 'multilabel'
        
        
        
        
        powerset_tau = None
        
        
        threshold, sigma_threshold, sigma_gnmax = [0, 0, 9]

        pick_labels = [x for x in range(20)]

        
        

        
        
        
        
        
        
        
        
        private_tau_norm = None

        if private_tau_norm == "2":
            private_tau = 1.8
            
        elif private_tau_norm == "1":
            private_tau = 3.4

        log_every_epoch = 1
        
        
        
        
        multilabel_prob_threshold = [0.5]
    elif dataset == "coco":
        setting = 1
        if setting == 1:
            optimizer = "SGD"
            momentum = 0.9
            scheduler_type = "ReduceLROnPlateau"
            lr = 0.01
            
            weight_decay = 0.0
            loss_type = "BCEWithLogits"
        elif setting == 2:
            optimizer = "Adam"
            scheduler_type = "OneCycleLR"
            lr = 0.0002
            weight_decay = 0.00001
            loss_type = "AsymmetricLossOptimized"
        else:
            raise Exception(f"Unknown setting: {setting}.")
        architecture = "tresnet_m"
        batch_size = 128
        eval_batch_size = 128
        end_id = 1
        num_models = 100
        
        
        num_epochs = 100
        
        
        budget = 20.0
        
        
        budgets = [budget]

        weak_classes = ""
        if debug:
            num_workers = 0
        else:
            num_workers = 8
        class_type = "multilabel"
        log_every_epoch = 1
        
        threshold, sigma_threshold, sigma_gnmax = [1.0, 1.0, 7.0]
        multilabel_prob_threshold = 0.8
    elif dataset in xray_datasets:
        architecture = f"densenet121_{dataset}"
        
        
        

        num_querying_parties = -1
        querying_party_ids = [-1]
        
        
        taskweights = True
        adam_amsgrad = True
        xray_views = ["AP", "PA"]
        optimizer = "Adam"
        scheduler_type = "ReduceLROnPlateau"
        lr = 0.001
        
        weight_decay = 1e-5
        
        momentum = 0.9
        
        batch_size = 64
        eval_batch_size = batch_size
        
        
        
        if dataset == "cxpert":
            
            num_models = 50
        elif dataset == "padchest":
            num_models = 20
            
            
        begin_id = 0
        end_id = num_models
        
        
        
        num_epochs = 100
        
        
        budget = 20.0
        
        
        
        budgets = [budget]
        loss_type = "BCEWithLogits"
        weak_classes = ""
        if debug:
            num_workers = 0
        else:
            num_workers = 8

        log_every_epoch = 1

        if dataset == "padchest":
            
            if num_models in [1, 10, 20, 50]:
                
                multilabel_prob_threshold = [
                    0.05,
                    0.01,
                    0.15,
                    0.01,
                    0.01,
                    0.01,
                    np.nan,
                    0.04,
                    0.09,
                    0.04,
                    0.06,
                    np.nan,
                    np.nan,
                    0.02,
                    0.01,
                    0.02,
                    0.01,
                    0.01,
                ]
            else:
                raise Exception(f"Unsupported number of models: {num_models}.")
        elif dataset == "cxpert":
            
            
            multilabel_prob_threshold = [
                0.53,
                0.5,
                0.18,
                0.56,
                0.56,
                np.nan,
                0.21,
                np.nan,
                0.23,
                np.nan,
                np.nan,
                0.46,
                0.7,
                np.nan,
                np.nan,
                np.nan,
                0.2,
                0.32,
            ]
        else:
            multilabel_prob_threshold = [0.5]

        
        
        
        
        
        
        
        
        
        pick_labels = [0, 1, 2, 3, 4, 6, 8, 11, 12, 16, 17]
        pick_labels = pick_labels[:6]
        
        commands = ['query_ensemble_model']

        class_type = 'multilabel'
        
        threshold, sigma_threshold, sigma_gnmax = [0, 0, 7]

        
        
        
        
        if (
                (pick_labels == [0, 1, 2, 3, 4])
                and (num_models == 50)
                and (dataset == "padchest")
                and (architecture == "densenet121_cxpert")
        ):
            threshold, sigma_threshold, sigma_gnmax = [50, 30, 9]

        if dataset == "padchest" and num_models == 10:
            if pick_labels is None:
                
                threshold, sigma_threshold, sigma_gnmax = [50, 30, 7.0]
            elif pick_labels == [1, 4]:
                threshold, sigma_threshold, sigma_gnmax = [0.01, 0.01, 7.0]
            elif pick_labels == [9]:
                threshold, sigma_threshold, sigma_gnmax = [0.01, 0.01, 5.0]
            elif pick_labels == [0, 1, 2, 3, 4]:
                threshold, sigma_threshold, sigma_gnmax = [0.0, 0.0, 9.0]
        else:
            
            
            threshold, sigma_threshold, sigma_gnmax = [0, 0, 7]
            

        if class_type in ["multilabel_tau", "multilabel_tau_pate"]:
            
            private_tau_norm = "2"
            if private_tau_norm == "1":
                private_tau = 8
                threshold, sigma_threshold, sigma_gnmax = [0, 0, 12]
            else:
                private_tau = np.sqrt(8)
                threshold, sigma_threshold, sigma_gnmax = [0, 0, 16]
        else:
            pass
            
            
            
            
            
            
            
            
            
            
            
            

    else:
        raise Exception("Unknown dataset: {}".format(dataset))

    if debug is True:
        debug = "True"
    else:
        debug = "False"

    parser = argparse.ArgumentParser(
        description="Confidential And Private Collaborative Learning"
    )

    
    parser.add_argument(
        "--commands",
        nargs="+",
        type=str,
        
        
        default=commands,
        help="which commands to run",
    )

    parser.add_argument("--timestamp", type=str, default=timestamp,
                        help="timestamp")
    parser.add_argument(
        "--path",
        type=str,
        default=f"/home/{user}/code/capc-learning",
        help="path to the project",
    )
    parser.add_argument(
        "--data_dir", type=str, default=f"/home/{user}/data",
        help="path to the data"
    )
    
    parser.add_argument(
        "--dataset", type=str, default=dataset, help="name of the dataset"
    )
    parser.add_argument(
        "--class_type",
        type=str,
        
        
        
        
        default=class_type,
        help="The type of the classification: binary, multiclass with a "
             "single class per data item, and multilabel classification with "
             "zero or more possible classes assigned to a data item.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=dataset_type,
        
        
        help="Type of the dataset.",
    )
    parser.add_argument(
        "--balance_type",
        type=str,
        
        
        default=balance_type,
        help="Type of the balance of classes in the dataset.",
    )
    parser.add_argument(
        "--begin_id",
        type=int,
        default=begin_id,
        help="train private models with id number in [begin_id, end_id)",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=end_id,
        help="train private models with id number in [begin_id, end_id)",
    )
    parser.add_argument(
        "--num_querying_parties",
        type=int,
        default=num_querying_parties,
        help="number of parties that pose queries",
    )
    parser.add_argument(
        "--querying_party_ids",
        type=int,
        nargs="+",
        default=querying_party_ids,
        help="the id of the querying party",
    )
    parser.add_argument(
        "--mode",
        type=str,
        
        
        
        
        
        default=selection_mode,
        help="method for generating utility scores",
    )
    parser.add_argument(
        "--weak_classes", type=str, default=weak_classes,
        help="indices of weak classes"
    )
    parser.add_argument(
        "--weak_class_ratio",
        type=float,
        default=0.1,
        help="ratio of samples belonging to weak classes",
    )
    parser.add_argument(
        "--verbose",
        default="True",
        
        type=str,
        choices=bool_choices,
        help="Detail info",
    )
    bool_params.append("verbose")
    parser.add_argument(
        "--debug",
        default=debug,
        
        type=str,
        choices=bool_choices,
        help="Debug mode of execution",
    )
    bool_params.append("debug")
    parser.add_argument(
        "--use_pretrained_models",
        default="False",
        
        type=str,
        choices=bool_choices,
        help="Pretrained weights for the initial training of models on private "
             "data",
    )
    bool_params.append("use_pretrained_models")
    parser.add_argument(
        "--retrain_fine_tune",
        default="False",
        
        type=str,
        choices=bool_choices,
        help="Pretrained weights for retraining models",
    )
    bool_params.append("retrain_fine_tune")
    parser.add_argument(
        "--sep", default=";", type=str, help="Separator for the output log."
    )
    parser.add_argument(
        "--log_every_epoch",
        default=log_every_epoch,
        type=int,
        help="Log test accuracy every n epchos.",
    )
    parser.add_argument(
        "--test_virtual",
        default=False,
        action="store_true",
        help="False for normal, True to train a larger qa model",
    )

    
    parser.add_argument(
        "--optimizer",
        type=str,
        default=optimizer,
        
        help="The type of the optimizer.",
    )
    parser.add_argument(
        "--adam_amsgrad",
        type=bool,
        default=adam_amsgrad,
        help="amsgrad param for Adam optimizer",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=loss_type,
        
        help="The type of the loss (e.g., MSE, CE, BCE, etc.).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=batch_size,
        help="batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=eval_batch_size,
        help="batch size for evaluation",
    )
    parser.add_argument(
        "--adaptive_batch_size",
        type=int,
        default=5,
        help="batch size for adaptive training",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="patience for adaptive training"
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="victim",
        help="steal 1 model (victim) or pate model (pate) or a "
             "different pate (another_pate)",
    )
    parser.add_argument(
        "--shuffle_dataset",
        action="store_true",
        default=False,
        help="shuffle dataset before split to train private "
             "models.  only implemented for mnist",
    )
    parser.add_argument(
        "--num_optimization_loop",
        type=int,
        default=20,
        help="num_optimization_loop for adaptive training with bayesian "
             "optimization",
    )

    parser.add_argument("--momentum", type=float, default=momentum,
                        help="SGD momentum")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=weight_decay,
        help="L2 weight decay factor",
    )
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--lr", type=float, default=lr,
                        help="initial learning rate")
    parser.add_argument(
        "--lr_factor", type=float, default=0.1,
        help="learning rate decay factor"
    )
    parser.add_argument(
        "--lr_epochs",
        type=int,
        nargs="+",
        default=[2],
        help="Epoch when learning rate decay occurs.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=num_epochs,
        help="number of epochs for training",
    )
    parser.add_argument(
        "--attacker_dataset",
        default=None,
        
        
        
        type=str,
        help="dataset used by model extraction attack, default to be the same "
             "as dataset",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        type=str,
        
        
        
        default=[architecture],
        help="The architectures of heterogeneous models.",
    )
    parser.add_argument(
        "--model_size",
        type=model_size,
        choices=list(model_size),
        default=default_model_size,
        help="The size of the model.",
    )
    parser.add_argument(
        "--device_ids",
        nargs="+",
        type=int,
        default=device_ids,
        
        
        help="Cuda visible devices.",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default=scheduler_type,
        
        
        help="Type of the scheduler.",
    )
    parser.add_argument(
        "--scheduler_milestones",
        nargs="+",
        type=int,
        default=scheduler_milestones,
        help="The milestones for the multi-step scheduler.",
    )
    parser.add_argument(
        "--schedule_factor", type=float, default=0.1,
        help="The factor for scheduler."
    )
    parser.add_argument(
        "--schedule_patience", type=int, default=10,
        help="The patience for scheduler."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=num_workers,
        help="Number of workers to fetch data.",
    )

    
    parser.add_argument(
        "--num_models", type=int, default=num_models,
        help="number of private models"
    )
    
    parser.add_argument(
        "--budget",
        type=float,
        default=budget,
        help="pre-defined epsilon value for (eps, delta)-DP",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=budgets,
        help="pre-defined epsilon value for (eps, delta)-DP",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=threshold,
        help="threshold value (a scalar) in the threshold mechanism",
    )
    
    parser.add_argument(
        "--sigma_gnmax",
        type=float,
        default=sigma_gnmax,
        help="std of the Gaussian noise in the GNMax mechanism",
    )
    parser.add_argument(
        "--sigma_gnmax_private_knn",
        type=float,
        default=sigma_gnmax_private_knn,
        help="std of the Gaussian noise in the GNMax mechanism used for the pknn cost",
    )
    parser.add_argument(
        "--sigma_threshold",
        type=float,
        default=sigma_threshold,
        help="std of the Gaussian noise in the threshold mechanism",
    )
    
    parser.add_argument(
        "--sigma_gnmax_confidence",
        type=float,
        default=sigma_gnmax_confidence,
        help="std of the Gaussian noise in the GNMax mechanism for releasing "
             "the confidence scores",
    )
    parser.add_argument(
        "--bins_confidence",
        type=int,
        default=bins_confidence,
        help="Number of confidence bins. We discretize the softmax vector by "
             "creating a histogram and mapping each element to the midpoint of "
             "the bin it belongs to.",
    )
    
    parser.add_argument(
        "--private_taus",
        nargs="+",
        type=float,
        default=[private_tau],
        help="The value of tau for the tau-approximation where we limit the "
             "sensitivity of a given teacher by limiting the positive votes to"
             "tau (in an L-norm).",
    )
    parser.add_argument(
        "--private_tau_norm",
        type=str,
        
        
        default=private_tau_norm,
        help="The norm for the tau-approximation.",
    )
    parser.add_argument(
        "--powerset_tau",
        type=int,
        default=powerset_tau,
        help="The tau value for the powerset.",
    )

    parser.add_argument(
        "--private_query_count",
        type=int,
        default=private_query_count,
        help="The number of queries to be answered privately. This is for the "
             "data independent privacy analysis_test with tau approximation.",
    )
    parser.add_argument(
        "--poisson_mechanism",
        default="False",
        type=str,
        choices=bool_choices,
        help="Apply or disable the poisson mechanism.",
    )
    bool_params.append("poisson_mechanism")

    
    parser.add_argument(
        "--multilabel_prob_threshold",
        default=multilabel_prob_threshold,
        type=float,
        nargs="+",
        help="threshold value",
    )
    parser.add_argument(
        "--coco_version", default="2017", type=str,
        help="the year of the dataset"
    )
    parser.add_argument(
        "--coco_image_size",
        default=448,
        type=int,
        help="input image size (default: 448)",
    )
    parser.add_argument(
        "--coco_data_loader",
        type=str,
        help="standard or custom data loader, where custom uses"
             "the pre-generated labels",
        default="custom",
    )
    parser.add_argument(
        "--coco_datasets",
        nargs="+",
        type=str,
        default=["train", "val"],
        
        help="Which datasets for original coco to load into the total data pool.",
    )
    parser.add_argument(
        "--coco_additional_datasets",
        nargs="+",
        type=str,
        
        default=[],
        help="Which datasets for original coco to load into the total data pool.",
    )

    
    parser = get_chexpert_paremeters(parser=parser, timestamp=timestamp)

    
    parser.add_argument("--data_aug", type=bool, default=True, help="")
    parser.add_argument("--data_aug_rot", type=int, default=45, help="")
    parser.add_argument("--data_aug_trans", type=float, default=0.15, help="")
    parser.add_argument("--data_aug_scale", type=float, default=0.15, help="")
    parser.add_argument(
        "--taskweights",
        default=taskweights,
        type=bool,
        help="Assign weight to tasks/labels based on their "
             "number of nan (not a number) values.",
    )
    parser.add_argument("--label_concat", type=bool, default=False, help="")
    parser.add_argument("--label_concat_reg", type=bool, default=False, help="")
    parser.add_argument("--labelunion", type=bool, default=False, help="")
    parser.add_argument("--featurereg", type=bool, default=False, help="")
    parser.add_argument("--weightreg", type=bool, default=False, help="")
    """
    The abbreviations PA and AP stand for posteroanterior and anteroposterior, 
    respectively. These describe the pathway of the x-rays through the patient 
    to the detector (or, in the old days, film). In a PA projection, the front 
    of the patient’s chest is against the detector and the x-rays pass through 
    the back (posterior)of the patient, through the front (anterior) of the 
    chest and then strike the detector. This is the usual projection obtained 
    in an ambulatory patient. In a patient who cannot stand, a cassette 
    containing the detector can be placed behind the patient’s back 
    (while they’re lying or sitting up in a gurney or hospital bed, 
    for example)and the exposure (often obtained with a portable x-ray unit) 
    obtained. In this scenario, the x-rays pass from the front of the patient’s 
    chest (anterior) through the back (posterior), then strike the detector, 
    yielding an AP view. From the point of view of image quality, a PA 
    projection is preferred for several reasons. For example, the portions of 
    the chest closest to the detector are the sharpest and least magnified on 
    the image. Since the heart sits in the anterior half of the chest in most 
    individuals, a more accurate representation of cardiac size and shape is 
    obtained on a PA view, compared to an AP view.
    Usually, radiologists see PA and lateral views.
    """
    parser.add_argument(
        "--xray_views",
        type=str,
        default=xray_views,
        nargs="+",
        help="The type of the views for the chext x-ray: lateral, PA, or AP.",
    )
    parser.add_argument(
        "--xray_datasets",
        type=str,
        default=xray_datasets,
        nargs="+",
        help="The names of the datasets with xray-s.",
    )

    parser.add_argument(
        "--count_noise",
        type=str,
        default="bounded",
        
        help="The type of noise added in the multiple-counting query mechanism.",
    )

    parser.add_argument(
        "--vote_type",
        type=str,
        
        
        
        default=vote_type,
        help="The type of votes. Discrete - each vote is a single number 0 or 1,"
             "or probability - the probability of a label being one.",
    )

    parser.add_argument(
        "--pick_labels",
        type=int,
        nargs="+",
        
        
        
        
        
        default=pick_labels,
        help="Which labels to limit the dataset to. Set to None to select all "
             "labels.",
    )

    parser.add_argument(
        "--query_set_type",
        type=str,
        default="raw",
        
        help="The type of query set saved for the retraining when we query the"
             "ensemble of the teacher models.",
    )

    parser.add_argument(
        "--test_models_type",
        type=str,
        
        default="private",
        help="The type of models to be tested.",
    )

    parser.add_argument(
        "--retrain_model_type",
        type=str,
        default="load",
        
        help="Should we load the private model for retraining (load) or start"
             "from scratch, i.e. from a raw model (raw).",
    )

    parser.add_argument(
        "--transfer_type",
        type=str,
        
        default="",
        help="The transfer of knowledge can be cross-domain, e.g., from the "
             "chexpert ensemble to the padchest models.",
    )

    parser.add_argument(
        "--sigmoid_op",
        type=str,
        default="apply",
        
        help="Apply or disable the sigmoid operation outside of model " "arhictecture.",
    )

    parser.add_argument(
        "--label_reweight",
        type=str,
        
        default="disable",
        help="Apply or disable the label reweighting based on the balanced "
             "accuracy found on the privately trained model.",
    )

    parser.add_argument(
        "--load_taus",
        type=str,
        
        default="disable",
        help="Apply or disable loading the taus (probability thresholds for "
             "each label) from the model checkpoint.",
    )

    parser.add_argument(
        "--show_dp_budget",
        type=str,
        default="apply",
        
        help="Apply or disable showing the current privacy budget.",
    )

    parser.add_argument(
        "--apply_data_independent_bound",
        
        default="True",
        type=str,
        choices=bool_choices,
        help="Disable it in case of the privacy estimate for " "model extraction.",
    )
    bool_params.append("apply_data_independent_bound")

    parser.add_argument(
        "--retrain_extracted_model",
        default="True",
        type=str,
        choices=bool_choices,
        help="Do we re-train the extracted / stolen model on the newly labeled data?",
    )
    bool_params.append("retrain_extracted_model")

    parser.add_argument(
        "--load_votes",
        default="True",
        
        type=str,
        choices=bool_choices,
        help="Do we re-load votes saved on disk?",
    )
    bool_params.append("load_votes")

    
    parser.add_argument('--DPSGD', type=str2bool,
                        default='False')
    parser.add_argument('--DPSGD_EPOCHS', type=int, default=10)
    parser.add_argument('--DPSGD_BATCH_SIZE', type=int, default=2)
    parser.add_argument('--DPSGD_NOISE_MULTIPLIER', type=float, default=1.3)
    parser.add_argument('--DPSGD_CCLIP', type=float, default=0)
    parser.add_argument('--DPSGD_LR', type=float, default=0.001)
    parser.add_argument('--DPSGD_PASCAL_PATH', type=str, required=False,
                        default='/VOC2012/')


    

    parser.add_argument(
        "--has_sensitive_attribute",
        default="False",
        type=str,
        choices=bool_choices,
        help="Does the dataset have a sensitive attribute",
    )
    bool_params.append("has_sensitive_attribute")

    parser.add_argument('--min_group_count', type=int, default = 50)
    parser.add_argument('--max_fairness_violation', type=float, default=0.2)
    parser.add_argument('--sensitive_group_list', type=int, nargs="+", default=[1, 2, 3], 
    help="senstive grpups labels (integers)")
    parser.add_argument('--file_name', type=str, default = '')
    parser.add_argument('--preprocess', type=bool, default = False)
    parser.add_argument('--inprocessing_fairness_penalty_scaler', type=float, default=1.0)
    parser.add_argument('--inprocessing_fairness', type=bool, default = False)

    args = parser.parse_args()
    args.cwd = os.getcwd()

    for param in bool_params:
        transform_bool(args=args, param=param)

    
    print_args(args=args)

    set_model_size(args=args)

    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def transform_bool(args, param: str):
    """
    Transform the string boolean params to python bool values.

    :param args: program args
    :param param: name of the boolean param
    """
    attr_value = getattr(args, param, None)
    if attr_value is None:
        raise Exception(f"Unknown param in args: {param}")
    if attr_value == "True":
        setattr(args, param, True)
    elif attr_value == "False":
        setattr(args, param, False)
    else:
        raise Exception(f"Unknown value for the args.{param}: {attr_value}.")


def get_chexpert_paremeters(parser: ArgumentParser, timestamp: str):
    """
    CheXpert parameters.

    :param parser: args parser
    :param timestamp: the global timestamp
    :return: parser with parameters for the CheXpert dataset.
    """

    parser.add_argument(
        "--save_path",
        default=f"./save-{timestamp}",
        metavar="SAVE_PATH",
        type=str,
        help="Path to the saved models",
    )
    parser.add_argument(
        "--pre_train",
        default=None,
        type=str,
        help="If get parameters from pretrained model",
    )
    parser.add_argument(
        "--resume", default=0, type=int, help="If resume from previous run"
    )
    parser.add_argument(
        "--logtofile",
        default=True,
        type=bool,
        help="Save log in save_path/log.txt if set True",
    )
    parser.add_argument(
        "--chexpert_dataset_type",
        
        
        default="pos",
        
        
        type=str,
        help="If get parameters from pretrained model",
    )
    parser.add_argument(
        "--nan",
        help="not a number or N/A values",
        
        type=float,
        default=np.nan,
    )
    return parser


def print_args(args, get_str=False):
    if "delimiter" in args:
        delimiter = args.delimiter
    elif "sep" in args:
        delimiter = args.sep
    else:
        delimiter = ";"

    print("args: ")
    keys = sorted(
        [
            a
            for a in dir(args)
            if not (
                a.startswith("__")
                or a.startswith("_")
                or a == "sep"
                or a == "delimiter"
        )
        ]
    )
    values = [getattr(args, key) for key in keys]
    if get_str:
        keys_str = delimiter.join([str(a) for a in keys])
        values_str = delimiter.join([str(a) for a in values])
        print(keys_str)
        print(values_str)
        return keys_str, values_str
    else:
        for key, value in zip(keys, values):
            print(key, ": ", value, flush=True)
    print("ARGS FINISHED", flush=True)

