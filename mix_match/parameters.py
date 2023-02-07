import argparse
import getpass
from datasets.utils import set_dataset


def get_args():
    user = getpass.getuser()
    parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
    # Optimization options
    parser.add_argument('--epochs', default=128, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=20, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    # Checkpoints
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Miscs
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
    # Device options
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    # Method options
    parser.add_argument('--n-labeled', type=int, default=250,
                        help='Number of labeled data')
    parser.add_argument('--train-iteration', type=int, default=100,
                        help='Number of iteration per epoch')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda-u', default=75, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--dataset', default='cxpert', type=str)
    # New args
    parser.add_argument('--class_type', default='multilabel', type=str)
    parser.add_argument('--mode', default='random', type=str)
    parser.add_argument('--threshold', default=0, type=float)
    parser.add_argument('--sigma_gnmax', default=7, type=float)
    parser.add_argument('--sigma_threshold', default=0, type=float)
    parser.add_argument('--transfer_type', default='', type=str)
    parser.add_argument('--query_set_type', default='numpy', type=str)
    parser.add_argument('--budget', default=8, type=float)
    parser.add_argument(
        '--capc_dir',
        default='/',
        type=str,
        help='The path to the root of the capc-learning project folder.')
    parser.add_argument(
        '--model_type',
        default='densenet121',
        type=str,
        help='The name of the model architecture.'
    )
    parser.add_argument(
        '--dataset_type',
        default='pre',
        # default='',
        type=str,
        help="The type of the dataset. If used with the pre-trained models "
             "then choose 'pre', otherwise choose empty string ''.",
    )
    parser.add_argument(
        '--num_models',
        default=50,
        type=int,
        help='The number of private models.',
    )
    parser.add_argument(
        "--xray_views",
        type=str,
        default=["AP", "PA"],
        nargs="+",
        help="The type of the views for the chext x-ray: lateral, PA, or AP.",
    )
    parser.add_argument(
        "--data_dir", type=str, default=f"/home/  /data",
        help="path to the global data",
    )
    parser.add_argument("--data_aug", type=bool, default=True, help="")
    parser.add_argument("--data_aug_rot", type=int, default=45, help="")
    parser.add_argument("--data_aug_trans", type=float, default=0.15, help="")
    parser.add_argument("--data_aug_scale", type=float, default=0.15, help="")
    parser.add_argument(
        "--taskweights",
        default=False,
        type=bool,
        help="Assign weight to tasks/labels based on their "
             "number of nan (not a number) values.",
    )
    parser.add_argument("--label_concat", type=bool, default=False, help="")
    parser.add_argument("--label_concat_reg", type=bool, default=False, help="")
    parser.add_argument("--labelunion", type=bool, default=False, help="")
    parser.add_argument("--featurereg", type=bool, default=False, help="")
    parser.add_argument("--weightreg", type=bool, default=False, help="")
    parser.add_argument("--binary", type=bool, default=True, help="Use seperate binary classifiers for each label")
    parser.add_argument(
        "--weak_classes", type=str, default="",
        help="indices of weak classes"
    )
    parser.add_argument(
        "--xray_datasets",
        type=str,
        default=["cxpert", "padchest", "mimic", "vin"],
        nargs="+",
        help="The names of the datasets with xray-s.",
    )
    parser.add_argument(
        "--num_querying_parties",
        type=int,
        default=-1,
        help="number of parties that pose queries",
    )
    parser.add_argument(
        "--querying_party_ids",
        type=int,
        nargs="+",
        default=-1,
        help="the id of the querying party",
    )


    args = parser.parse_args()

    set_dataset(args=args)

    return args
