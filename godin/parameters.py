import argparse

from getpass import getuser

user = getuser()


def print_args(args, get_str=False):
    if "delimiter" in args:
        delimiter = args.delimiter
    elif "sep" in args:
        delimiter = args.sep
    else:
        delimiter = ";"
    print("###################################################################")
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
    print("######################################################")


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


def get_args():
    bool_params = []
    bool_choices = ["True", "False"]

    parser = argparse.ArgumentParser(
        description='Pytorch Detecting out-of-distribution examples in '
                    'neural networks')

    # Device arguments
    parser.add_argument('--gpu',
                        default=0,
                        type=int,
                        help='gpu index')

    parser.add_argument('--num_workers',
                        default=4,
                        type=int,
                        help='number of workers to fetch the data')

    # Model loading arguments
    parser.add_argument('--load_model',
                        # action='store_true',
                        # action='store_false',
                        type=str,
                        default='True',
                        choices=bool_choices,
                        )
    bool_params.append('load_model')

    parser.add_argument('--model_dir', default='./models', type=str,
                        help='model name for saving')

    # Architecture arguments
    parser.add_argument(
        '--architecture',
        default='resnet',
        type=str,
        help='underlying architecture (densenet | resnet | wideresnet)')
    parser.add_argument(
        '--similarity', default='cosine', type=str,
        help='similarity function for decomposed confidence '
             'numerator (cosine | inner | euclid | baseline | none)')
    parser.add_argument('--loss_type', default='ce', type=str)

    # Data loading arguments
    parser.add_argument('--data_dir', default=f'/home/{user}/data', type=str)
    parser.add_argument('--in_dataset',
                        default='CIFAR10',
                        type=str,
                        help='in-distribution dataset')
    parser.add_argument('--out_dataset',
                        # default='Imagenet',
                        default='SVHN',
                        type=str,
                        help='out-of-distribution dataset')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')

    # Training arguments
    parser.add_argument('--train',
                        # action='store_true',
                        # dest='train'
                        type=str,
                        default='False',
                        choices=bool_choices,
                        )
    bool_params.append('train')

    parser.add_argument('--weight_decay', default=0.0001, type=float,
                        help='weight decay during training')
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of epochs during training')

    # Testing arguments
    parser.add_argument('--test',
                        # action='store_false',
                        # dest='test'
                        type=str,
                        default='True',
                        choices=bool_choices,
                        )
    bool_params.append('test')

    parser.add_argument('--magnitudes', nargs='+',
                        # default=[0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08],
                        default=[0.005],
                        type=float,
                        help='perturbation magnitudes')

    parser.set_defaults(argument=True)
    args = parser.parse_args()

    for param in bool_params:
        transform_bool(args=args, param=param)

    print_args(args=args)
    return args
