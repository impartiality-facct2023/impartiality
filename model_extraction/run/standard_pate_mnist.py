from model_extraction.run.utils_run import get_standard_args
from model_extraction.main_model_extraction import run_model_extraction


def main():
    args = get_standard_args()
    args.mode = 'random'
    args.target_model = 'victim'
    args.attacker_dataset = None

    run_model_extraction(args=args)


if __name__ == "__main__":
    print('start standard pate mnist')
    main()
