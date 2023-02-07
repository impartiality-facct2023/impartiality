python main.py --gpu 1 --load_model --architecture resnet --similarity cosine --in_dataset CIFAR10 --out_dataset SVHN --batch_size 256 --train 'False' --weight_decay 0.0005 --epochs 200 --magnitudes 0 0.0025 0.005 0.01 0.02 0.04 0.08

python main.py --gpu 1 --load_model --architecture wideresnet --similarity cosine --in_dataset CIFAR10 --out_dataset SVHN --batch_size 256 --train 'False' --weight_decay 0.0005 --epochs 200 --magnitudes 0.005
