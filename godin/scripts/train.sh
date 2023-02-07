python main.py --in_dataset CIFAR100 --architecture resnet --weight_decay 0.0005 --batch_size 128 --epochs 200 --gpu 1 --train 'True'

python main.py --in_dataset CIFAR10 --out_dataset SVHN --architecture densenet --similarity cosine --weight_decay 0.0001 --batch_size 64 --epochs 300 --gpu 0

python main.py --in_dataset CIFAR10 --out_dataset SVHN --architecture resnet --similarity cosine --weight_decay 0.0005 --batch_size 128 --epochs 200 --gpu 1

python main.py --in_dataset CIFAR10 --out_dataset SVHN --architecture wideresnet --similarity cosine --weight_decay 0.0005 --batch_size 128 --epochs 200 --gpu 2

python main.py --in_dataset CIFAR100 --out_dataset SVHN --architecture densenet --similarity cosine --weight_decay 0.0001 --batch_size 64 --epochs 300 --gpu 0

python main.py --in_dataset CIFAR100 --out_dataset SVHN --architecture resnet --similarity cosine --weight_decay 0.0005 --batch_size 128 --epochs 200 --gpu 1

python main.py --in_dataset CIFAR100 --out_dataset SVHN --architecture wideresnet --similarity cosine --weight_decay 0.0005 --batch_size 128 --epochs 200 --gpu 2

PYTHONPATH=../ python main.py --in_dataset CIFAR10 --out_dataset SVHN --architecture resnet --similarity cosine --weight_decay 0.0005 --batch_size 128 --epochs 300 --gpu 2 --train 'True' --test 'True' --num_workers 4
