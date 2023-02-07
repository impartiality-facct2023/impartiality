#!/bin/bash
# Testing submission

# Job parameters
#SBATCH --array=0
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH -c 4
#SBATCH --partition=t4v2
# Operations
echo "Job start at $(date)"

# Job steps
source ~/miniconda_nic3/bin/activate 

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='utkface'
architecture='resnet50_pretrained' \
num_models=100
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python ~/capc-learning/main.py \
--path / \
--data_dir / \
--dataset ${DATASET} \
--dataset_type 'balanced' \
--balance_type 'standard' \
--begin_id 0 \
--end_id ${num_models} \
--num_querying_parties 3 \
--num_models ${num_models} \
--num_epochs 35 \
--architectures ${architecture} \
--commands 'train_private_models' \
--threshold 50 \
--sigma_gnmax 7.0 \
--sigma_threshold 30 \
--budgets 20.0 \
--mode 'random' \
--lr 0.00005 \
--optimizer 'Adam' \
--weight_decay 0.00001 \
--schedule_factor 0.1 \
--scheduler_type 'ReduceLROnPlateau' \
--loss_type 'BCEWithLogits' \
--num_workers 8 \
--batch_size 200 \
--eval_batch_size 1000 \
--class_type 'multiclass' \
--device_ids 0 1 \
--momentum 0.9 \
--weak_classes '' \
--log_every_epoch 0 \
--debug 'False' \
--has_sensitive_attribute 'True'


echo "Job end at $(date)"

~