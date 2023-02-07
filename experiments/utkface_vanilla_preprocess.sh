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

source ~/miniconda_nic3/bin/activate 


file_name='utkface_vanilla_pate_preprocess.csv'

fairness_threshold=(0.01 0.02 0.05 0.1 0.15 0.2 0.25)
budget=(1 2 4 6 8 10)


threshold=(50)
sigma1=(40)
sigma2=(15)


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='utkface'
architecture='resnet50_pretrained'
num_models=100
command="train_student_vanilla_pate"

for ft in ${!fairness_factor[@]}
do
    for b in ${!budget[@]}
    do
        log_file=${command}_${DATASET}_${timestamp}.txt
        CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python ~/capc-learning/main.py \
        --path / \
        --data_dir / \
        --dataset ${DATASET} \
        --num_querying_parties 1 \
        --num_models ${num_models} \
        --num_epochs 30 \
        --architecture ${architecture} \
        --commands ${command} \
        --eval_batch_size 1000 \
        --sigma_gnmax ${sigma2} \
        --threshold ${threshold} \
        --sigma_threshold ${sigma1} \
        --budget ${budget[$b]} \
        --lr 0.00005 \
        --optimizer 'Adam' \
        --weight_decay 0.0001 \
        --scheduler_type 'ReduceLROnPlateau' \
        --loss_type 'CEWithDemParityLoss' \
        --mode 'random' \
        --class_type 'multiclass' \
        --device_ids 0 \
        --momentum 0.9 \
        --weak_classes '' \
        --chexpert_dataset_type 'pos' \
        --log_every_epoch 1 \
        --debug 'False' \
        --xray_views '' \
        --query_set_type 'raw' \
        --pick_labels -1 \
        --transfer_type '' \
        --test_models_type 'private' \
        --min_group_count 20 \
        --max_fairness_violation 1 \
        --sensitive_group_list 0 1 2 3 4 \
        --has_sensitive_attribute 'True' \
        --model_size 'big' \
        --vote_type 'discrete' \
        --batch_size 100 \
        --file_name ${file_name} \
        --preprocess 1
        
    done
done

echo "Job end at $(date)"

~

# JOBID
# 17543_[1-30]
# 25702
# 25707 