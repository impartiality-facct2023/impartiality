#!/bin/bash
#SBATCH --array=0
#SBATCH --mem=80GB
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --qos=nic
#SBATCH --gres=gpu:1
#SBATCH -A nic
#SBATCH -c 4
#SBATCH --partition=nic
#SBATCH --nodelist=caballus
# Operations
echo "Job start at $(date)"

# Job steps

file_name='chexpert_vanilla_pate_preprocessing.csv'
fairness_threshold=(0.01)
budget=(10)

threshold=(50)
sigma1=(40)
sigma2=(15)


timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='utkface'
architecture='resnet50_pretrained'
num_models=1
command="train_dpsgd"

for ft in ${!fairness_threshold[@]}
do
    for b in ${!budget[@]}
    do
        log_file=${command}_${DATASET}_${timestamp}.txt
        PYTHONPATH=. python ~/capc-learning/main.py \
        --path / \
        --dataset ${DATASET} \
        --data_dir /mfsnic/datasets/ \
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
        --lr 0.005 \
        --optimizer 'SGD' \
        --weight_decay 0.0 \
        --scheduler_type 'ReduceLROnPlateau' \
        --loss_type 'CE' \
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
        --min_group_count 15 \
        --max_fairness_violation ${fairness_threshold[$ft]} \
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

# JobID
# slurm-17511