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

file_name='utkface_students.csv'
#echo "achieved epsilon,fairness gap,query fairness gap,number answered,student accuracy,coverage,sigma1,sigma2,threshold" >> ${file_name}

threshold=(50)
fairness_threshold=(0.05)
sigma1=(40)
sigma2=(15)
budget=(7)

timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='utkface'
architecture='resnet50_pretrained'
num_models=100
command="train_student_model"

for t in ${!threshold[@]}
do
    for ft in ${!fairness_threshold[@]}
    do
        for s1 in ${!sigma1[@]}
        do
            for s2 in ${!sigma2[@]}
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
                    --num_epochs 15 \
                    --batch_size 100 \
                    --architecture ${architecture} \
                    --commands ${command} \
                    --eval_batch_size 1000 \
                    --sigma_gnmax ${sigma2[$s2]} \
                    --threshold ${threshold[$t]}\
                    --sigma_threshold ${sigma1[$s1]} \
                    --budget ${budget[$b]} \
                    --lr 0.00005 \
                    --optimizer 'Adam' \
                    --weight_decay 0.0001 \
                    --scheduler_type 'ReduceLROnPlateau' \
                    --loss_type 'BCEWithLogits' \
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
                    --max_fairness_violation ${fairness_threshold[$ft]} \
                    --has_sensitive_attribute 'True' \
                    --model_size 'small' \
                    --vote_type 'discrete' \
                    --sensitive_group_list 0 1 2 3 4 \
                    --file_name ${file_name}
                done
            done
        done
    done
done

echo "Job end at $(date)"

~