#!/bin/bash
scp -r .cache/clip ~/.cache/
export PATH=$PATH:path/to/robot-flamingo/robot_flamingo
export PYTHONPATH=$PYTHONPATH:path/to/robot-flamingo/robot_flamingo

# dataset path
calvin_dataset_path='/mnt/bn/robotics/manipulation_data/calvin_data/task_ABCD_D'
# language model path
lm_path='path/to/mpt-7b'
# tokenizer path
tokenizer_path='path/to/mpt-7b'
# openflamingo ckpt path
openflamingo_checkpoint='path/to/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt'

subfix=`date "+%Y%m%d-%H%M"`
log_file="logs/training_"${subfix}".log"
source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt
#python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=2  --master_port=6042 robot_flamingo/train/train_calvin.py \
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6042 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --co_train \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 5 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingoDBGCotrain \
    --calvin_dataset ${calvin_dataset_path} \
    --lm_path ${lm_path} \
    --tokenizer_path ${tokenizer_path} \
    --openflamingo_checkpoint ${openflamingo_checkpoint} \
    --cross_attn_every_n_layers 4 \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 > ${log_file} 2>&1
