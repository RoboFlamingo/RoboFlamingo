#!/bin/bash

device="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES=$device


# 获取当前工作目录
current_dir=$(pwd)
# 提取路径的第二部分并构造 ckpt_root
second_dir=$(echo "$current_dir" | cut -d"/" -f2)


export SSL_CERT_FILE=/$second_dir/dmh/anaconda3/envs/roboma/ssl/cert.pem
export SSL_CERT_DIR=/$second_dir/dmh/anaconda3/envs/roboma/ssl
export openssl_capath=/$second_dir/dmh/anaconda3/envs/roboma/ssl


# export PATH=$PATH:path/to/robot-flamingo/robot_flamingo
# export PYTHONPATH=$PYTHONPATH:path/to/robot-flamingo/robot_flamingo
export PATH=$PATH:/$second_dir/dmh/cobra/cobra/cobra/cobra
export PYTHONPATH=$PYTHONPATH:/$second_dir/dmh/cobra/cobra/cobra/cobra
export PATH=$PATH:/$second_dir/dmh/cobra/cobra/cobra/cobra/robot_flamingo
export PYTHONPATH=$PYTHONPATH:/$second_dir/dmh/cobra/cobra/cobra/cobra/robot_flamingo
export PATH=$PATH:/$second_dir/dmh/hydra/open_flamingo
export PYTHONPATH=$PYTHONPATH:/$second_dir/dmh/hydra/open_flamingo

# dataset path
calvin_dataset_path="/$second_dir/dmh/cobra/cobra/cobra/dataset/task_D_D"
# language model path
# lm_path="/$second_dir/dmh/cobra/cobra/cobra/dataset/mpt-1b-redpajama-200b-dolly"
lm_path="/$second_dir/dmh/cobra/cobra/cobra/dataset/mpt-1b-redpajama-200b-dolly"
# tokenizer path
tokenizer_path="/$second_dir/dmh/cobra/cobra/cobra/dataset/mpt-1b-redpajama-200b-dolly"
# openflamingo ckpt path
openflamingo_checkpoint="/$second_dir/dmh/cobra/cobra/cobra/dataset/MPT/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_3b_4.pth"

subfix=`date "+%Y%m%d-%H%M"`
log_file="logs/training_"${subfix}".log"
# source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt
#python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=2  --master_port=6042 robot_flamingo/train/train_calvin.py \
# COMPRESS_TOKEN_TYPE表示压缩token的方式，有LINEAR、POOL两种
COMPRESS_TOKEN_TYPE=LINEAR torchrun --nnodes=1 --nproc_per_node=8 --master_port=6042 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mamba_2.8b_hf \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 10 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 4 \
    --run_name RobotFlamingoDBG \
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
    --learning_rate 3e-5 \
    --save_every_iter 2500 \
    --from_scratch \
    --window_size 12 
    # > ${log_file} 2>&1
