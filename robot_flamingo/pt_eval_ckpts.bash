#!/bin/bash

export EVALUTION_ROOT=$(pwd)

# # Install dependency for calvin
# sudo apt-get -y install libegl1-mesa libegl1
# sudo apt-get -y install libgl1

# sudo apt-get update -y -qq
# sudo apt-get install -y -qq libegl1-mesa libegl1-mesa-dev

# sudo apt install -y mesa-utils libosmesa6-dev llvm
# sudo apt-get -y install meson
# sudo apt-get -y build-dep mesa



export EVALUTION_ROOT=$(pwd)

# 获取当前工作目录
current_dir=$(pwd)
# 提取路径的第二部分并构造 ckpt_root
second_dir=$(echo "$current_dir" | cut -d"/" -f2)

export SSL_CERT_FILE=/$second_dir/apj/anaconda3/envs/robomaV2/ssl/cert.pem
export SSL_CERT_DIR=/$second_dir/apj/anaconda3/envs/robomaV2/ssl
export openssl_capath=/$second_dir/apj/anaconda3/envs/robomaV2/ssl

export PATH=$PATH:/$second_dir/dmh/hydra/robot_flamingo
export PYTHONPATH=$PYTHONPATH:/$second_dir/dmh/hydra/robot_flamingo
export PATH=$PATH:/$second_dir/dmh/hydra/open_flamingo
export PYTHONPATH=$PYTHONPATH:/$second_dir/dmh/hydra/open_flamingo
export PATH=$PATH:/$second_dir/dmh/RoboFlamingo/calvin/calvin_env
export PYTHONPATH=$PYTHONPATH:/$second_dir/dmh/RoboFlamingo/calvin/calvin_env
export PATH=$PATH:/$second_dir/dmh/RoboFlamingo
export PYTHONPATH=$PYTHONPATH:/$second_dir/dmh/RoboFlamingo


# !!! Set for your own path
calvin_dataset_path="/$second_dir/dmh/cobra/cobra/cobra/dataset/task_ABCD_D"
# calvin_conf_path
calvin_conf_path="/$second_dir/dmh/RoboFlamingo/calvin/calvin_models/conf"
# language model path
lm_path="facebook/opt-30b"
# tokenizer path
tokenizer_path="facebook/opt-30b"


evaluate_from_checkpoint=$1
log_file=$2
use_gripper=$3
use_state=$4
fusion_mode=$5
window_size=$6
compress_type=$8
export MESA_GL_VERSION_OVERRIDE=4.1
echo logging to ${log_file}
node_num=1
COMPRESS_TOKEN_TYPE=$compress_type MAMBA_TYPE=MOE_MAMBA N_EXPERT=2 torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=6065 robot_flamingo/eval/eval_calvin.py \
    --precision fp32 \
    --use_gripper \
    --use_state \
    --window_size ${window_size} \
    --fusion_mode ${fusion_mode} \
    --run_name RobotFlamingoDBG \
    --calvin_dataset ${calvin_dataset_path} \
    --lm_path ${lm_path} \
    --tokenizer_path ${tokenizer_path} \
    --cross_attn_every_n_layers 4 \
    --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
    --calvin_conf_path ${calvin_conf_path} \
    --workers 1 > ${log_file} 2>&1

# if [ ${use_gripper} -eq 1 ] && [ ${use_state} -eq 1 ]
# then
# torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=6066 robot_flamingo/eval/eval_calvin.py \
#     --precision fp32 \
#     --use_gripper \
#     --use_state \
#     --window_size ${window_size} \
#     --fusion_mode ${fusion_mode} \
#     --run_name RobotFlamingoDBG \
#     --calvin_dataset ${calvin_dataset_path} \
#     --lm_path ${lm_path} \
#     --tokenizer_path ${tokenizer_path} \
#     --cross_attn_every_n_layers 4 \
#     --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
#     --calvin_conf_path ${calvin_conf_path} \
#     --workers 1 > ${log_file} 2>&1
# fi

# if [ ${use_gripper} -eq 1 ] && [ ${use_state} -eq 0 ]
# then
# torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=6099 robot_flamingo/eval/eval_calvin.py \
#     --precision fp32 \
#     --use_gripper \
#     --window_size ${window_size} \
#     --fusion_mode ${fusion_mode} \
#     --run_name RobotFlamingoDBG \
#     --calvin_dataset ${calvin_dataset_path} \
#     --lm_path ${lm_path} \
#     --tokenizer_path ${tokenizer_path} \
#     --cross_attn_every_n_layers 4 \
#     --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
#     --calvin_conf_path ${calvin_conf_path} \
#     --workers 1 > ${log_file} 2>&1
# fi

# if [ ${use_gripper} -eq 0 ] && [ ${use_state} -eq 0 ]
# then
# torchrun --nnodes=1 --nproc_per_node=${node_num}  --master_port=6066 robot_flamingo/eval/eval_calvin.py \
#     --precision fp32 \
#     --run_name RobotFlamingoDBG \
#     --window_size ${window_size} \
#     --fusion_mode ${fusion_mode} \
#     --calvin_dataset ${calvin_dataset_path} \
#     --lm_path ${lm_path} \
#     --tokenizer_path ${tokenizer_path} \
#     --cross_attn_every_n_layers 4 \
#     --evaluate_from_checkpoint ${evaluate_from_checkpoint} \
#     --calvin_conf_path ${calvin_conf_path} \
#     --workers 1 > ${log_file} 2>&1
# fi