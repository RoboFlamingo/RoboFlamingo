import os
import argparse

ckpt_root = '/' + os.getcwd().split('/')[1] + '/dmh'

ckpt_list = {
    "roboflamingo": {
        "dir": "hydra/checkpoints",
        "ckpt": "RoboFlamingo_cross_mambablock_compress_type_linear_lr3e-5_Step42500_tokenizer_grad8_window_size_12_mamba_1.4b_hf_post.pth",
        "script": "robot_flamingo/pt_eval_ckpts.bash"
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="roboflamingo") # robohydra(ours) roboflamingo
args = parser.parse_args()

assert args.task in ckpt_list.keys(), f"the task {args.task} is not supported yet."

task_info = ckpt_list[args.task]

ckpt_dir = os.path.join(ckpt_root, task_info['dir'])

ckpt_names = [task_info['ckpt']]

# ckpt_names = ['lstm_head_robomamba_lr1e-4_multistep_1_epoch1_step10000_tokenizer_non-state_post.pth']
if "pool" in task_info['ckpt']:
    compress_type="POOL"
elif "linear" in task_info['ckpt']:
    compress_type="LINEAR"
else:
    raise f"COMPRESS_TYPE should be LINEAR or POOL."
print(ckpt_names)
for ckpt_name in ckpt_names:
    use_gripper = 1 if 'gripper' in ckpt_name else 0
    use_state = 1 if 'state' in ckpt_name else 0
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    os.makedirs('logs_new', exist_ok=True)
    log_file = 'logs_new/evaluate_{}.log'.format(ckpt_name.split('.')[0])
    ckpt_ix = ckpt_names.index(ckpt_name)
    print('evaluating {}/{} checkpoint'.format(ckpt_ix+1, len(ckpt_names)))
    fusion_mode = 'pre'
    if 'post' in ckpt_name:
        fusion_mode = 'post'
    if 'two_way' in ckpt_name:
        fusion_mode = 'two_way'
    window_size = 12
    ckpt_attrs = ckpt_name.split('_')
    if 'ws' in ckpt_attrs:
        window_size = int(ckpt_attrs[ckpt_attrs.index('ws')+1])
    # print('bash robot_flamingo/pt_eval_ckpts.bash {} {} {} {}'.format(ckpt_path, log_file, use_gripper, use_state))
    # exit(0)
    node_num = 1
    if 'mpt_9b' in ckpt_name:
        node_num = 5
    os.system('bash {} {} {} {} {} {} {} {} {}'.format(task_info['script'], ckpt_path, log_file, use_gripper, use_state, fusion_mode, window_size, node_num, compress_type))

