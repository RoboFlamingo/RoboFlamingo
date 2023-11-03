""" Main training script """

import argparse
import glob
import os
import random
from robot_flamingo.eval.eval_utils import eval_one_epoch_calvin_ddp
from torch.distributed.elastic.multiprocessing.errors import record

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import numpy as np
import torch
import wandb
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP

from robot_flamingo.data.data import get_data
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from eval_utils import eval_one_epoch_calvin, eval_one_epoch_calvin_ddp
from robot_flamingo.models.factory import create_model_and_transforms, mpt_dict


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=4,
        help="how often to add a cross-attention layer after each transformer layer",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="RobotFlamingo",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size_calvin", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--openflamingo_checkpoint", type=str, default="")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)  # 1e-4
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument(
        "--calvin_dataset",
        type=str,
        help="path to calvin_dataset",
    )
    parser.add_argument("--loss_multiplier_calvin", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--evaluate_from_checkpoint",
        type=str,
        help="path to checkpoint to evaluate , this should contain model",
        default=None,
    )
    # data args
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples_calvin", type=int, default=100)
    parser.add_argument("--dataset_resampled", action="store_true")
    parser.add_argument("--calvin_conf_path", type=str, help="path to calvin configuration file")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )
    parser.add_argument(
        "--freeze_embed",
        default=False,
        action="store_true",
        help="freeze the parameters of embedding layer",
    )
    parser.add_argument(
        "--use_gripper",
        default=False,
        action="store_true",
        help="whether to use gripper image as input",
    )
    parser.add_argument(
        "--use_state",
        default=False,
        action="store_true",
        help="whether to use low-dim state as input",
    )
    parser.add_argument(
        "--fusion_mode",
        default="post",
        type=str,
        help="pre or post to fusion multi vision info",
    )
    parser.add_argument("--hist_window", type=int, default=1)  # input history window size for the model
    # history window size when evaluating, for FC head equals to hist_window, for LSTM head means refresh frequency
    parser.add_argument("--eval_hist_size", type=int, default=-1)
    parser.add_argument(
        "--sep_resampler",
        default=False,
        action="store_true",
        help="whether use separate resamplers for third party and gripper camera",
    )
    parser.add_argument("--train_params", type=int, default=-1)
    parser.add_argument('--rgb_pad', type=int, default=-1)
    parser.add_argument('--gripper_pad', type=int, default=-1)
    parser.add_argument('--n_timesteps', type=int, default=150, help="diffusion time steps")
    parser.add_argument(
        "--predict_epsilon",
        default=False,
        action="store_true",
        help="whether diffusion model should predict epsilon",
    )
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument('--head_type', type=str, default="lstm")  # diffusion
    parser.add_argument(
        "--from_scratch",
        default=False,
        action="store_true",
        help="whether to train the model from scratch",
    )
    parser.add_argument("--n_obs_steps", default=6, type=int)
    parser.add_argument("--future_act_len", default=-1, type=int)
    parser.add_argument("--diff_horizon", default=32, type=int)
    parser.add_argument(
        "--last_action",
        default=False,
        action="store_true",
        help="whether using last action as input",
    )
    parser.add_argument(
        "--use_hist",
        default=False,
        action="store_true",
        help="whether using multi-image encoder"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--visualize",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--reset",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--sep_lm_head",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--clip_state",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--convert_rgb",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--diverse_inst",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--residual",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--tcp_rel",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--replan",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--freeze_sampler",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--fwd_pred",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--fwd_pred_hand",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--no_image_patch",
        default=False,
        action="store_true"
    )
    parser.add_argument("--global_latent", type=int, default=1)
    parser.add_argument("--save_every_iter", type=int, default=-1)
    parser.add_argument("--pad_length", type=int, default=-1)
    # For GPT decoder
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--decoder_type", type=str, default='lstm')

    parser.add_argument("--min_window_size", type=int, default=12)
    parser.add_argument("--max_window_size", type=int, default=24)
    parser.add_argument("--llm_name", type=str, default='llama_9b')
    parser.add_argument("--pooling", type=str, default='max')
    parser.add_argument("--multi_step_action", type=int, default=1, help="multiple step action prediction")


    args = parser.parse_args()
    
    if args.head_type == "diffusion":
        args.pad_length = args.n_obs_steps
    if args.eval_hist_size == -1:
        args.eval_hist_size = args.window_size
        if args.head_type == "diffusion":
            args.eval_hist_size = args.n_obs_steps
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if 'sep' in args.evaluate_from_checkpoint:
        args.sep_resampler = True
    if 'lm_head' in args.evaluate_from_checkpoint:
        args.sep_lm_head = True
    if 'res_' in args.evaluate_from_checkpoint:
        args.residual = True
    if 'tcp' in args.evaluate_from_checkpoint:
        args.tcp_rel = True
    if 'fur' in args.evaluate_from_checkpoint.split('_'):
        name_attrs = args.evaluate_from_checkpoint.split('_')
        args.multi_step_action = int(name_attrs[name_attrs.index('fur')-1])
    if 'difws' in args.evaluate_from_checkpoint:
        args.dif_ws = True
        name_attrs = args.evaluate_from_checkpoint.split('_')
        ix = name_attrs.index('difws')
        min_ws = int(name_attrs[ix+1])
        max_ws = int(name_attrs[ix+2])
        args.min_window_size = min_ws
        args.max_window_size = max_ws
        args.window_size = max_ws
    if 'latent' in args.evaluate_from_checkpoint:
        name_attrs = args.evaluate_from_checkpoint.split('_')
        ix = name_attrs.index('latent')
        args.global_latent = int(name_attrs[ix+1])
    if 'no_image_patch' in args.evaluate_from_checkpoint:
        args.no_image_patch = True
    if 'gpt' in args.evaluate_from_checkpoint:
        args.decoder_type = 'gpt'
        name_attrs = args.evaluate_from_checkpoint.split('_')
        hidden_size = int(name_attrs[name_attrs.index('gpt')+1])
        args.hidden_size = hidden_size
    for name in ['mpt_3b', 'mpt_4b', 'mpt_9b', 'mpt_dolly_3b', 'mpt_base_4b']:
        if name in args.evaluate_from_checkpoint:
            args.llm_name = name
            break
    
    args.lm_path = mpt_dict[args.llm_name]["lang_encoder_path"]
    args.tokenizer_path = mpt_dict[args.llm_name]["tokenizer_path"]
    args.cross_attn_every_n_layers = mpt_dict[args.llm_name]["cross_attn_every_n_layers"]
    args.openflamingo_checkpoint = mpt_dict[args.llm_name]["openflamingo_checkpoint"]
    
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    print("world_size: ", torch.distributed.get_world_size())
    random_seed(args.seed)

    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        use_media_placement_augmentation=args.use_media_placement_augmentation,
        window_size=args.eval_hist_size,
        freeze_embed=args.freeze_embed,
        train_params=args.train_params,
        sep_resampler=args.sep_resampler,
        last_action=args.last_action,
        use_diff=(args.head_type == "diffusion"),
        n_timesteps=args.n_timesteps,
        diff_horizon=args.diff_horizon,
        fusion_mode=args.fusion_mode,
        use_gripper=args.use_gripper,
        use_state=args.use_state,
        use_hist=args.use_hist,
        pad_length=args.pad_length,
        debug=args.debug,
        multi_step_action=args.multi_step_action,
        llm_name=args.llm_name,
        sep_lm_head=args.sep_lm_head,
        return_feature=True,
        residual=args.residual,
        tcp_rel=args.tcp_rel,
        replan=args.replan,
        decoder_type=args.decoder_type,
        hidden_size=args.hidden_size,
        freeze_sampler=args.freeze_sampler,
        fwd_pred=args.fwd_pred,
        fwd_pred_hand=args.fwd_pred_hand,
        no_image_patch=args.no_image_patch,
        global_latent=args.global_latent,
        # refresh=args.refresh
    )
    checkpoint_path = args.openflamingo_checkpoint
    print("Loading origin flamingo checkpoint from ", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    if args.sep_lm_head:
        model.lm_head.requires_grad_(True)
    else:
        model.lang_encoder.lm_head.requires_grad_(True)

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    else:
        model = model.float()
    model = model.to(device_id)
    model.eval()

    ddp_model = DDP(model, device_ids=[device_id])
    if args.residual:
        model.lang_encoder.clone_parameters()
    # if args.evaluate_from_checkpoint is specified, load checkpoint
    assert args.evaluate_from_checkpoint is not None, "Please specify a checkpoint to evaluate."
    if args.rank == 0:
        print(f"Loading robot-flamingo checkpoint from {args.evaluate_from_checkpoint}")
    checkpoint = torch.load(args.evaluate_from_checkpoint, map_location="cpu")
    ddp_model.load_state_dict(checkpoint["model_state_dict"], False)  # 只保存了求梯度的部分

    ddp_model.eval()
    eval_log_dir = None
    if args.visualize:
        eval_log_dir = 'evaluate/{}'.format(args.evaluate_from_checkpoint.split('.')[0])
    eval_one_epoch_calvin_ddp(
        args=args,
        model=ddp_model,
        image_processor=image_processor,
        tokenizer=tokenizer,
        dataset_path=args.calvin_dataset,
        future_act_len=args.future_act_len,
        eval_log_dir=eval_log_dir,
        debug=args.visualize,
        reset=args.reset,
        diverse_inst=args.diverse_inst
    )


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = '1'
    main()
