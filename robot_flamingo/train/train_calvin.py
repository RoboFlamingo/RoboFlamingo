""" Main training script """

import argparse
import copy
import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
from huggingface_hub import hf_hub_download
import itertools
from torch.nn.parallel import DistributedDataParallel as DDP

from robot_flamingo.data.data import get_data
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from train_utils import get_checkpoint, train_one_epoch_calvin, train_one_epoch_calvin_diff, train_one_epoch_calvin_cotrain, train_one_epoch_calvin_two_way, \
get_ckpt_name, get_ckpt_name_pattern
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from robot_flamingo.models.factory import create_model_and_transforms, mpt_dict


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

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
    parser.add_argument("--window_size", type=int, default=32)
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
    # hot fix for torch.distributed.launch
    # parser.add_argument("--local-rank", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples_calvin", type=int, default=100)
    parser.add_argument("--dataset_resampled", action="store_true")
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
        default="pre",
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
    parser.add_argument('--head_type', type=str, default="lstm") # diffusion
    parser.add_argument(
        "--from_scratch",
        default=False,
        action="store_true",
        help="whether to train the model from scratch",
    )
    parser.add_argument("--n_obs_steps", default=6, type=int)
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
        action="store_true"
    )
    parser.add_argument(
        "--traj_cons",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--debug",
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
        "--unfreeze_vit",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--text_aug",
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
        "--dif_ws",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--partial_data",
        default=False,
        action="store_true"
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
        "--no_pretrain",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--real_data",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--no_image_patch",
        default=False,
        action="store_true"
    )
    # Co-Train settings
    parser.add_argument(
        "--cotrain",
        default=False,
        action="store_true"
    )
    parser.add_argument("--batch_size_vl", type=int, default=20)
    parser.add_argument("--vl_task_weights", type=float, default=0.005)

    parser.add_argument("--global_latent", type=int, default=1)
    parser.add_argument("--save_every_iter", type=int, default=-1)
    # For GPT decoder
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--decoder_type", type=str, default='lstm')
    
    parser.add_argument("--min_window_size", type=int, default=12)
    parser.add_argument("--max_window_size", type=int, default=24)
    parser.add_argument("--llm_name", type=str, default='llama_9b')
    parser.add_argument("--pooling", type=str, default='max')
    parser.add_argument("--multi_step_action", type=int, default=1, help="multiple step action prediction")

    args = parser.parse_args()
    
    if args.eval_hist_size == -1:
        args.eval_hist_size = args.window_size
        if args.head_type == "diffusion":
            args.eval_hist_size = args.n_obs_steps
    if args.tcp_rel:
        args.clip_state = True
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    device_id = init_distributed_device(args)
    print("device_id: ", device_id)

    random_seed(args.seed)
    args.lm_path = mpt_dict[args.llm_name]["lang_encoder_path"]
    args.tokenizer_path = mpt_dict[args.llm_name]["tokenizer_path"]
    args.cross_attn_every_n_layers = mpt_dict[args.llm_name]["cross_attn_every_n_layers"]
    args.openflamingo_checkpoint = mpt_dict[args.llm_name]["openflamingo_checkpoint"]

    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_gripper=args.use_gripper,
        use_state=args.use_state,
        use_hist=args.use_hist,
        fusion_mode=args.fusion_mode,
        use_local_files=args.offline,
        use_media_placement_augmentation=args.use_media_placement_augmentation,
        window_size=args.eval_hist_size,
        freeze_embed=args.freeze_embed,
        train_params=args.train_params,
        sep_resampler=args.sep_resampler,
        last_action=args.last_action,
        use_diff=(args.head_type == "diffusion"), # Diff still have bugs of loaded data mismatch
        n_timesteps=args.n_timesteps,
        diff_horizon=args.diff_horizon,
        predict_epsilon=args.predict_epsilon,
        sep_lm_head=args.sep_lm_head,
        unfreeze_vit=args.unfreeze_vit,
        multi_step_action=args.multi_step_action,
        llm_name=args.llm_name,
        pooling=args.pooling,
        residual=args.residual,
        tcp_rel=args.tcp_rel,
        decoder_type=args.decoder_type,
        hidden_size=args.hidden_size,
        freeze_sampler=args.freeze_sampler,
        fwd_pred=args.fwd_pred,
        fwd_pred_hand=args.fwd_pred_hand,
        no_image_patch=args.no_image_patch,
        global_latent=args.global_latent,
    )

    checkpoint_path = args.openflamingo_checkpoint
    if not args.debug and not args.no_pretrain:
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        if args.residual:
            model.lang_encoder.clone_parameters()

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    if args.debug:
        calvin_dataset = get_data(args, image_processor, tokenizer, "debug")
    elif args.real_data:
        calvin_dataset = get_data(args, image_processor, tokenizer, "real")
    else:
        calvin_dataset = get_data(args, image_processor, tokenizer, "calvin")
    
    if args.co_train:
        coco_loader = get_data(args, image_processor, tokenizer, "coco")
        vqa_loader = get_data(args, image_processor, tokenizer, "vqa")
        coco_cycle_loader = itertools.cycle(coco_loader)
        vqa_cycle_loader = itertools.cycle(vqa_loader)

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

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
    if args.head_type == "diffusion" and (not args.debug):
        normalizer = model.diffusion_model.normalizer
        all_actions = np.vstack([calvin_dataset.dataset.__getitem__((i,1),True)["actions"] for i in range(0,10000)])
        normalizer.fit(all_actions, last_n_dims=1, mode='limits')

    model = model.to(device_id)

    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                "gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x
            )

        for n, p in model.named_parameters():
            
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": [p for p in params_with_wd if p.requires_grad], "weight_decay": args.weight_decay},
            {"params": [p for p in params_without_wd if p.requires_grad], "weight_decay": 0.0},
        ]
    args.learning_rate = args.learning_rate * args.batch_size_calvin / 6 # adaptive lr
    optimizer = torch.optim.AdamW(get_grouped_params(ddp_model), lr=args.learning_rate)

    total_training_steps = (
        (args.train_num_samples_calvin) // (args.batch_size_calvin * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    use_diff = (args.head_type == "diffusion")
    # check if a checkpoint exists for this run

    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        ckpt_name = get_ckpt_name_pattern(args)
        checkpoint_list = glob.glob(f"{args.run_name}/{ckpt_name}")
        print(ckpt_name)
        checkpoint_list = [_ for _ in checkpoint_list if "__sep" not in _ and 'iter' not in _ and 'weights' not in _]
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
            )

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None and args.from_scratch is False:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        
        def filter_ckpt(checkpoint, skip_keys=[]):
            new_state_dict = OrderedDict()
            for key, value in checkpoint.items():
                flag = True
                for skip_key in skip_keys:
                    if skip_key in key:
                        flag = False
                        break
                if flag:
                    new_state_dict[key] = value
            return new_state_dict
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        if not args.real_data:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            resume_from_epoch = checkpoint["epoch"] + 1
    ddp_model.train()
    if args.real_data:
        resume_from_epoch = 0 
    for epoch in range(resume_from_epoch, args.num_epochs):
        calvin_dataset.set_epoch(epoch)
        calvin_loader = calvin_dataset.dataloader

        if args.head_type == "diffusion":
            train_one_epoch_calvin_diff(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
            )
        elif args.fusion_mode == 'two_way':
            train_one_epoch_calvin_two_way(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
            )
        elif args.co_train:
            train_one_epoch_calvin_cotrain(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                coco_loader=coco_cycle_loader,
                vqa_loader=vqa_cycle_loader,
                device_id=device_id,
                wandb=wandb,
            )
        else:
            train_one_epoch_calvin(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
            )

        if args.rank == 0:
            if not os.path.exists(args.run_name):
                os.makedirs(args.run_name)

            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(ddp_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }

            ckpt_name = get_ckpt_name(args, epoch)
            ckpt_path = os.path.join(args.run_name, ckpt_name)

            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint_dict, ckpt_path)
            if args.delete_previous_checkpoint:
                if epoch > 0:
                    os.remove(ckpt_path)

    if args.rank == 0:
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        ckpt_name = get_ckpt_name(args,)
        torch.save(get_checkpoint(ddp_model), f"{args.run_name}/{ckpt_name}")
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.run_name}/{ckpt_name}")


if __name__ == "__main__":
    main()
