import time
from contextlib import suppress

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from robot_flamingo.utils import world_to_tcp_frame, tcp_to_world_frame
import itertools
from einops import rearrange


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
    

def get_ckpt_name(args, epoch=-1):
    if args.use_gripper:
        ckpt_name = 'checkpoint_gripper_{}_hist_{}_{}'.format(args.fusion_mode, args.hist_window, '' if not args.sep_resampler else 'sep_')
    else:
        ckpt_name = 'checkpoint_no_gripper_hist_{}_{}'.format(args.hist_window, '' if not args.sep_resampler else 'sep_')
    if args.real_data:
        ckpt_name += 'real_'
    if args.train_params != -1:
        ckpt_name += 'train_{}_'.format(args.train_params)
    if args.no_pretrain:
        ckpt_name += 'no_pretrain_'
    if args.fwd_pred:
        ckpt_name += 'pred_rgb_'
    if args.fwd_pred_hand:
        ckpt_name += 'pred_hand_'
    if args.freeze_sampler:
        ckpt_name += 'freeze_sam_'
    if args.use_state:
        ckpt_name += 'state_'
    if args.rgb_pad != -1 or args.gripper_pad != -1:
        ckpt_name += 'aug_{}_{}_'.format(args.rgb_pad, args.gripper_pad)
    if args.use_hist:
        ckpt_name += 'fc_'
    if args.head_type == "diffusion":
        ckpt_name += 'diff_'
    if args.traj_cons:
        ckpt_name += 'traj_cons_'
    if args.sep_lm_head:
        ckpt_name += 'lm_head_'
    if args.dif_ws:
        ckpt_name += 'difws_{}_{}_'.format(args.min_window_size, args.max_window_size)
    elif args.window_size != 8:
        ckpt_name += 'ws_{}_'.format(args.window_size)
    if args.unfreeze_vit:
        ckpt_name += 'unfreeze_vit_'
    if args.llm_name != 'llama':
        ckpt_name += '{}_'.format(args.llm_name)
    if args.pooling != 'max':
        ckpt_name += '{}_'.format(args.pooling)
    if args.text_aug:
        ckpt_name += 'text_aug_'
    if args.residual:
        ckpt_name += 'res_'
    if args.freeze_embed:
        ckpt_name += 'freeze_emb_'
    if args.tcp_rel:
        ckpt_name += 'tcp_'
    if args.multi_step_action != 1:
        ckpt_name += '{}_fur_step_'.format(args.multi_step_action)
    if args.decoder_type != 'lstm':
        ckpt_name += '{}_{}_'.format(args.decoder_type, args.hidden_size)
    if args.lr_scheduler != 'constant':
        ckpt_name += '{}_'.format(args.lr_scheduler)
    ckpt_name += '{}.pth'.format(epoch)

    if epoch != -1:
        if epoch > 1000:
            ckpt_name += '{}_iter.pth'.format(epoch)
        else:
            ckpt_name += '{}.pth'.format(epoch)
    else:
        ckpt_name += 'final_weights.pth'
    return ckpt_name

def get_ckpt_name_pattern(args):
    if args.use_gripper:
        ckpt_name = 'checkpoint_gripper_{}_hist_{}_{}'.format(args.fusion_mode, args.hist_window, '' if not args.sep_resampler else 'sep_')
    else:
        ckpt_name = 'checkpoint_no_gripper_hist_{}_{}'.format(args.hist_window, '' if not args.sep_resampler else 'sep_')
    if args.real_data:
        ckpt_name += 'real_'
    if args.train_params != -1:
        ckpt_name += 'train_{}_'.format(args.train_params)
    if args.no_pretrain:
        ckpt_name += 'no_pretrain_'
    if args.fwd_pred:
        ckpt_name += 'pred_rgb_'
    if args.fwd_pred_hand:
        ckpt_name += 'pred_hand_'
    if args.freeze_sampler:
        ckpt_name += 'freeze_sam_'
    if args.use_state:
        ckpt_name += 'state_'
    if args.rgb_pad != -1 or args.gripper_pad != -1:
        ckpt_name += 'aug_{}_{}_'.format(args.rgb_pad, args.gripper_pad)
    if args.use_hist:
        ckpt_name += 'fc_'
    if args.head_type == "diffusion":
        ckpt_name += 'diff_'
    if args.traj_cons:
        ckpt_name += 'traj_cons_'
    if args.sep_lm_head:
        ckpt_name += 'lm_head_'
    if args.dif_ws:
        ckpt_name += 'difws_{}_{}_'.format(args.min_window_size, args.max_window_size)
    elif args.window_size != 8:
        ckpt_name += 'ws_{}_'.format(args.window_size)
    if args.unfreeze_vit:
        ckpt_name += 'unfreeze_vit_'
    if args.llm_name != 'llama':
        ckpt_name += '{}_'.format(args.llm_name)
    if args.pooling != 'max':
        ckpt_name += '{}_'.format(args.pooling)
    if args.text_aug:
        ckpt_name += 'text_aug_'
    if args.residual:
        ckpt_name += 'res_'
    if args.freeze_embed:
        ckpt_name += 'freeze_emb_'
    if args.tcp_rel:
        ckpt_name += 'tcp_'
    if args.multi_step_action != 1:
        ckpt_name += '{}_fur_step_'.format(args.multi_step_action)
    if args.decoder_type != 'lstm':
        ckpt_name += '{}_{}_'.format(args.decoder_type, args.hidden_size)
    if args.lr_scheduler != 'constant':
            ckpt_name += '{}_'.format(args.lr_scheduler)
    ckpt_name += '*.pth'
    return ckpt_name

def train_one_epoch_calvin_diff(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
    normalizer=None,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    if isinstance(model, DistributedDataParallel):
        diffusion_model = model.module.diffusion_model
    else:
        diffusion_model = model.diffusion_model
    
    if normalizer is None:
        normalizer = diffusion_model.normalizer

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []

    if isinstance(model, DistributedDataParallel):
        action_dim = model.module.action_head.out_features + 1 # joint + gripper
    else:
        action_dim = model.action_head.out_features + 1 # joint + gripper
 
    class LowdimMaskGenerator(nn.Module):
        def __init__(self,
            action_dim, obs_dim,
            # obs mask setup
            max_n_obs_steps=3, 
            fix_obs_steps=True, 
            # action mask
            action_visible=True,
            return_one_mask=False
            ):
            super().__init__()
            self.action_dim = action_dim
            self.obs_dim = obs_dim
            self.max_n_obs_steps = max_n_obs_steps
            self.fix_obs_steps = fix_obs_steps
            self.action_visible = action_visible
            self.return_one_mask = return_one_mask

        @torch.no_grad()
        def forward(self, shape, device, seed=None):
            # device = self.device
            B, T, D = shape
            assert D == (self.action_dim + self.obs_dim)

            # create all tensors on this device
            rng = torch.Generator(device=device)
            if seed is not None:
                rng = rng.manual_seed(seed)

            # generate dim mask
            dim_mask = torch.zeros(size=shape, 
                dtype=torch.bool, device=device)
            is_action_dim = dim_mask.clone()
            is_action_dim[...,:self.action_dim] = True
            is_obs_dim = ~is_action_dim

            # generate obs mask
            if self.fix_obs_steps:
                obs_steps = torch.full((B,), 
                fill_value=self.max_n_obs_steps, device=device)
            else:
                obs_steps = torch.randint(
                    low=1, high=self.max_n_obs_steps+1, 
                    size=(B,), generator=rng, device=device)
                
            steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
            obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
            obs_mask = obs_mask

            # generate action mask
            if self.action_visible:
                action_steps = torch.maximum(
                    obs_steps - 1, 
                    torch.tensor(0,
                        dtype=obs_steps.dtype, 
                        device=obs_steps.device))
                action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
                action_mask = action_mask & is_action_dim


            if self.return_one_mask:
                mask = obs_mask & is_obs_dim
                if self.action_visible:
                    mask = mask | action_mask
            
                return mask
            if self.obs_dim <= 0:
                assert self.fix_obs_steps, "We require fix obs steps to obtain obs masks"
                obs_mask = obs_mask[0,:,0]
            return action_mask, obs_mask     

    mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=args.n_obs_steps,
            fix_obs_steps=True,
            action_visible=True,
    )

    act_mask, obs_mask = None, None
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        # put images and labels on device
        images = (batch_calvin[0].unsqueeze(2).unsqueeze(2))
        gripper = (batch_calvin[3].unsqueeze(2).unsqueeze(2))

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        input_ids = batch_calvin[1][0].unsqueeze(1).repeat(1, images.shape[1], 1)

        # do the same to the attention mask 
        attention_mask = batch_calvin[1][1].unsqueeze(1).repeat(1, images.shape[1], 1)
        state_tensor = batch_calvin[4].unsqueeze(2).unsqueeze(2)

        actions = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        actions = normalizer.normalize(actions) # labels normalization

        if act_mask is None or obs_mask is None:
            act_mask, obs_mask = mask_generator(actions.shape, images.device)

        batch_size = actions.shape[0]
        # Mask and leave history data for generating features
        images = images[:,obs_mask,...]
        gripper = gripper[:,obs_mask,...]
        input_ids = input_ids[:,obs_mask,...]
        attention_mask = attention_mask[:,obs_mask,...]
        state_tensor = state_tensor[:,obs_mask,...]

         # put images and labels on device
        images = images.to(device_id, dtype=cast_dtype, non_blocking=True)
        gripper = gripper.to(device_id, dtype=cast_dtype, non_blocking=True)

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        input_ids = input_ids.to(device_id, non_blocking=True)

        # do the same to the attention mask 
        attention_mask = attention_mask.to(device_id, non_blocking=True)
        state_tensor = state_tensor.to(device_id, dtype=cast_dtype, non_blocking=True)

        # print("test", images.shape, gripper.shape, input_ids.shape, attention_mask.shape, state_tensor.shape)
        # import pdb; pdb.set_trace()
        
        # merge the batch and the sequence dimension
        images = images.flatten(0, 1)
        gripper = gripper.flatten(0, 1)
        state_tensor = state_tensor.flatten(0, 1)
        input_ids = input_ids.flatten(0, 1)
        attention_mask = attention_mask.flatten(0, 1)

        with autocast():
            model_out = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=gripper,
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None
            ) # Features
            model_out = model_out.logits

        # compute loss
        tt = torch.randint(0, args.n_timesteps, (batch_size,), device=actions.device).long()
        noise = torch.randn_like(actions)
        
        action_noisy = diffusion_model.q_sample(x_start=actions, t=tt, noise=noise)
 
        # apply conditioning
        action_noisy[act_mask] = actions[act_mask]
        # pred = diffusion_model(action_noisy, tt, global_cond=None)
        pred = diffusion_model(action_noisy, tt, global_cond=model_out)
        pred[act_mask] = actions[act_mask] # So we remove the gradient
        assert noise.shape == pred.shape

        if args.predict_epsilon:
            loss = F.mse_loss(pred, noise, reduction='none')
        else:
            loss = F.mse_loss(pred, actions, reduction='none')

        loss_calvin = loss.mean()

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_calvin * args.loss_multiplier_calvin
        )
        mv_avg_loss.append(loss.item())
        loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_calvin": divided_loss_calvin.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )


        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item()})


def train_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        # put images and labels on device
        images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        if args.fusion_mode != 'vit_concat':
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1)
        else:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)
        # input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)

        # do the same to the attention mask 
        if args.fusion_mode != 'vit_concat':
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1)
        else:
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True)
        
        state_tensor = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        robot_obs = batch_calvin[5].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.clip_state:
            state_tensor = torch.cat([state_tensor[..., :6], state_tensor[..., [-1]]], dim=-1)
        labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        
        state_tensor = state_tensor.unsqueeze(2).unsqueeze(2)

        # merge the batch and the sequence dimension
        images = images.flatten(0, 1)
        gripper = gripper.flatten(0, 1)
        state_tensor = state_tensor.flatten(0, 1)
        if args.fusion_mode != 'vit_concat':
            input_ids = input_ids.flatten(0, 1)
            attention_mask = attention_mask.flatten(0, 1)

        # [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        if args.use_hist:
            labels = labels[:, [-1]]  # only calculate last step action
        if args.fusion_mode == 'vit_concat':
            labels = labels[:, -1]
        labels = [labels[..., :6], (labels[..., 6:] + 1) // 2]

        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=gripper,
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None
            )

        # compute loss
        num_actions, bin_actions = output.logits[0], output.logits[1]

        # reshape for loss calculation
        if args.multi_step_action != 1:
            bs, seq_len = num_actions.shape[:2]
            num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)

        loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
        loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        if args.real_data:
            loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
        else:
            loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_calvin * args.loss_multiplier_calvin
        )
        mv_avg_loss.append(loss.item())
        loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad and m.weight.grad is not None:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        # model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_calvin": divided_loss_calvin.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )


        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f} (mse){loss_calvin_num.item():.3f} (bce){loss_calvin_bin.item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item()})

        if args.save_every_iter != -1 and global_step % args.save_every_iter == 0 and global_step > 0:
                
            if args.rank == 0:
                import os
                if not os.path.exists(args.run_name):
                    os.makedirs(args.run_name)

                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": get_checkpoint(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                }

                ckpt_name = get_ckpt_name(args, global_step)
                ckpt_path = os.path.join(args.run_name, ckpt_name)
                print(f"Saving checkpoint to {ckpt_path}")
                torch.save(checkpoint_dict, ckpt_path)
                if args.delete_previous_checkpoint:
                    if epoch > 0:
                        os.remove(ckpt_path)


def train_one_epoch_calvin_cotrain(
    args,
    model,
    epoch,
    calvin_loader,
    coco_loader,
    vqa_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    # setup loaders
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    t = tqdm(
        enumerate(zip(coco_loader, vqa_loader, calvin_loader)),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")

    mv_avg_loss = []
    mv_avg_loss_coco = []
    mv_avg_loss_vqa = []
    for num_steps, (batch_coco, batch_vqa, batch_calvin) in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        #### COCO FORWARD PASS ####
        images = batch_coco[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
        input_ids = batch_coco[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_coco[1][1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        def calculate_vl_cross_entropy(logits, labels, mask=None):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if mask is None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(
                        -1, logits.shape[-1]
                    ),
                    shift_labels.view(-1),
                )
            else:
                # TODO: mask is with the same shape of labels, 
                # 1 represents valid, 0 for non-valid, only calculate loss for valid tokens
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                shift_logits.view(
                        -1, logits.shape[-1]
                    ),
                shift_labels.view(-1),
                )
                # mask the loss
                mask = mask[..., 1:].contiguous()
                loss = loss * mask.reshape(-1)
                # mean
                loss = loss.mean()
            return loss

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                mode = 'vision_lang'
            )
        
        logits = output.logits
        loss_coco = calculate_vl_cross_entropy(logits, labels)
        mv_avg_loss_coco.append(loss_coco.item())
        divided_loss_coco = loss_coco * args.vl_task_weights
        divided_loss_coco = divided_loss_coco / args.gradient_accumulation_steps
        
        (divided_loss_coco * args.loss_multiplier_calvin).backward()

        #### VQA FORWARD PASS ####
        images = batch_vqa[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images = rearrange(images, "(b t f) c h w -> b t f c h w", t=1, f=1)
        input_ids = batch_vqa[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_vqa[1][1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )
        ques_mask = batch_vqa[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids.to(device_id),
                attention_mask=attention_mask.to(device_id),
                # labels=labels,
                mode = 'vision_lang'
            )
        
        logits = output.logits
        loss_vqa = calculate_vl_cross_entropy(logits, labels, ques_mask)
        mv_avg_loss_vqa.append(loss_vqa.item())
        divided_loss_vqa = loss_vqa * 0.5
        divided_loss_vqa = divided_loss_vqa / args.gradient_accumulation_steps
        (divided_loss_vqa * args.loss_multiplier_calvin).backward()
        
        #### CALVIN FORWARD PASS ####
        images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))

        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        if args.fusion_mode != 'vit_concat':
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1)
        else:
            input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)
        # input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)

        # do the same to the attention mask 
        if args.fusion_mode != 'vit_concat':
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1)
        else:
            attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True)
        
        state_tensor = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        robot_obs = batch_calvin[5].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.clip_state:
            state_tensor = torch.cat([state_tensor[..., :6], state_tensor[..., [-1]]], dim=-1)
        labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.tcp_rel:
            if args.multi_step_action == 1:
                labels = world_to_tcp_frame(labels, state_tensor)
            else:
                bs, seq_len = labels.shape[:2]
                labels = world_to_tcp_frame(labels, robot_obs)
                labels = labels.view(bs, seq_len, args.multi_step_action, -1)
        
        state_tensor = state_tensor.unsqueeze(2).unsqueeze(2)

        # merge the batch and the sequence dimension
        images = images.flatten(0, 1)
        gripper = gripper.flatten(0, 1)
        state_tensor = state_tensor.flatten(0, 1)
        if args.fusion_mode != 'vit_concat':
            input_ids = input_ids.flatten(0, 1)
            attention_mask = attention_mask.flatten(0, 1)

        # [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        if args.use_hist:
            labels = labels[:, [-1]]  # only calculate last step action
        if args.fusion_mode == 'vit_concat':
            labels = labels[:, -1]
        labels = [labels[..., :6], (labels[..., 6:] + 1) // 2]

        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=gripper,
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None
            )

        # compute loss
        num_actions, bin_actions = output.logits[0], output.logits[1]

        def discretize_actions(pose_action):
            action_min = -1.001
            action_max = 1.001
            action_len = (action_max - action_min) / args.act_disc
            pose_action = (pose_action - action_min) / action_len
            pose_action = torch.floor(pose_action).long()
            return pose_action
        
        if args.act_disc != -1:
            # assert labels[0].max() < 1.0, f"{labels[0].max()} >= 1.0"
            # assert labels[0].min() > -1.0, f"{labels[0].min()} <= -1.0"
            labels[0] = discretize_actions(labels[0])
            assert labels[0].max() < args.act_disc, f"{labels[0].max()} >= {args.act_disc}"
            assert labels[0].min() >= 0, f"{labels[0].min()} < 0"
        # reshape for loss calculation
        if args.multi_step_action != 1:
            bs, seq_len = num_actions.shape[:2]
            num_actions = num_actions.reshape(bs, seq_len, args.multi_step_action, -1)
            bin_actions = bin_actions.reshape(bs, seq_len, args.multi_step_action, -1)

        loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        if args.act_disc == -1:
            loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
            if args.real_data:
                loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
            else:
                loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01
        else:
            bs, seq_len = num_actions.shape[:2]
            num_actions = num_actions.view(bs, seq_len, -1, args.act_disc).permute(0, 3, 1, 2)
            labels[0] = labels[0].view(bs, seq_len, -1)
            # print('-'*100)
            # print(num_actions, labels[0])
            loss_calvin_num = torch.nn.functional.cross_entropy(num_actions, labels[0])
            if args.real_data:
                loss_calvin = loss_calvin_num + loss_calvin_bin * 0.2
            else:
                loss_calvin = loss_calvin_num + loss_calvin_bin * 0.1
        

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_calvin * args.loss_multiplier_calvin
        )
        mv_avg_loss.append(loss.item())
        loss.backward()
        
        # clip gradient norm
        if args.fsdp:
            """
            The way we clip gradients with FSDP is different than the non-FSDP case,
            because during FSDP, gradient norms are computed over certain submodules,
            rather than the entire model.
            At least for OPT-125M, this didn't seem to make a difference in performance.
            """
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            # rank 0 logging
            if args.rank == 0 and args.report_to_wandb:
                coco_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_vl
                    * args.world_size
                    / step_time_m.val
                )
                coco_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_vl
                    / step_time_m.val
                )
                vqa_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_vl
                    * args.world_size
                    / step_time_m.val
                )
                vqa_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_vl
                    / step_time_m.val
                )
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "coco_samples_per_second": coco_samples_per_second,
                        "coco_samples_per_second_per_gpu": coco_samples_per_second_per_gpu,
                        "vqa_samples_per_second": vqa_samples_per_second,
                        "vqa_samples_per_second_per_gpu": vqa_samples_per_second_per_gpu,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_coco": loss_coco.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )
                wandb.log(
                    {"loss_vqa": loss_vqa.item(), "global_step": global_step},
                    commit=False,
                )

                wandb.log(
                    {
                        "loss_calvin": divided_loss_calvin.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete.  Loss coco: {loss_coco.item():.3f} // Loss vqa: {loss_vqa.item():.3f} // Loss CALVIN: (all){loss_calvin.item():.3f} (mse){loss_calvin_num.item():.3f} (bce){loss_calvin_bin.item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg calvin loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "avg coco loss": sum(mv_avg_loss_coco[-avg_horizon:]) / avg_horizon, "avg vqa loss": sum(mv_avg_loss_vqa[-avg_horizon:]) / avg_horizon,
                        "loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item()})


def train_one_epoch_calvin_two_way(
    args,
    model,
    epoch,
    calvin_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    
    num_batches_per_epoch_calvin = calvin_loader.num_batches

    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch
        
        # put images and labels on device
        images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        # images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(1).unsqueeze(1))
        vision_x = torch.cat([images, gripper], dim=0)
        # input_ids is LongTensor and does not require conversion precision
        # repeat the input_ids to match the sequence length of the images
        input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(2, images.shape[1], 1)

        # input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)

        # do the same to the attention mask 
        attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(2, images.shape[1], 1)
        state_tensor = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True).repeat(2, 1, 1).unsqueeze(2).unsqueeze(2)
        # import pdb; pdb.set_trace()
        # merge the batch and the sequence dimension
        # images = images.flatten(0, 1)
        # gripper = gripper.flatten(0, 1)
        images = images.detach().cpu()
        gripper = gripper.detach().cpu()
        vision_x = vision_x.flatten(0, 1)
        state_tensor = state_tensor.flatten(0, 1)
        input_ids = input_ids.flatten(0, 1)
        attention_mask = attention_mask.flatten(0, 1)

        # attention_mask = batch_calvin[1][1].to(device_id, dtype=cast_dtype, non_blocking=True)
        # attention_mask = None

        labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        # [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        if args.use_hist:
            labels = labels[:, [-1]]  # only calculate last step action
        labels = [labels[..., :6], (labels[..., 6:] + 1) // 2]
        # labels = [labels[..., :6], labels[..., 6:]]

        with autocast():
            output = model(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,  # loss计算放在外面
                vision_gripper=None,
                state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None
            )

        # compute loss
        num_actions, bin_actions = output.logits
        loss_calvin_num = torch.nn.functional.huber_loss(num_actions, labels[0])
        loss_calvin_bin = torch.nn.functional.binary_cross_entropy(bin_actions, labels[1])
        # loss_calvin = loss_calvin_num + loss_calvin_bin * 0.05
        loss_calvin = loss_calvin_num + loss_calvin_bin * 0.01

        divided_loss_calvin = loss_calvin / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_calvin * args.loss_multiplier_calvin
        )
        mv_avg_loss.append(loss.item())
        loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_calvin
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_calvin": divided_loss_calvin.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )


        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: (all){loss_calvin.item():.3f} (mse){loss_calvin_num.item():.3f} (bce){loss_calvin_bin.item():.3f}"
            )
        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "Lnum": loss_calvin_num.item(), "Lbin": loss_calvin_bin.item()})


def train_one_epoch(
    args,
    model,
    epoch,
    laion_loader,
    mmc4_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    num_batches_per_epoch_laion = laion_loader.num_batches
    num_batches_per_epoch_mmc4 = mmc4_loader.num_batches

    assert (
        num_batches_per_epoch_laion == num_batches_per_epoch_mmc4
    ), "Number of batches in laion and mmc4 datasets must be the same"
    num_batches_per_epoch = num_batches_per_epoch_mmc4
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    for num_steps, (batch_laion, batch_mmc4) in tqdm(
        enumerate(zip(laion_loader, mmc4_loader)),
        # disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)

        global_step = num_steps + epoch * num_batches_per_epoch

        #### LAION FORWARD PASS ####
        images = (
            batch_laion[0]
            .to(device_id, dtype=cast_dtype, non_blocking=True)
            .unsqueeze(1)
            .unsqueeze(1)
        )

        input_ids = batch_laion[1][0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch_laion[1][1].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == media_token_id] = -100
        labels.to(device_id)

        with autocast():
            loss_laion = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
        divided_loss_laion = loss_laion / args.gradient_accumulation_steps

        #### C4 FORWARD PASS ####
        images = (
            batch_mmc4[0]
            .to(device_id, dtype=cast_dtype, non_blocking=True)
            .unsqueeze(2)
        )
        input_ids = torch.stack([x[0] for x in batch_mmc4[1]]).squeeze(1)
        attention_mask = torch.stack([x[1] for x in batch_mmc4[1]]).squeeze(1)

        # NOTE: irena: expected shape of clip_text_input_ids / attention_mask is (N, I, max_seq_len)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100

        for i in range(labels.shape[0]):
            # remove loss for any token before the first <image> token
            label_idx = 0
            while (
                label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id
            ):
                labels[i][label_idx] = -100
                label_idx += 1

            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while (
                    token_idx < labels.shape[1]
                    and labels[i][token_idx] != media_token_id
                ):
                    labels[i][token_idx] = -100
                    token_idx += 1

        labels[labels == media_token_id] = -100
        labels.to(device_id)

        with autocast():
            loss_mmc4 = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]

            # if loss is nan, skip this batch
            if torch.isnan(loss_mmc4):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad()
                continue

        divided_loss_mmc4 = loss_mmc4 / args.gradient_accumulation_steps

        #### BACKWARD PASS ####
        loss = (
            divided_loss_laion * args.loss_multiplier_laion
            + divided_loss_mmc4 * args.loss_multiplier_mmc4
        )
        loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                laion_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    * args.world_size
                    / step_time_m.val
                )
                laion_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_laion
                    / step_time_m.val
                )

                c4_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size_mmc4
                    * args.world_size
                    / step_time_m.val
                )
                c4_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size_mmc4
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "laion_samples_per_second": laion_samples_per_second,
                        "laion_samples_per_second_per_gpu": laion_samples_per_second_per_gpu,
                        "c4_samples_per_second": c4_samples_per_second,
                        "c4_samples_per_second_per_gpu": c4_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_laion": divided_loss_laion.item(),
                        "global_step": global_step,
                    },
                    commit=False,
                )
                wandb.log(
                    {"loss_mmc4": divided_loss_mmc4.item(), "global_step": global_step},
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0):
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss LAION: {loss_laion.item():.3f} // Loss MMC4: {loss_mmc4.item():.3f}"
            )


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad and 'normalizer' not in name:
            del state_dict[name]

    return state_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
