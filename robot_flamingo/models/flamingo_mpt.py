import torch
from einops import rearrange, repeat
from torch import nn
import copy
from open_flamingo.src.helpers import PerceiverResampler
from robot_flamingo.models.action_head import DeterministicDecoder, DiffusionDecoder, FCDecoder, GPTDecoder
from collections import namedtuple


class MPTFlamingo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        use_media_placement_augmentation: bool = False,
        # this is the window size sampled from the episode
        window_size: int = 8,
        use_gripper=False,
        fusion_mode='',
        sep_resampler=False,
        use_state=False,
        use_diff=False,
        diff_horizon=32,
        last_action=False,
        n_timesteps=150,
        state_dim=15,
        use_hist=False,
        debug=False,
        predict_epsilon=True,
        pad_length=-1,
        multi_step_action=1,
        sep_lm_head=False,
        return_feature = False,
        llm='llama',
        pooling='max',
        residual=False,
        tcp_rel=False,
        replan=-1,
        decoder_type='lstm',
        hidden_size=None,
        fwd_pred=False,
        fwd_pred_hand=False,
        global_latent=10,
        no_image_patch=False,
        refresh=-1
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
            use_media_placement_augmentation (bool, optional): Whether to randomly assign images to the preceding or following text in training. Defaults to False.
        """
        super().__init__()

        self.use_gripper = use_gripper
        self.use_state = use_state
        self.fusion_mode = fusion_mode
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = use_media_placement_augmentation
        self.vis_dim = vis_dim
        self.window_size = window_size
        self.tcp_rel = tcp_rel
        self.act_step = multi_step_action
        print('window size: {}'.format(window_size))
        self.vision_encoder = vision_encoder
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.sep_resampler = sep_resampler
        self.use_hist = use_hist
        self.lang_encoder = lang_encoder
        self.pad_length = pad_length
        self.replan = replan
        if self.replan != -1:
            self.replan = min(int(replan * self.window_size), 180)
        self.refresh = refresh
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size

        self.residual = residual
        print(self.vis_dim, self.lang_dim)
        print(lang_encoder.config)
        if not debug:
            if 'llama' in llm:
                self.lang_encoder.init_flamingo(
                    media_token_id=media_token_id,
                    vis_hidden_size=self.vis_dim,
                    cross_attn_every_n_layers=cross_attn_every_n_layers,
                    use_media_placement_augmentation=self.use_media_placement_augmentation,
                    residual=residual,
                )
            else:
                self.lang_encoder.init_flamingo(
                    media_token_id=media_token_id,
                    lang_hidden_size=self.lang_dim,
                    vis_hidden_size=self.vis_dim,
                    cross_attn_every_n_layers=cross_attn_every_n_layers,
                    gradient_checkpointing=False,
                )

        if sep_resampler:
            self.perceiver_gripper = PerceiverResampler(dim=self.vis_dim)
            self.perceiver_gripper.load_state_dict(copy.deepcopy(self.perceiver.state_dict()))
        if use_state:
            self.state_fc = nn.Linear(state_dim, self.vis_dim)
        if use_hist:
            self.frame_embs = nn.Parameter(torch.randn(self.window_size, self.vis_dim))
        # To-do: nn archiecture for actor
        self.llm = llm
        if llm=='llama':
            in_features = lang_encoder.lm_head.in_features
        else:
            in_features = self.lang_dim
        self.use_diff = use_diff
        self.decoder_type = decoder_type
        if decoder_type == 'lstm':
            lm_head = DeterministicDecoder(in_features, self.window_size, 
            use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action, pooling=pooling)
            self.lang_encoder.lm_head = lm_head
        elif decoder_type == 'fc':
            if use_hist:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
            elif 'vit_concat' in fusion_mode:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
            else:
                raise NotImplementedError
        elif decoder_type == 'diffusion':
            if use_diff:
                self.diffusion_model = DiffusionDecoder(
                    self.action_head.hidden_size, 
                    self.window_size,
                    input_dim=self.action_head.out_features+1,
                    n_timesteps=n_timesteps,
                    horizon=diff_horizon,
                    predict_epsilon=predict_epsilon,
                )
            else:
                raise NotImplementedError
        elif decoder_type=='gpt':
            lm_head = GPTDecoder(in_features, self.window_size, use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, multi_step_action=multi_step_action, pooling=pooling, hidden_size=hidden_size)
            self.lang_encoder.lm_head = self.action_head = lm_head
        else:
            raise NotImplementedError
        
        sep_lm_head = True
        self.sep_lm_head = sep_lm_head
        if sep_lm_head:
            self.lm_head = self.lang_encoder.lm_head
            self.lang_encoder.lm_head = nn.Identity()

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper = None,
        state_tensor = None,
        return_feature = False,
        policy_mask=None
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        raw_rgb = vision_x.clone()
        raw_gripper = vision_gripper.clone()
        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            if self.use_hist:
                self._encode_history_vision_post_fusion(vision_x, vision_gripper)
            else:
                if not self.use_gripper or self.fusion_mode == 'two_way':
                    vision_x = self._encode_vision_x(vision_x=vision_x)
                else:
                    if self.fusion_mode == 'pre':
                        self._encode_multi_vision_pre_fusion(vision_x, vision_gripper)
                    elif self.fusion_mode == 'post':
                        self._encode_multi_vision_post_fusion(vision_x, vision_gripper)
                    elif self.fusion_mode == 'vit_concat':
                        self._encode_history_vision_fc_post(vision_x, vision_gripper)
        
        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask.bool(),
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True
        )

        output_hs = output.hidden_states[-1]
        output_hs = self.lm_head(output_hs, state_tensor=state_tensor, return_feature=return_feature)
        output.logits = output_hs
        
        return output

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_vision(self, vision_x: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        return vision_x

    def _encode_multi_vision_pre_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)
        vision_x = torch.cat([vision_rgb, vision_gripper], dim=3)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_multi_vision_post_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)
        vision_rgb = self.perceiver(vision_rgb)
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2)  # reshapes to (b, T, 2*n, d)
        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=2)  # reshapes to (b, T, 2*n+1, d)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_multi_vision_two_way(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)
        vision_rgb = self.perceiver(vision_rgb)
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=0)  # reshapes to (b, T, 2*n, d)
        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=0)  # reshapes to (b, T, 2*n+1, d)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_history_vision_post_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        vision_rgb = self._encode_vision(vision_rgb)
        vision_gripper = self._encode_vision(vision_gripper)
        bs = int(vision_rgb.shape[0] // self.window_size)
        vision_rgb = vision_rgb.view(bs, self.window_size, *vision_rgb.shape[1:])
        _, _, T, p, v_tok, dim = vision_rgb.shape[:6]
        frame_embs = repeat(self.frame_embs, 'F d -> b F T p v d', b=bs, T=T, p=p, v=v_tok)
        vision_rgb = vision_rgb + frame_embs
        vision_rgb = rearrange(vision_rgb, 'b F T p v d -> (b F) T p v d')
        vision_rgb = self.perceiver(vision_rgb)

        vision_gripper = vision_gripper.view(vision_gripper.shape[0] // self.window_size, self.window_size,
                                             *vision_gripper.shape[1:])
        frame_embs = repeat(self.frame_embs, 'F d -> b F T p v d', b=bs, T=T, p=p, v=v_tok)
        vision_gripper = vision_gripper + frame_embs
        vision_gripper = rearrange(vision_gripper, 'b F T p v d -> (b F) T p v d')
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2)  # reshapes to (b, T, 2*n, d)
        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=2)  # reshapes to (b, T, 2*n+1, d)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x
    
    def _encode_history_vision_fc_post(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_rgb (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            vision_gripper (torch.Tensor): Vision rgb input
                shape (B, T_img, F, C, H, W)
            Images in the same chunk are collated along T_img, and frames are collated along F
            Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        bs = int(vision_rgb.shape[0] // self.window_size)
        vision_rgb = self._encode_vision(vision_rgb)
        vision_rgb = self.perceiver(vision_rgb) # BxL, T, n, d
        vision_rgb = vision_rgb.view(-1, self.window_size, *vision_rgb.shape[1:]) # B, L, T, n, d
        vision_rgb = rearrange(vision_rgb, 'b L T n d -> b T (n L) d')

        vision_gripper = self._encode_vision(vision_gripper)
        if self.sep_resampler:
            vision_gripper = self.perceiver_gripper(vision_gripper)
        else:
            vision_gripper = self.perceiver(vision_gripper)
        vision_gripper = vision_gripper.view(-1, self.window_size, *vision_gripper.shape[1:]) # B, L, T, n, d
        vision_gripper = rearrange(vision_gripper, 'b L T n d -> b T (n L) d')

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2)

        if self.use_state and state_tensor is not None:
            state_tensor = self.state_fc(state_tensor)
            vision_x = torch.cat([vision_x, state_tensor], dim=2)  # reshapes to (b, T, 2*n+1, d)
        
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x