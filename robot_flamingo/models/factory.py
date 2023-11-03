from logging import debug
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import open_clip
from typing import Optional
from robot_flamingo.models.flamingo_bc import BCFlamingo
from robot_flamingo.models.flamingo_mpt import MPTFlamingo
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance
from open_flamingo.src.factory import _infer_decoder_layers_attr_name

mpt_dict = {
    "mpt_3b": {
        "lang_encoder_path": "path_to/mpt-1b-redpajama-200b", 
        "tokenizer_path": "path_to/mpt-1b-redpajama-200b", 
        "cross_attn_every_n_layers": 1,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt"
    }, 
    "mpt_dolly_3b": {
        "lang_encoder_path": "path_to/mpt-1b-redpajama-200b-dolly", 
        "tokenizer_path": "path_to/mpt-1b-redpajama-200b-dolly", 
        "cross_attn_every_n_layers": 1,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-3B-vitl-mpt1b-langinstruct/checkpoint.pt"
    },
    "mpt_4b": {
        "lang_encoder_path": "path_to/RedPajama-INCITE-Instruct-3B-v1", 
        "tokenizer_path": "path_to/RedPajama-INCITE-Instruct-3B-v1", 
        "cross_attn_every_n_layers": 2,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-4B-vitl-rpj3b-langinstruct/checkpoint.pt"
    },
    "mpt_base_4b": {
        "lang_encoder_path": "path_to/RedPajama-INCITE-Base-3B-v1", 
        "tokenizer_path": "path_to/RedPajama-INCITE-Base-3B-v1", 
        "cross_attn_every_n_layers": 2,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-4B-vitl-rpj3b/checkpoint.pt"
    },
    "mpt_9b": {
        "lang_encoder_path": "path_to/mpt-7b", 
        "tokenizer_path": "path_to/mpt-7b", 
        "cross_attn_every_n_layers": 4,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt"
    },
    "llama_9b": {
        "lang_encoder_path": "path_to/llama-7b-hf-jxu124", 
        "tokenizer_path": "path_to/llama-7b-hf-jxu124", 
        "cross_attn_every_n_layers": 4,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-9B/checkpoint.pt"
    }
}



def get_transforms(
    clip_vision_encoder_path: str = "ViT-L-14",
    clip_vision_encoder_pretrained: str = "openai",
    tokenizer_path: str = "path_to/llama-7b-hf-jxu124",
    use_local_files: bool = False,
):
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )

    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    return image_processor, text_tokenizer


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    # this is the window size sampled from the episode
    window_size: int = 32,
    freeze_embed: bool = False,
    train_params = -1,
    use_gripper=False,
    use_state=False,
    last_action=False,
    fusion_mode='',
    pad_length=-1,
    debug=False,
    sep_resampler=False,
    sep_lm_head=False,
    unfreeze_vit=False,
    return_feature=False,
    multi_step_action=1,
    llm_name='llama_9b',
    pooling='max',
    residual=False,
    tcp_rel=False,
    replan=-1,
    decoder_type='lstm',
    hidden_size=None,
    freeze_sampler=False,
    fwd_pred=False, 
    fwd_pred_hand=False,
    no_image_patch=False,
    global_latent=1,
    refresh=-1,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=use_local_files
    )
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    if debug:
        # Load the local checkpoint into a model instance.
        lang_encoder = AutoModelForCausalLM.from_pretrained(lang_encoder_path, ignore_keys=["config"], trust_remote_code=True)
        # Set the `init_weights` parameter to `False` to prevent the model from loading the pretrained weights.
        lang_encoder.init_weights(False)
    else:
        print(lang_encoder_path)
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path, local_files_only=use_local_files, trust_remote_code=True
        )
        # print(lang_encoder_path)
        # if llm_name == 'llama':
        #     lang_encoder = AutoModelForCausalLM.from_pretrained(
        #     lang_encoder_path, local_files_only=use_local_files
        # )
        # else:
        #     # name = 'mosaicml/mpt-7b'
        #     config = {
        #         "model_type": "auto",
        #         "add_lm_head": True,
        #     }
        #     lang_encoder = AutoModelForCausalLM.from_pretrained(
        #         lang_encoder_path, local_files_only=use_local_files
        #     )
    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_encoder_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings
        extend_instance(lang_encoder, EmbeddingFnMixin)
    
    extend_instance(lang_encoder, FlamingoLMMixin)
    
    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    # print(lang_encoder.base_model_prefix)
    # print(getattr(lang_encoder, lang_encoder.base_model_prefix, lang_encoder))
    # print(lang_encoder)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))
    
    if 'llama' in llm_name:
        Model_fn = BCFlamingo
    elif 'mpt' in llm_name:
        Model_fn = MPTFlamingo
    else:
        raise NotImplementedError
    
    model = Model_fn(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
            "width"
        ],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        window_size=window_size,
        use_gripper=use_gripper,
        use_state=use_state,
        fusion_mode=fusion_mode,
        last_action=last_action,
        pad_length=pad_length,
        sep_resampler=sep_resampler,
        sep_lm_head=sep_lm_head,
        return_feature=return_feature,
        multi_step_action=multi_step_action,
        llm=llm_name,
        pooling=pooling,
        residual=residual,
        tcp_rel=tcp_rel,
        replan=replan,
        decoder_type=decoder_type,
        hidden_size=hidden_size,
        refresh=refresh,
        fwd_pred=fwd_pred,
        fwd_pred_hand=fwd_pred_hand,
        no_image_patch=no_image_patch,
        global_latent=global_latent,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    # model.perceiver.requires_grad_(True)
    if train_params == -1:
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        model.perceiver.requires_grad_(True)
    else:
        param_per_layer = 140
        layer_num = int(train_params / param_per_layer + 0.5)
        cnt = 0
        for ix in range(len(model.lang_encoder.gated_cross_attn_layers)-1, -1, -1):
            if cnt >= layer_num:
                break
            if model.lang_encoder.gated_cross_attn_layers[ix] is not None:
                model.lang_encoder.gated_cross_attn_layers[ix].requires_grad_(True)
                cnt += 1
    if freeze_sampler:
        model.perceiver.requires_grad_(False)
    if not freeze_embed:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
    model.lang_encoder.lm_head.requires_grad_(True)

    if model.sep_lm_head:
        model.lm_head.requires_grad_(True)
    if model.use_diff:
        model.diffusion_model.requires_grad_(True)
    if unfreeze_vit:
        model.vision_encoder.requires_grad_(True)
    # # Unfreeze the action head 
    # model.action_head.requires_grad_(True)

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer
