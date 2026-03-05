# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

# Attention utilities for missing object handling
from lvqa_dino.attention_utils import (
    AttentionStore,
    register_attention_control,
    restore_attention_processors,
    get_token_indices,
    compute_attention_loss
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

# L-DINO-CoT: Localized VQA Scoring imports

from lvqa_dino import LDINOOptimizer, EntityInfo, create_entities_from_simple_format
from lvqa_dino.prompt_decomposer import StructuredCoTDecomposer
LVQA_AVAILABLE = True

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained(
        ...     "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

class DependencyGraphEvaluator:
    """
    Evaluates a dependency graph of VQA prompts using:
    1. Zero-Out Logic: If a parent node scores < 0.5, child nodes are mathematically forced to 0.0.
    2. Multiplicative Scoring: Final score of a root tree is the product of all valid node scores in that tree.
    """
    def __init__(self, dependency_graph):
        self.nodes = {n["id"]: n for n in dependency_graph.get("nodes", [])}
        self.threshold = 0.5
        self.questions = {n["id"]: n["question"] for n in self.nodes.values()}

    def evaluate(self, vqa_scores_dict):
        """
        Args:
            vqa_scores_dict: A dictionary mapping node_id -> raw VQA score (float).
        Returns:
            masked_scores: dict mapping node_id -> masked score.
            root_scores: dict mapping root_id -> multiplied score for that entire tree.
            avg_root_score: average score across all root trees.
        """
        masked_scores = {}
        
        # 1. Zero-Out Masking (Topological traverse)
        # Helper to compute the mask value recursively
        def get_mask(node_id):
            node = self.nodes.get(node_id)
            if not node: return 1.0
            
            parents = node.get("parent_id")
            if not parents:
                return 1.0  # Root node
            
            if isinstance(parents, str):
                parents = [parents]
            
            # For relations with multiple parents, all parents must be > threshold
            mask = 1.0
            for p_id in parents:
                if p_id in masked_scores:
                    parent_masked_score = masked_scores[p_id]
                else:
                    # Resolve parent mask first
                    parent_mask = get_mask(p_id)
                    parent_raw = vqa_scores_dict.get(p_id, 0.0)
                    parent_masked_score = parent_mask * parent_raw
                    masked_scores[p_id] = parent_masked_score
                
                if parent_masked_score < self.threshold:
                    mask = 0.0
                    break
            return mask

        # Apply masks strictly
        for node_id in self.nodes:
            if node_id not in masked_scores:
                mask = get_mask(node_id)
                raw = vqa_scores_dict.get(node_id, 0.0)
                masked_scores[node_id] = mask * raw
                
        # 2. Multiplicative Scoring per independent root
        # Identify roots
        roots = [n_id for n_id, n in self.nodes.items() if not n.get("parent_id")]
        
        root_scores = {}
        for root_id in roots:
            # Gather all descendants of this root (plus the root itself)
            # A node belongs to a tree if it is the root or can reach the root via parent chain
            tree_nodes = set()
            def dfs(nid):
                if nid in tree_nodes: return
                tree_nodes.add(nid)
                # Find children
                for child_id, child in self.nodes.items():
                    parents = child.get("parent_id")
                    if isinstance(parents, list):
                        if nid in parents: dfs(child_id)
                    elif parents == nid:
                        dfs(child_id)
            
            dfs(root_id)
            
            # Product of all nodes in this isolated tree
            tree_score = 1.0
            for nid in tree_nodes:
                tree_score *= masked_scores.get(nid, 1e-9) # Avoid exact zero to keep gradients flowing if possible, though masking kills it usually
            
            root_scores[root_id] = tree_score
            
        # 3. Final average across roots
        avg_score = sum(root_scores.values()) / len(root_scores) if root_scores else 0.0
        
        return masked_scores, root_scores, avg_score


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class DependencyGraphEvaluator:
    """
    Evaluates a dependency graph of VQA prompts using:
    1. Zero-Out Logic: If a parent node scores < 0.5, child nodes are mathematically forced to 0.0.
    2. Multiplicative Scoring: Final score of a root tree is the product of all valid node scores in that tree.
    """
    def __init__(self, dependency_graph):
        self.nodes = {n["id"]: n for n in dependency_graph.get("nodes", [])}
        self.threshold = 0.5
        self.questions = {n["id"]: n["question"] for n in self.nodes.values()}

    def evaluate(self, vqa_scores_dict):
        """
        Args:
            vqa_scores_dict: A dictionary mapping node_id -> raw VQA score (float).
        Returns:
            masked_scores: dict mapping node_id -> masked score.
            root_scores: dict mapping root_id -> multiplied score for that entire tree.
            avg_root_score: average score across all root trees.
        """
        masked_scores = {}
        
        # 1. Zero-Out Masking (Topological traverse)
        def get_mask(node_id):
            node = self.nodes.get(node_id)
            if not node: return 1.0
            
            parents = node.get("parent_id")
            if not parents:
                return 1.0  # Root node
            
            if isinstance(parents, str):
                parents = [parents]
            
            # For relations with multiple parents, all parents must be >= threshold
            mask = 1.0
            for p_id in parents:
                if p_id in masked_scores:
                    parent_masked_score = masked_scores[p_id]
                else:
                    parent_mask = get_mask(p_id)
                    parent_raw = vqa_scores_dict.get(p_id, 0.0)
                    parent_masked_score = parent_mask * parent_raw
                    masked_scores[p_id] = parent_masked_score
                
                if parent_masked_score < self.threshold:
                    mask = 0.0
                    break
            return mask

        # Apply masks strictly
        for node_id in self.nodes:
            if node_id not in masked_scores:
                mask = get_mask(node_id)
                raw = vqa_scores_dict.get(node_id, 0.0)
                masked_scores[node_id] = mask * raw
                
        # 2. Multiplicative Scoring per independent root
        roots = [n_id for n_id, n in self.nodes.items() if not n.get("parent_id")]
        
        root_scores = {}
        for root_id in roots:
            tree_nodes = set()
            def dfs(nid):
                if nid in tree_nodes: return
                tree_nodes.add(nid)
                for child_id, child in self.nodes.items():
                    parents = child.get("parent_id")
                    if isinstance(parents, list):
                        if nid in parents: dfs(child_id)
                    elif parents == nid:
                        dfs(child_id)
            
            dfs(root_id)
            
            tree_score = 1.0
            for nid in tree_nodes:
                tree_score *= max(masked_scores.get(nid, 0.0), 1e-9) # Avoid exact zero to keep gradients flowing if possible
            
            root_scores[root_id] = tree_score
            
        # 3. Final average across roots
        avg_score = sum(root_scores.values()) / len(root_scores) if root_scores else 0.0
        
        return masked_scores, root_scores, avg_score


class StableDiffusionNDPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    """
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) for
            more details about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if scheduler is not None and getattr(scheduler.config, "steps_offset", 1) != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if scheduler is not None and getattr(scheduler.config, "clip_sample", False) is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = (
            unet is not None
            and hasattr(unet.config, "_diffusers_version")
            and version.parse(version.parse(unet.config._diffusers_version).base_version) < version.parse("0.9.0.dev0")
        )
        self._is_unet_config_sample_size_int = unet is not None and isinstance(unet.config.sample_size, int)
        is_unet_sample_size_less_64 = (
            unet is not None
            and hasattr(unet.config, "sample_size")
            and self._is_unet_config_sample_size_int
            and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- stable-diffusion-v1-5/stable-diffusion-v1-5"
                " \n- stable-diffusion-v1-5/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            # Generate a neutral noise seed by ensembling M independent vectors
            # M = 5
            # noise_sum = torch.zeros(shape, device=device, dtype=dtype)
            # for _ in range(M):
            #     noise_sum += randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # Average and renormalize to maintain unit variance
            # latents = (noise_sum / M) * math.sqrt(M)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    # def directional_gaussian(self, s: np.ndarray, alpha: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    #     """
    #     Generates a Gaussian vector with a specified direction and isotropic component.
    #     Args:
    #         s (np.ndarray): Direction vector.
    #         alpha (float): Proportion of the vector in the direction of s.
    #         sigma (float): Standard deviation of the isotropic component.
    #     Returns:
    #         np.ndarray: A Gaussian vector with the specified properties.
    #     """
    #     s = np.asarray(s)
    #     dim = len(s)
    #     s_norm = np.linalg.norm(s)
        
    #     # Add explicit zero vector check
    #     if s_norm < 1e-6:
    #         raise ValueError("Direction vector cannot be zero")
        
    #     if alpha < 1e-6:
    #         return np.random.normal(0, sigma, dim)
        
    #     # Rest of the function remains unchanged
    #     dir_magnitude = np.random.normal(0, np.sqrt(alpha))
    #     dir_component = (dir_magnitude / s_norm) * s
    #     iso_component = np.random.normal(0, np.sqrt(1 - alpha), dim)
    #     return sigma * (dir_component + iso_component)

    @staticmethod
    def directional_gaussian_torch(grad, alpha, beta, generator=None):
        """
        Samples a directional Gaussian noise vector in PyTorch with
        covariance Σ = β I + α (g g^T - (1/n) I)
        where g = normalized grad (PyTorch tensor).
        Args:
            grad: 1D torch tensor (CPU or CUDA)
            alpha: scalar, must be >= 0
            beta: scalar, must be > alpha / n
            generator: optional torch.Generator for reproducibility
        Returns:
            v: torch tensor, same device/dtype as grad
        """
        n = grad.numel()
        device = grad.device
        dtype = grad.dtype
        g = grad.view(-1) / (grad.view(-1).norm() + 1e-9)
        lam1 = beta + alpha * (1 - 1/n)
        lam2 = beta - alpha / n

        # Sample standard normals: one for direction, one for orthogonal
        w = torch.randn(n, device=device, dtype=dtype, generator=generator)
        c = torch.dot(g, w)                     # scalar
        w_perp = w - c * g                      # vector, orthogonal to g

        noise = (lam1**0.5) * c * g + (lam2**0.5) * w_perp
        return noise.view(grad.shape)


    def init_vqa_model(self, vqa_model,device):
        self.vqa_model = vqa_model
        self.vqa_model_device = device
        self.ldino_optimizer = None  # Will be initialized if entity_attributes provided


    #@torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        decomposed_prompts: List[str] = None,
        dependency_graph: Optional[Dict] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        optimization_epoch=50,
        use_localized_vqa: bool = True,  # L-DINO-CoT: enable localized scoring
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        def denoise(latents, prompt_embeds=prompt_embeds, max_steps=None, return_latents=False, return_tweedie=False, start_step=0):
            max_steps = max_steps if max_steps is not None else num_inference_steps

            with self.progress_bar(total=(max_steps - start_step)) as progress_bar:
                noise_list = {}
                for i, t in enumerate(timesteps):
                    if i < start_step:
                        continue
                    if i >= max_steps:
                        break
                    if self.interrupt:
                        continue

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    noise_list[t] = noise_pred.detach()
                    # compute the previous noisy sample x_t -> x_t-1
                    current_latents = latents
                    step_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                    latents = step_output.prev_sample

                    if return_tweedie and (i == max_steps - 1 or i == len(timesteps) - 1):
                        if getattr(step_output, "pred_original_sample", None) is not None:
                            tweedie_est = step_output.pred_original_sample
                        else:
                            alpha_prod_t = self.scheduler.alphas_cumprod[t.item() if torch.is_tensor(t) else t].to(latents.device)
                            beta_prod_t = 1 - alpha_prod_t
                            tweedie_est = (current_latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == max_steps - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

                    if XLA_AVAILABLE:
                        xm.mark_step()

            if return_tweedie and return_latents:
                return noise_list, tweedie_est, latents
            if return_tweedie:
                return noise_list, tweedie_est
            if return_latents:
                return noise_list, latents  # also return the final computed latent
            return noise_list

        def reverse(latents, noise_list):
            for t in noise_list:
                noise_pred = noise_list[t]
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            return latents

        latents_init = latents.clone().detach()
        with torch.no_grad():
            noise_list = denoise(latents_init)
            self.scheduler.set_timesteps(num_inference_steps)

        latents_init.requires_grad_(True)
        latents = reverse(latents_init, noise_list)
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        dir = f"sdxltest/{prompt[:25].replace(' ', '_')}/{generator.initial_seed()}"
        import os, copy, gc
        save_debug = False
        os.makedirs(dir, exist_ok=True)

        image = image.clone()  # Keep on cuda:0 — .to(cuda:1) zeroes data in multi-GPU setup
        if dependency_graph is None:
            # Fallback for backwards compatibility
            if decomposed_prompts is None:
                decomposed_prompts = [prompt] if isinstance(prompt, str) else prompt
            print("[Warning] No dependency_graph provided, falling back to flat list evaluation.")
            dependency_graph = {
                "nodes": [
                    {"id": f"q{i}", "type": "Entity", "concept": p, "question": p, "parent_id": None}
                    for i, p in enumerate(decomposed_prompts)
                ]
            }
            
        # Initialize graph evaluator
        graph_evaluator = DependencyGraphEvaluator(dependency_graph)
        print(f"[Graph] Evaluator initialized with {len(graph_evaluator.nodes)} logical nodes.")
        
        # L-DINO-CoT: Initialize optimizer early for segmentation-aware gradients
        entities = None
        use_ldino = use_localized_vqa and LVQA_AVAILABLE
        
        if use_ldino:
            # Dynamically extract tracking identities directly from the dependency graph
            auto_entity_attributes = {
                n["concept"]: [] for n in graph_evaluator.nodes.values() if n.get("type") == "Entity"
            }
            
            # Initialize optimizer if not done
            if not hasattr(self, 'ldino_optimizer') or self.ldino_optimizer is None:
                from lvqa_dino import LDINOOptimizer
                # Import differentiable blur
                from lvqa_dino.differentiable_blur import apply_blur_mask
                
                self.ldino_optimizer = LDINOOptimizer(
                    vqa_model=self.vqa_model,
                    device=self.vqa_model_device,
                    warmup_ratio=0.2,
                    lambda_ref=1.0,
                    save_visualizations=True
                )
                print("[L-DINO-CoT] Optimizer auto-initialized for segmentation!")
            
            if len(auto_entity_attributes) > 0:
                entities = self.ldino_optimizer.setup_entities(
                    prompt, auto_entity_attributes, output_dir=dir
                )
                self.ldino_optimizer.print_entity_summary(entities)
            else:
                use_ldino = False
        
        # Aggressive memory cleanup before VQA scoring
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        num_tasks = len(graph_evaluator.nodes)
        print(f"[Gradient] Processing {num_tasks} nodes in dependency graph")
        
        # ============== PER-PROMPT GRADIENT COMPUTATION ==============
        # For each decomposed prompt:
        #   1. Segment entity (if L-DINO active)
        #   2. Blur background
        #   3. Compute VQA score on blurred image
        #   4. Backpropagate to get gradient
        #   5. Store gradient on CPU
        
        gradients_cpu = []  # Store gradients on CPU
        individual_vqa_scores_cpu = []  # Store scalar scores
        
        # Pre-compute masks for initial image if L-DINO active
        initial_masks = {}
        if use_ldino and self.ldino_optimizer and hasattr(self.ldino_optimizer, 'segmenter'):
            with torch.no_grad():
                image_pil = self.image_processor.postprocess(image.detach().cpu(), output_type="pil")[0]
                all_entity_names = [e.name for e in entities]
                print(f"[L-DINO] Pre-computing initial masks for: {all_entity_names}")
                initial_masks = self.ldino_optimizer.segmenter.segment_multiple(image_pil, all_entity_names)
                
                mask_dir = f"{dir}/ldino_debug/initial"
                os.makedirs(mask_dir, exist_ok=True)
                for e_name, m_np in initial_masks.items():
                    if m_np is not None:
                        m_pil = Image.fromarray((m_np * 255).astype(np.uint8))
                        m_pil.save(f"{mask_dir}/mask_obj_{e_name}.png")
                
                if hasattr(self.ldino_optimizer.segmenter, 'visualize_masks') and initial_masks:
                    combined_np = self.ldino_optimizer.segmenter.visualize_masks(image_pil, initial_masks)
                    Image.fromarray(combined_np).save(f"{mask_dir}/combined_masks.png")

        from lvqa_dino.differentiable_blur import apply_blur_mask
        
        for n_id, p in graph_evaluator.questions.items():
            # Zero existing gradients
            if latents_init.grad is not None:
                latents_init.grad.zero_()
            
            # Forward pass with gradient tracking
            latents = reverse(latents_init, noise_list)
            self.scheduler.set_timesteps(num_inference_steps)
            latents_decoded = latents / self.vae.config.scaling_factor
            image_for_vqa = self.vae.decode(latents_decoded, return_dict=False)[0]
            
            # Correct range from [-1, 1] to [0, 1] for CLIP vision encoder
            image_for_vqa_scaled = (image_for_vqa / 2 + 0.5).clamp(0, 1)
            
            # Transfer the scaled and differentiable image to the VQA device
            image_for_vqa_cuda1 = image_for_vqa_scaled.detach().to(self.vqa_model_device).requires_grad_(True)
            target_image = image_for_vqa_cuda1
            matched_entity = None
            mask_np = None
            
            if use_ldino:
                concept = graph_evaluator.nodes[n_id].get("concept", p)
                for ent in entities:
                    if ent.name in concept:
                        matched_entity = ent.name
                        break
                
                if matched_entity and matched_entity in initial_masks:
                    mask_np = initial_masks[matched_entity]
                    if mask_np is not None:
                        mask_tensor = torch.from_numpy(mask_np).to(self.vqa_model_device).float()
                        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0) 
                        target_image = apply_blur_mask(image_for_vqa_cuda1, mask_tensor, blur_radius=21)

            # Compute VQA score on the (potentially blurred) image
            with torch.cuda.device(self.vqa_model_device):
                score = self.vqa_model(target_image, [p])
                score_val = score.item()
                individual_vqa_scores_cpu.append((n_id, score_val))
                print(f"node: {n_id}, question: {p}, score: {score_val}")
                
                # Backpropagate to get gradient for this prompt node
            # Directly backpropagate from the score since the computation graph spans devices
            score.backward()
            
            # Manually inject float32 gradients to SD device avoiding autocast bugs
            grad_on_cuda0 = image_for_vqa_cuda1.grad.to(device=image_for_vqa.device, dtype=torch.float32)
            grad_on_cuda0 = torch.nan_to_num(grad_on_cuda0, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Pass gradient back through the scaling operation manually
            grad_on_cuda0_pre_scale = grad_on_cuda0 / 2.0
            image_for_vqa.backward(grad_on_cuda0_pre_scale)

            print("latents_init.grad NaN:", latents_init.grad.isnan().any().item() if latents_init.grad is not None else "None")

            # Store gradient on CPU immediately, masked to entity region
            if latents_init.grad is not None:
                grad_cpu = latents_init.grad.detach().cpu().clone()
                # Defense against bfloat16/cross-device NaN gradients
                grad_cpu = torch.nan_to_num(grad_cpu, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Mask gradient to entity region if L-DINO is active
                if use_ldino and matched_entity and matched_entity in initial_masks:
                    mask_np_loc = initial_masks[matched_entity]
                    if mask_np_loc is not None:
                        # Downscale mask from image space (H, W) to latent space (H/8, W/8)
                        latent_h, latent_w = grad_cpu.shape[2], grad_cpu.shape[3]
                        mask_pil = Image.fromarray((mask_np_loc * 255).astype(np.uint8))
                        mask_latent = mask_pil.resize((latent_w, latent_h), Image.BILINEAR)
                        mask_latent = np.array(mask_latent) / 255.0
                        mask_tensor_latent = torch.from_numpy(mask_latent).float().unsqueeze(0).unsqueeze(0)
                        # Apply mask to gradient (zero out regions outside entity)
                        grad_cpu = grad_cpu * mask_tensor_latent
                        print(f"  [Grad Mask] Clipped gradient to '{matched_entity}' region")
                
                gradients_cpu.append((n_id, grad_cpu))
            else:
                grad_cpu = torch.zeros_like(latents_init.detach().cpu())
                gradients_cpu.append((n_id, grad_cpu))
            
            del score
            
            # Clear GPU memory
            latents_init.grad = None
            del image_for_vqa, latents_decoded, target_image
            gc.collect()
            torch.cuda.empty_cache()
        
        # Evaluate hierarchical VQA scores
        raw_scores_map = {n_id: s for n_id, s in individual_vqa_scores_cpu}
        masked_scores, root_scores, avg_vqa_score = graph_evaluator.evaluate(raw_scores_map)
        
        # We construct a gradient corresponding to the objective L_i = S_i * M_i for each node
        node_gradients = []
        for n_id, g in gradients_cpu:
            # Use raw scores (unmasked) for coefficients to avoid zero gradients
            raw_val = max(raw_scores_map.get(n_id, 1e-9), 1e-9)
            coeff = 1.0 # Base coefficient is 1.0 if not using the tree logic
            node_grad = coeff * g
            node_gradients.append(node_grad)

        if len(node_gradients) > 1:
            avg_gradient = torch.mean(torch.stack(node_gradients), dim=0)
        else:
            avg_gradient = node_gradients[0]
        
        print(f"[Gradient] max: {torch.max(avg_gradient)}")
        print(f"[Gradient] min: {torch.min(avg_gradient)}")
        print(f"[Gradient] norm: {torch.norm(avg_gradient)}")
        
        # Move averaged gradient back to GPU and set on latents_init
        if avg_gradient.isnan().any():
            print("NaN detected in avg_gradient before moving!")
        latents_init.grad = avg_gradient.to(latents_init.device)
        if latents_init.grad.isnan().any():
            print("NaN detected in latents_init.grad after moving!")
        del gradients_cpu, avg_gradient, node_gradients
        gc.collect()
        torch.cuda.empty_cache()
        
        # Compute final VQA score for logging (no gradient needed)
        # avg_vqa_score computed above via evaluate()
        
        # Regenerate image for final display/saving (clean image)
        with torch.no_grad():
            latents_final = reverse(latents_init.detach(), noise_list)
            self.scheduler.set_timesteps(num_inference_steps)
            latents_final = latents_final / self.vae.config.scaling_factor
            image = self.vae.decode(latents_final, return_dict=False)[0]
            # CRITICAL: Clone immediately and keep on cuda:0
            # .to(cuda:1) silently zeroes tensor data in this multi-GPU setup
            image = image.clone()
        
        # Create dummy vqa_score tensor for compatibility with rest of code (using raw average)
        raw_avg_vqa_score = sum(s for _, s in individual_vqa_scores_cpu) / max(len(individual_vqa_scores_cpu), 1)
        vqa_score = torch.tensor(raw_avg_vqa_score, device=latents_init.device, requires_grad=False)
        individual_vqa_score = [torch.tensor(s, device=latents_init.device) for n_id, s in individual_vqa_scores_cpu]
        # ==============================================================
        
        # L-DINO-CoT: Compute localized VQA scores with visualization
        if use_ldino:
            print("\n[L-DINO-CoT] Evaluating localized scores (for logging)")
            with torch.no_grad():
                # Pass a CPU copy to L-DINO to avoid cross-device issues
                localized_loss, entity_info = self.ldino_optimizer.compute_localized_loss(
                    image.detach().cpu(), self.image_processor, 
                    step=0, total_steps=optimization_epoch, 
                    entities=entities, save_prefix="init",
                    # Pass pre-computed masks to avoid reloading models
                    all_masks=initial_masks
                )
            
            print(f"[L-DINO-CoT] Target Attribute Validations:")
            for info in entity_info:
                ref_score = info.get('reflection_score', 0)
                if isinstance(ref_score, (int, float)):
                    print(f"  {info['entity']} score: {ref_score:.3f}")
                else:
                    print(f"  {info['entity']} score: {ref_score}")
        
        max_vqa_score = vqa_score.item()
        self.vqa_history = [{"epoch": 0, "current_vqa": max_vqa_score, "best_vqa": max_vqa_score}]
        print(f"max_vqa_score: {max_vqa_score}")
        print(f"individual score: {', '.join([str(x.item()) for x in individual_vqa_score])}")
        target = image.detach().clone()  # Clone on cuda:0 for safety
        # Save 0.png directly from cuda:0 tensor
        image_0 = self.image_processor.postprocess(image.detach().cpu(), output_type="pil")[0]
        image_0.save(f"{dir}/0.png")
        
        # Store gradient for optimization loop (updated after each iteration)
        stored_grad = latents_init.grad.clone()
        
        for i in range(optimization_epoch):
            # Use mean of raw individual scores for step_lr to avoid 1.0 lr jump
            raw_avg = sum(v for _, v in individual_vqa_scores_cpu) / max(len(individual_vqa_scores_cpu), 1)
            vqa_score_val = max(raw_avg, 1e-6)
            step_lr = max(min(1 - vqa_score_val ** 0.5, 0.8), 0.01)

            grad = stored_grad
            grad_flat = grad.detach().view(-1)
            grad_norm = grad_flat.norm().item()

            n = grad_flat.numel()
            # Choose stable positive-definite parameters (theoretically optimal: alpha = n/(n-2)*beta)
            beta = 1.0
            alpha = math.sqrt(n)  # 90% of theoretical to ensure numerical safety

            noise_pool = []
            pool_size = 5  # Number of noise candidates to sample

            if grad_norm > 1e-6:
                # Preferred: fully directional, using only grad direction
                # 1. Generate a pool of 10 directional noise samples
                for _ in range(pool_size):
                    noise_sample = self.directional_gaussian_torch(grad_flat, alpha=alpha, beta=beta, generator=generator)
                    noise_pool.append(noise_sample.view(grad.shape))
            else:
                # Fallback: if grad is too small, use standard random noise
                for _ in range(pool_size):
                    noise_sample = torch.randn(grad.shape, device=grad.device, dtype=grad.dtype, generator=generator)
                    noise_pool.append(noise_sample)

            scores = []
            
            # Score each noise candidate with a cheap QUICK_STEPS-step preview + Tweedie x_0 estimate
            QUICK_STEPS = 20
            
            # Save scheduler state for restoration between candidates
            scheduler_snapshot = copy.deepcopy(self.scheduler)
            best_candidate_cache = None
            best_noise_idx = -1
            best_score = -float('inf')
            prompt_str = prompt[0] if isinstance(prompt, list) else prompt

            with torch.no_grad():
                for idx, noise_candidate in enumerate(noise_pool):
                    print(f"Candidate {idx} norm: {noise_candidate.norm().item():.3f} nan: {noise_candidate.isnan().any().item()}")
                    latents_tmp = (1 - step_lr) ** 0.5 * latents_init.detach() + step_lr ** 0.5 * noise_candidate
                    print(f"latents_tmp norm: {latents_tmp.norm().item():.3f} nan: {latents_tmp.isnan().any().item()}")

                    # Restore scheduler to pre-candidate state
                    self.scheduler = copy.deepcopy(scheduler_snapshot)
                    noise_list_tmp, tweedie_est, latents_at_quick_steps = denoise(latents_tmp, max_steps=QUICK_STEPS, return_tweedie=True, return_latents=True)
                    print(f"tweedie_est norm: {tweedie_est.norm().item():.3f} nan: {tweedie_est.isnan().any().item()}")

                    latents_fwd = tweedie_est / self.vae.config.scaling_factor
                    image_fwd = self.vae.decode(latents_fwd, return_dict=False)[0]
                    image_fwd = image_fwd.to(self.vqa_model_device)
                    image_fwd = (image_fwd / 2 + 0.5).clamp(0, 1)

                    with torch.cuda.device(self.vqa_model_device):
                        s = self.vqa_model(image_fwd, [prompt_str])
                        vqa_score_tmp = s.item()
                        del s

                    scores.append(vqa_score_tmp)
                    print(f"vqa_score_tmp (quick, {QUICK_STEPS} steps): {vqa_score_tmp}")

                    if vqa_score_tmp > best_score:
                        best_score = vqa_score_tmp
                        best_noise_idx = idx
                        best_candidate_cache = {
                            "latents": latents_at_quick_steps.detach(),
                            "noise_list": {k: v.detach() for k, v in noise_list_tmp.items()},
                            "scheduler": copy.deepcopy(self.scheduler)
                        }

                    # OPT #5: Only save candidate previews in debug mode
                    if save_debug:
                        image_fwd_pil = self.image_processor.postprocess(image_fwd.detach().cpu(), output_type="pil")[0]
                        candidate_dir = f"{dir}/epoch_{i}"
                        os.makedirs(candidate_dir, exist_ok=True)
                        image_fwd_pil.save(f"{candidate_dir}/candidate_{idx}_score{vqa_score_tmp:.3f}.png")

                    del latents_tmp, latents_at_quick_steps, tweedie_est, noise_list_tmp, latents_fwd, image_fwd
                    torch.cuda.empty_cache()

            # 3. Select the noise that maximizes the score
            best_noise_idx = np.argmax(scores)
            best_noise = noise_pool[best_noise_idx]
            print(f"[NoisePool] Best candidate: idx={best_noise_idx}, score={scores[best_noise_idx]:.4f}")

            latents_tmp = latents_init.detach()
            latents_tmp = (1 - step_lr) ** 0.5 * latents_tmp + step_lr ** 0.5 * best_noise

            latents_init = latents_tmp.detach()
            with torch.no_grad():
                # Resume generation from QUICK_STEPS using the cached state
                # Restore scheduler from best candidate
                self.scheduler = best_candidate_cache["scheduler"]
                latents_remaining = best_candidate_cache["latents"]
                noise_list_best_first_part = best_candidate_cache["noise_list"]
                
                noise_list_best_second_part = denoise(
                    latents_remaining,
                    start_step=QUICK_STEPS,
                    max_steps=num_inference_steps
                )
                
                # Combine the noise lists for gradient backprop
                noise_list = {**noise_list_best_first_part, **noise_list_best_second_part}
                self.scheduler.set_timesteps(num_inference_steps)
            latents_init.requires_grad = True
            latents = reverse(latents_init, noise_list)
            self.scheduler.set_timesteps(num_inference_steps)
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            # CRITICAL: Clone and keep on cuda:0 — .to(cuda:1) zeroes tensor data
            image = image.clone()
            # OPT #5: Only save full generation image in debug mode
            if save_debug:
                with torch.no_grad():
                    best_image_pil = self.image_processor.postprocess(image.detach().cpu(), output_type="pil")[0]
                    candidate_dir = f"{dir}/epoch_{i}"
                    os.makedirs(candidate_dir, exist_ok=True)
                    best_image_pil.save(f"{candidate_dir}/best_candidate_{best_noise_idx}_full.png")
            print(f"[NoisePool] Best candidate: idx={best_noise_idx} at epoch {i}")

            # ============== PER-PROMPT GRADIENT COMPUTATION (Optimization Loop) ==============
            gradients_cpu_epoch = []
            individual_vqa_scores_epoch = []
            
            import copy
            from lvqa_dino.differentiable_blur import apply_blur_mask
            
            # Update segmentation masks for the new generated image
            # This ensures gradients flow through the correct entity regions
            if use_ldino and self.ldino_optimizer and hasattr(self.ldino_optimizer, 'segmenter'):
                with torch.no_grad():
                    image_pil_epoch = self.image_processor.postprocess(image.detach().cpu(), output_type="pil")[0]
                    all_entity_names = [e.name for e in entities]
                    print(f"[L-DINO] Updating masks at epoch {i+1} for: {all_entity_names}")
                    # Recompute masks based on current generated image
                    current_masks = self.ldino_optimizer.segmenter.segment_multiple(image_pil_epoch, all_entity_names)
                    
                    # Save epoch mask images to ldino_debug/epoch_{i+1}/final/
                    mask_dir = f"{dir}/ldino_debug/epoch_{i+1}/final"
                    os.makedirs(mask_dir, exist_ok=True)
                    for e_name, m_np in current_masks.items():
                        if m_np is not None:
                            m_pil = Image.fromarray((m_np * 255).astype(np.uint8))
                            m_pil.save(f"{mask_dir}/mask_obj_{e_name}.png")
                    
                    # Save combined mask visualization
                    if hasattr(self.ldino_optimizer.segmenter, 'visualize_masks') and current_masks:
                        combined_np = self.ldino_optimizer.segmenter.visualize_masks(image_pil_epoch, current_masks)
                        Image.fromarray(combined_np).save(f"{mask_dir}/combined_masks.png")
            else:
                current_masks = getattr(self, "initial_masks", getattr(self, "_temp_initial_masks", {}))

            for n_id, p in graph_evaluator.questions.items():
                latents_init.requires_grad_(True) 
                
                if latents_init.grad is not None:
                    latents_init.grad.zero_()
                
                image_scaled = (image / 2 + 0.5).clamp(0, 1)
                image_cuda1 = image_scaled.detach().to(self.vqa_model_device).requires_grad_(True)
                target_image = image_cuda1
                matched_entity = None
                mask_np = None
                
                if use_ldino:
                    concept = graph_evaluator.nodes[n_id].get("concept", p)
                    for ent in entities:
                        if ent.name in concept:
                            matched_entity = ent.name
                            break
                    
                    if matched_entity and matched_entity in current_masks:
                        mask_np = current_masks[matched_entity]
                        if mask_np is not None:
                            mask_tensor = torch.from_numpy(mask_np).to(self.vqa_model_device).float()
                            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                            target_image = apply_blur_mask(image_cuda1, mask_tensor, blur_radius=21)
            
                with torch.cuda.device(self.vqa_model_device):
                    score = self.vqa_model(target_image, [p])
                    s_val = score.item()
                    individual_vqa_scores_epoch.append((n_id, s_val))
                    print(f"node: {n_id}, question: {p}, score: {s_val}")
                # Direct backpropagation across devices
                score.backward()
                
            grad_on_cuda0 = image_cuda1.grad.to(device=image.device, dtype=torch.float32)
            grad_on_cuda0 = torch.nan_to_num(grad_on_cuda0, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Since target_image is generated via image_scaled = (image / 2 + 0.5), we apply
            # the analytical chain rule grad(image_scaled)/dv = 0.5 
            grad_on_cuda0_pre_scale = grad_on_cuda0 / 2.0
            image.backward(grad_on_cuda0_pre_scale)
                
            if latents_init.grad is not None:
                grad_cpu = latents_init.grad.detach().cpu().clone()
                # Defense against bfloat16/cross-device NaN gradients
                grad_cpu = torch.nan_to_num(grad_cpu, nan=0.0, posinf=0.0, neginf=0.0)
                
                if use_ldino and matched_entity and matched_entity in current_masks:
                    mask_np_loc = current_masks[matched_entity]
                    if mask_np_loc is not None:
                        latent_h, latent_w = grad_cpu.shape[2], grad_cpu.shape[3]
                        mask_pil = Image.fromarray((mask_np_loc * 255).astype(np.uint8))
                        mask_latent = mask_pil.resize((latent_w, latent_h), Image.BILINEAR)
                        mask_latent = np.array(mask_latent) / 255.0
                        mask_tensor_latent = torch.from_numpy(mask_latent).float().unsqueeze(0).unsqueeze(0)
                        grad_cpu = grad_cpu * mask_tensor_latent
                        print(f"  [Grad Mask] Clipped gradient to '{matched_entity}' region")
                
                gradients_cpu_epoch.append((n_id, grad_cpu))
            else: 
                gradients_cpu_epoch.append((n_id, torch.zeros_like(latents_init.detach().cpu())))
            
            try:
                del score
            except:
                pass
                
            # Clear GPU gradient
            latents_init.grad = None
            gc.collect()
            torch.cuda.empty_cache()
            
            # Evaluate hierarchical VQA scores for epoch
            raw_scores_map_epoch = {n_id: s for n_id, s in individual_vqa_scores_epoch}
            masked_scores_epoch, root_scores_epoch, avg_score_epoch = graph_evaluator.evaluate(raw_scores_map_epoch)
            
            # Update individual scores array for the next epoch's step_lr calculation
            individual_vqa_scores_cpu = individual_vqa_scores_epoch
            
            root_gradients_epoch = []
            for root_id, tree_score in root_scores_epoch.items():
                tree_nodes = set()
                def dfs(nid):
                    if nid in tree_nodes: return
                    tree_nodes.add(nid)
                    for child_id, child in graph_evaluator.nodes.items():
                        parents = child.get("parent_id")
                        if isinstance(parents, list):
                            if nid in parents: dfs(child_id)
                        elif parents == nid:
                            dfs(child_id)
                dfs(root_id)
                
                tree_grad = torch.zeros_like(gradients_cpu_epoch[0][1])
                for n_id, g in gradients_cpu_epoch:
                    if n_id in tree_nodes:
                        # Use raw scores to avoid zero gradients when mask is 0
                        coeff = 1.0
                        for on_id in tree_nodes:
                            if on_id != n_id:
                                coeff *= max(raw_scores_map_epoch.get(on_id, 1e-9), 1e-9)
                        tree_grad += coeff * g
                root_gradients_epoch.append(tree_grad)
            
            if len(root_gradients_epoch) > 1:
                avg_gradient_epoch = torch.mean(torch.stack(root_gradients_epoch), dim=0)
            else:
                avg_gradient_epoch = root_gradients_epoch[0]
                
            if avg_gradient_epoch.isnan().any():
                print("NaN detected in avg_gradient_epoch before moving!")
            latents_init.grad = avg_gradient_epoch.to(latents_init.device)
            if latents_init.grad.isnan().any():
                print("NaN detected in latents_init.grad (epoch) after moving!")
            
            # Update stored_grad for next iteration
            stored_grad = latents_init.grad.clone()
            
            del gradients_cpu_epoch, avg_gradient_epoch, root_gradients_epoch
            gc.collect()
            torch.cuda.empty_cache()
            
            # Compute average score for logging (using raw average)
            raw_avg_score_epoch = sum(s for _, s in individual_vqa_scores_epoch) / max(len(individual_vqa_scores_epoch), 1)
            vqa_score = torch.tensor(raw_avg_score_epoch, device=latents_init.device, requires_grad=False)
            individual_vqa_score = [torch.tensor(s, device=latents_init.device) for n_id, s in individual_vqa_scores_epoch]
            
            print(f"vqa_score in epoch {i + 1}: {raw_avg_score_epoch}")
            print(f"individual score: {', '.join([str(x[1]) for x in individual_vqa_scores_epoch])}")
            self.vqa_history.append({"epoch": i + 1, "current_vqa": raw_avg_score_epoch, "best_vqa": max(max_vqa_score, raw_avg_score_epoch)})
            # ==================================================================================
            
            # L-DINO-CoT: Compute localized scores and save visualizations
            if use_ldino and entities is not None:
                with torch.no_grad():
                    loc_loss, entity_info = self.ldino_optimizer.compute_localized_loss(
                        image.detach().cpu(), self.image_processor,
                        step=i+1, total_steps=optimization_epoch,
                        entities=entities, save_prefix=f"epoch{i+1}",
                        all_masks=current_masks
                    )
                    self.ldino_optimizer.print_entity_scores(entity_info, step=i+1)
            
            # if avg_score_epoch > 0.95: # Early stopping
            #     print(f"Reached high score {avg_score_epoch}, stopping optimization.")
            #     break
            if vqa_score.item() > max_vqa_score:
                print("update target")
                target = image.detach().clone()
                max_vqa_score = vqa_score.item()
                # Update best_vqa in the last history entry
                self.vqa_history[-1]["best_vqa"] = max_vqa_score
            image = self.image_processor.postprocess(image.detach().cpu(), output_type=output_type)[0]

            latents_init.grad = None
            del noise_pool, scores
            gc.collect()
            torch.cuda.empty_cache()

            image.save(f'{dir}/{i + 1}.png')
        target = self.image_processor.postprocess(target.detach().cpu(), output_type=output_type)[0]
        target.save(f'{dir}/target.png')

        del vqa_score

        image = target

        # if not output_type == "latent":
        #     image = target
        #     has_nsfw_concept = None
        # else:
        #     image = latents
        #     has_nsfw_concept = None

        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
