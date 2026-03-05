# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
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
import os
import gc
import copy
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from PIL import Image
from transformers import (
    BaseImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    PreTrainedModel,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

# L-DINO-CoT: Localized VQA Scoring imports
from lvqa_dinot import (
    LDINOOptimizer, 
    EntityInfo, 
    create_entities_from_simple_format,
    GroundedSAMSegmenter,
    DependencyGraphEvaluator
)
from lvqa_dinot.differentiable_blur import apply_blur_mask

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from pipelines.pipeline_stable_diffusion_3_nd import StableDiffusion3NDPipeline

        >>> pipe = StableDiffusion3NDPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        ```
"""

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
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

class StableDiffusion3DiNOTPipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin, SD3IPAdapterMixin):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        image_encoder: PreTrainedModel = None,
        feature_extractor: BaseImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.patch_size = (
            self.transformer.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )
        
        # New: VQA and DINO state
        self.vqa_model = None
        self.vqa_model_device = None
        self.ldino_optimizer = None

    def init_vqa_model(self, vqa_model, device):
        self.vqa_model = vqa_model
        self.vqa_model_device = device
        self.ldino_optimizer = None # Will be initialized in __call__ if use_localized_vqa

    @staticmethod
    def directional_gaussian_torch(grad, alpha, beta, generator=None):
        """
        Samples a directional Gaussian noise vector in PyTorch.
        """
        n = grad.numel()
        device = grad.device
        dtype = grad.dtype
        g = grad.view(-1) / (grad.view(-1).norm() + 1e-9)
        lam1 = beta + alpha * (1 - 1/n)
        lam2 = beta - alpha / n

        # Sample standard normals
        w = torch.randn(n, device=device, dtype=dtype, generator=generator)
        c = torch.dot(g, w)
        w_perp = w - c * g

        noise = (lam1**0.5) * c * g + (lam2**0.5) * w_perp
        return noise.view(grad.shape)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
    ):
        device = device or self._execution_device

        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if (
            height % (self.vae_scale_factor * self.patch_size) != 0
            or width % (self.vae_scale_factor * self.patch_size) != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * self.patch_size} but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}"
            )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    def init_vqa_model(self, vqa_model, device):
        self.vqa_model = vqa_model
        self.vqa_model_device = device

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        optimization_epoch = 50,
        dependency_graph: Optional[Dict] = None,
        use_localized_vqa: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt, prompt_2, prompt_3, height, width,
            negative_prompt, negative_prompt_2, negative_prompt_3,
            prompt_embeds, negative_prompt_embeds,
            pooled_prompt_embeds, negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs, max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt, prompt_2=prompt_2, prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        num_channels_latents = self.transformer.config.in_channels
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

        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
            
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
        )
        self._num_timesteps = len(timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # Optimization Setup
        def denoise(latents, prompt_embeds=prompt_embeds, max_steps=None, return_latents=False, return_tweedie=False, start_step=0):
            max_steps = max_steps if max_steps is not None else num_inference_steps
            noise_list = {}
            tweedie_est = None
            
            for i, t in enumerate(timesteps):
                if i < start_step:
                    continue
                if i >= max_steps:
                    break
                
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                noise_list[t] = noise_pred.detach()
                
                # For Tweedie estimate in Flow Matching: x_0 = x_t - t * v_t
                # Here sigma = t in diffusers FlowMatchEulerDiscreteScheduler
                if return_tweedie and (i == max_steps - 1 or i == len(timesteps) - 1):
                    # Normalized sigma for flow matching is t/1000 usually
                    sigma = t / 1000.0  
                    tweedie_est = latents - sigma * noise_pred
                
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                if XLA_AVAILABLE:
                    xm.mark_step()
            
            if return_tweedie and return_latents:
                return noise_list, tweedie_est, latents
            if return_tweedie:
                return noise_list, tweedie_est
            if return_latents:
                return noise_list, latents
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
        
        # SD3 VAE decode logic
        latents_decoded = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents_decoded, return_dict=False)[0]
        
        dir = f"sd3_dinot_details/{prompt[:20].replace(' ', '_')}_{generator.initial_seed()}"
        os.makedirs(dir, exist_ok=True)
        
        # Save initial generated image (decoded under no_grad to avoid float16 corruption)
        with torch.no_grad():
            init_pil = self.image_processor.postprocess(image.detach().cpu(), output_type="pil")[0]
            init_pil.save(f"{dir}/0.png")
            print(f"[SD3-DINO] Saved initial image to {dir}/0.png")
        
        image = image.to(self.vqa_model_device)
        
        if dependency_graph is None:
            dependency_graph = {
                "nodes": [{"id": "q0", "type": "Entity", "concept": prompt, "question": prompt, "parent_id": None}]
            }
            
        graph_evaluator = DependencyGraphEvaluator(dependency_graph)
        
        # Segment and mask if use_localized_vqa
        use_ldino = use_localized_vqa
        entities = None
        initial_masks = {}
        
        if use_ldino:
            if self.ldino_optimizer is None:
                from lvqa_dinot import LDINOOptimizer
                self.ldino_optimizer = LDINOOptimizer(
                    vqa_model=self.vqa_model,
                    device=self.vqa_model_device,
                    save_visualizations=True
                )
            
            auto_entity_attributes = {
                n["concept"]: [] for n in graph_evaluator.nodes.values() if n.get("type") == "Entity"
            }
            
            if auto_entity_attributes:
                entities = self.ldino_optimizer.setup_entities(prompt, auto_entity_attributes, output_dir=dir)
                with torch.no_grad():
                    image_pil = self.image_processor.postprocess(image.detach().cpu(), output_type="pil")[0]
                    all_entity_names = [e.name for e in entities]
                    print(f"[L-DINO] Segmenting entities: {all_entity_names}")
                    initial_masks = self.ldino_optimizer.segmenter.segment_multiple(image_pil, all_entity_names)
                    
                    # Save debug masks
                    mask_dir = f"{dir}/ldino_debug/initial"
                    os.makedirs(mask_dir, exist_ok=True)
                    for e_name, m_np in initial_masks.items():
                        if m_np is not None:
                            m_pil = Image.fromarray((m_np * 255).astype(np.uint8))
                            m_pil.save(f"{mask_dir}/mask_{e_name}.png")
                    print(f"[L-DINO] Saved initial masks to {mask_dir}/")

        # VQA Scoring Loop (Initial Gradient)
        gradients_cpu = []
        individual_scores = []
        
        for n_id, q_text in graph_evaluator.questions.items():
            if latents_init.grad is not None:
                latents_init.grad.zero_()
                
            curr_latents = reverse(latents_init, noise_list)
            self.scheduler.set_timesteps(num_inference_steps)
            curr_latents_dec = (curr_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            curr_image_raw = self.vae.decode(curr_latents_dec.to(self.vae.dtype), return_dict=False)[0]
            curr_image_scaled = (curr_image_raw.float() / 2 + 0.5).clamp(0, 1)
            # CPU round-trip to fix cross-GPU P2P copy bug (detach + requires_grad for relay)
            curr_image = curr_image_scaled.detach().cpu().to(self.vqa_model_device).requires_grad_(True)
            
            target_image = curr_image
            matched_entity = None
            
            if use_ldino:
                concept = graph_evaluator.nodes[n_id].get("concept", q_text)
                for ent in entities:
                    if ent.name in concept:
                        matched_entity = ent.name
                        break
                
                if matched_entity and matched_entity in initial_masks:
                    mask_np = initial_masks[matched_entity]
                    if mask_np is not None:
                        mask_tensor = torch.from_numpy(mask_np).to(self.vqa_model_device).float().unsqueeze(0).unsqueeze(0)
                        target_image = apply_blur_mask(curr_image, mask_tensor)
            
            score = self.vqa_model(target_image, [q_text])
            score_val = score.item()
            if math.isnan(score_val):
                score_val = 0.0
            individual_scores.append((n_id, score_val))
            
            score.backward()
            
            # Gradient relay: copy grad from vqa device back through VAE to latents_init
            grad_on_sd = curr_image.grad.cpu().to(curr_image_raw.device, dtype=torch.float32) if curr_image.grad is not None else torch.zeros_like(curr_image_raw).float()
            grad_on_sd = torch.nan_to_num(grad_on_sd, nan=0.0, posinf=0.0, neginf=0.0)
            curr_image_raw.backward(grad_on_sd / 2.0)
            
            if latents_init.grad is not None:
                grad_cpu = latents_init.grad.detach().cpu().clone()
                grad_cpu = torch.nan_to_num(grad_cpu, nan=0.0, posinf=0.0, neginf=0.0)
                if use_ldino and matched_entity and matched_entity in initial_masks:
                    # Mask gradient in latent space
                    mask_np_loc = initial_masks[matched_entity]
                    mask_pil = Image.fromarray((mask_np_loc * 255).astype(np.uint8))
                    mask_latent = mask_pil.resize((grad_cpu.shape[3], grad_cpu.shape[2]), Image.BILINEAR)
                    mask_tensor_latent = torch.from_numpy(np.array(mask_latent)/255.0).float().unsqueeze(0).unsqueeze(0)
                    grad_cpu = grad_cpu * mask_tensor_latent
                
                gradients_cpu.append((n_id, grad_cpu))
        
        # Evaluate graph
        raw_scores_map = dict(individual_scores)
        masked_scores, root_scores, avg_vqa_score = graph_evaluator.evaluate(raw_scores_map)
        
        # Build composite gradient (Hierarchical)
        root_gradients = []
        for root_id, _ in root_scores.items():
            tree_nodes = set()
            def dfs(nid):
                if nid in tree_nodes: return
                tree_nodes.add(nid)
                for child_id, child in graph_evaluator.nodes.items():
                    parents = child.get("parent_id")
                    if (isinstance(parents, list) and nid in parents) or parents == nid:
                        dfs(child_id)
            dfs(root_id)
            
            tree_grad = torch.zeros_like(gradients_cpu[0][1])
            for n_id, g in gradients_cpu:
                if n_id in tree_nodes:
                    # Use RAW scores to avoid zero gradient when all masked scores are 0
                    coeff = 1.0
                    for on_id in tree_nodes:
                        if on_id != n_id:
                            coeff *= max(raw_scores_map.get(on_id, 1e-9), 1e-9)
                    tree_grad += coeff * g
            root_gradients.append(tree_grad)
            
        final_grad = torch.mean(torch.stack(root_gradients), dim=0) if root_gradients else torch.zeros_like(latents_init.detach().cpu())
        stored_grad = final_grad.to(latents_init.device)
        stored_grad = torch.nan_to_num(stored_grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = stored_grad.float().norm().item()
        MAX_GRAD_NORM = 100.0
        if grad_norm > MAX_GRAD_NORM:
            stored_grad = stored_grad * (MAX_GRAD_NORM / grad_norm)
            print(f"[Gradient] Clipped grad norm: {grad_norm:.2f} -> {MAX_GRAD_NORM}")
        print(f"[Gradient] stored_grad norm: {stored_grad.float().norm().item():.6f}")
        
        # Optimization Epochs
        raw_avg = sum(v for _, v in individual_scores) / max(len(individual_scores), 1)
        max_vqa = raw_avg
        target_latents = latents_init.detach().clone()
        vqa_score_val = max(raw_avg, 1e-6)
        print(f"\n[SD3-DINO] Initial raw avg VQA: {raw_avg:.4f}, hierarchical: {avg_vqa_score:.4f}")
        print(f"[SD3-DINO] Starting {optimization_epoch} optimization epochs...")
        
        for ep in range(optimization_epoch):
            # Guard against NaN in vqa_score_val
            if math.isnan(vqa_score_val):
                vqa_score_val = max(max_vqa, 1e-6)
            step_lr = max(min(1 - vqa_score_val ** 0.5, 0.3), 0.01)
            grad = stored_grad
            
            # 5-candidate noise pool
            pool_size = 5
            QUICK_STEPS = 20
            noise_pool = []
            
            # Sample directional noise
            grad_flat = grad.view(-1)
            n = grad_flat.numel()
            alpha = math.sqrt(n)
            beta = 1.0
            
            for _ in range(pool_size):
                noise_sample = self.directional_gaussian_torch(grad_flat, alpha=alpha, beta=beta, generator=generator)
                noise_pool.append(noise_sample.view(grad.shape))
            
            # Select best candidate with 20-step preview
            candidate_scores = []
            candidate_caches = []
            
            with torch.no_grad():
                for noise_cand in noise_pool:
                    # Additive noise: gentle perturbation instead of spherical interpolation
                    lat_std = latents_init.float().std().item()
                    latents_tmp = (latents_init.detach() + step_lr * lat_std * noise_cand).to(latents_init.dtype)
                    
                    # Store scheduler state
                    scheduler_checkpoint = copy.deepcopy(self.scheduler)
                    noise_list_tmp, tweedie_est, latents_preview = denoise(
                        latents_tmp, max_steps=QUICK_STEPS, return_tweedie=True, return_latents=True
                    )
                    
                    # Decode Tweedie estimate in float32
                    lat_dec = (tweedie_est / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    old_vae_dtype = self.vae.dtype
                    self.vae.to(dtype=torch.float32)
                    img_preview = self.vae.decode(lat_dec.float(), return_dict=False)[0]
                    self.vae.to(dtype=old_vae_dtype)
                    # CPU round-trip for cross-GPU transfer
                    img_preview = img_preview.float().cpu().to(self.vqa_model_device)
                    
                    # Score against the main prompt (or averaged graph if complex)
                    s_preview = self.vqa_model(img_preview, [prompt]).item()
                    candidate_scores.append(s_preview)
                    candidate_caches.append({
                        "latents": latents_preview.detach(),
                        "noise_list": {k: v.detach() for k, v in noise_list_tmp.items()},
                        "scheduler": copy.deepcopy(self.scheduler)
                    })
                    
                    # Restore scheduler for next candidate
                    self.scheduler = scheduler_checkpoint
            
            best_idx = np.argmax(candidate_scores)
            best_noise = noise_pool[best_idx]
            
            # Update latents and resume from preview
            lat_std = latents_init.float().std().item()
            latents_init = (latents_init.detach() + step_lr * lat_std * best_noise).to(latents_init.dtype).detach().requires_grad_(True)
            
            with torch.no_grad():
                best_cache = candidate_caches[best_idx]
                self.scheduler = best_cache["scheduler"]
                latents_remaining = best_cache["latents"]
                noise_list_first = best_cache["noise_list"]
                
                noise_list_second = denoise(
                    latents_remaining, start_step=QUICK_STEPS, max_steps=num_inference_steps
                )
                
                noise_list = {**noise_list_first, **noise_list_second}
                self.scheduler.set_timesteps(num_inference_steps)
                
            # Decode WITHOUT gradient for image saving
            # (SD3 VAE produces corrupt grey output in float16 when grad is tracked)
            with torch.no_grad():
                curr_latents_save = reverse(latents_init.detach(), noise_list)
                self.scheduler.set_timesteps(num_inference_steps)
                save_lat_dec = (curr_latents_save / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                opt_image_save = self.vae.decode(save_lat_dec, return_dict=False)[0]
                ep_pil = self.image_processor.postprocess(opt_image_save.detach().cpu(), output_type="pil")[0]
                ep_pil.save(f"{dir}/epoch_{ep}.png")
            
            # Compute new gradients and hierarchical score
            new_individual_scores = []
            new_gradients_cpu = []
            
            # Update segmentation masks on the new generated image
            if use_ldino and self.ldino_optimizer and hasattr(self.ldino_optimizer, 'segmenter'):
                with torch.no_grad():
                    all_entity_names = [e.name for e in entities]
                    print(f"  [L-DINO] Updating masks at epoch {ep} for: {all_entity_names}")
                    current_masks = self.ldino_optimizer.segmenter.segment_multiple(ep_pil, all_entity_names)
                    
                    mask_dir = f"{dir}/ldino_debug/epoch_{ep}"
                    os.makedirs(mask_dir, exist_ok=True)
                    for e_name, m_np in current_masks.items():
                        if m_np is not None:
                            m_pil = Image.fromarray((m_np * 255).astype(np.uint8))
                            m_pil.save(f"{mask_dir}/mask_{e_name}.png")
                    if hasattr(self.ldino_optimizer.segmenter, 'visualize_masks') and current_masks:
                        combined_np = self.ldino_optimizer.segmenter.visualize_masks(ep_pil, current_masks)
                        Image.fromarray(combined_np).save(f"{mask_dir}/combined_masks.png")
                    print(f"  [L-DINO] Saved epoch {ep} masks to {mask_dir}/")
            else:
                current_masks = initial_masks
                
            for n_id, q_text in graph_evaluator.questions.items():
                if latents_init.grad is not None:
                    latents_init.grad.zero_()
                
                # Re-run forward with gradient for this specific node
                latents_for_node = reverse(latents_init, noise_list)
                self.scheduler.set_timesteps(num_inference_steps)
                lat_dec_node = (latents_for_node / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                img_for_node_raw = self.vae.decode(lat_dec_node.to(self.vae.dtype), return_dict=False)[0]
                img_for_node_scaled = (img_for_node_raw.float() / 2 + 0.5).clamp(0, 1)
                img_for_node = img_for_node_scaled.detach().cpu().to(self.vqa_model_device).requires_grad_(True)
                
                target_img_node = img_for_node
                matched_ent_node = None
                if use_ldino:
                    concept_node = graph_evaluator.nodes[n_id].get("concept", q_text)
                    for ent in entities:
                        if ent.name in concept_node:
                            matched_ent_node = ent.name
                            break
                    if matched_ent_node and matched_ent_node in current_masks:
                        mk_np = current_masks[matched_ent_node]
                        if mk_np is not None:
                            mk_ts = torch.from_numpy(mk_np).to(self.vqa_model_device).float().unsqueeze(0).unsqueeze(0)
                            target_img_node = apply_blur_mask(img_for_node, mk_ts)
                
                node_score = self.vqa_model(target_img_node, [q_text])
                s_val = node_score.item()
                if math.isnan(s_val):
                    s_val = 0.0
                    print(f"  [WARN] NaN VQA score for node {n_id}, using 0.0")
                new_individual_scores.append((n_id, s_val))
                node_score.backward()
                
                # Gradient relay: copy grad from vqa device back through VAE to latents_init
                grad_relay = img_for_node.grad.cpu().to(img_for_node_raw.device, dtype=torch.float32) if img_for_node.grad is not None else torch.zeros_like(img_for_node_raw).float()
                grad_relay = torch.nan_to_num(grad_relay, nan=0.0, posinf=0.0, neginf=0.0)
                img_for_node_raw.backward(grad_relay / 2.0)
                
                if latents_init.grad is not None:
                    g_cpu = latents_init.grad.detach().cpu().clone()
                    g_cpu = torch.nan_to_num(g_cpu, nan=0.0, posinf=0.0, neginf=0.0)
                    if use_ldino and matched_ent_node and matched_ent_node in current_masks:
                        mk_np_loc = current_masks[matched_ent_node]
                        mk_pil = Image.fromarray((mk_np_loc * 255).astype(np.uint8))
                        mk_lat = mk_pil.resize((g_cpu.shape[3], g_cpu.shape[2]), Image.BILINEAR)
                        mk_ts_lat = torch.from_numpy(np.array(mk_lat)/255.0).float().unsqueeze(0).unsqueeze(0)
                        g_cpu = g_cpu * mk_ts_lat
                    new_gradients_cpu.append((n_id, g_cpu))
            
            raw_map_ep = dict(new_individual_scores)
            masked_ep, roots_ep, _ = graph_evaluator.evaluate(raw_map_ep)
            raw_avg_ep = sum(v for _, v in new_individual_scores) / max(len(new_individual_scores), 1)
            vqa_score_val = max(raw_avg_ep, 1e-6)
            
            # Reconstruct stored_grad using RAW scores (not masked) to avoid zero gradients
            root_gradients_ep = []
            for root_id, _ in roots_ep.items():
                tree_nodes = set()
                def dfs(nid):
                    if nid in tree_nodes: return
                    tree_nodes.add(nid)
                    for child_id, child in graph_evaluator.nodes.items():
                        parents = child.get("parent_id")
                        if (isinstance(parents, list) and nid in parents) or parents == nid:
                            dfs(child_id)
                dfs(root_id)
                
                tree_grad = torch.zeros_like(new_gradients_cpu[0][1])
                for n_id, g in new_gradients_cpu:
                    if n_id in tree_nodes:
                        coeff = 1.0
                        for on_id in tree_nodes:
                            if on_id != n_id:
                                coeff *= max(raw_map_ep.get(on_id, 1e-9), 1e-9)
                        tree_grad += coeff * g
                root_gradients_ep.append(tree_grad)
                
            stored_grad = torch.mean(torch.stack(root_gradients_ep), dim=0).to(latents_init.device) if root_gradients_ep else torch.zeros_like(latents_init.detach().cpu()).to(latents_init.device)
            stored_grad = torch.nan_to_num(stored_grad, nan=0.0, posinf=0.0, neginf=0.0)
            g_norm_ep = stored_grad.float().norm().item()
            if g_norm_ep > MAX_GRAD_NORM:
                stored_grad = stored_grad * (MAX_GRAD_NORM / g_norm_ep)
            
            if vqa_score_val > max_vqa:
                max_vqa = vqa_score_val
                target_latents = latents_init.detach().clone()
            
            print(f"  [Epoch {ep}] VQA={vqa_score_val:.4f} (best={max_vqa:.4f}), lr={step_lr:.4f}")
            gc.collect()
            torch.cuda.empty_cache()
            
        # Final Decoding
        print(f"\n[SD3-DINO] Optimization complete! Best VQA: {max_vqa:.4f}")
        with torch.no_grad():
            noise_list = denoise(target_latents)
            self.scheduler.set_timesteps(num_inference_steps)
            final_latents = reverse(target_latents, noise_list)
            self.scheduler.set_timesteps(num_inference_steps)
            final_lat_dec = (final_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            final_image = self.vae.decode(final_lat_dec, return_dict=False)[0]
            
        images = self.image_processor.postprocess(final_image.detach().cpu(), output_type=output_type)
        
        self.maybe_free_model_hooks()
        if not return_dict: return (images,)
        return StableDiffusion3PipelineOutput(images=images)
