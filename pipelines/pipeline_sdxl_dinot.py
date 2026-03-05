import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import gc
import os
import copy
import numpy as np
from PIL import Image

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    StableDiffusionXLPipelineOutput,
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.image_processor import PipelineImageInput

from lvqa_dino import DependencyGraphEvaluator


class StableDiffusionXLDiNOTPipeline(StableDiffusionXLPipeline):
    """
    Stable Diffusion XL pipeline with Noise Direction (ND) optimization using LVQA/DINO.
    """
    def init_vqa_model(self, vqa_model, device):
        self.vqa_model = vqa_model
        self.vqa_model_device = device
        
    def directional_gaussian_torch(self, grad: torch.Tensor, alpha: float, beta: float, generator: torch.Generator) -> torch.Tensor:
        """
        Samples a vector from a generalized directional Gaussian distribution in PyTorch.
        """
        device = grad.device
        dim = grad.numel()
        grad = grad.view(-1)
        grad_norm = grad.norm(p=2) + 1e-8
        mu = grad / grad_norm

        z = torch.randn(dim, generator=generator, device=device)
        u = z - (z @ mu) * mu
        u_norm = u.norm(p=2) + 1e-8

        s_sq_dist = torch.distributions.Chi2(torch.tensor([dim - 1.0], device=device))
        s = torch.sqrt(s_sq_dist.sample())
        w_dist = torch.distributions.Normal(torch.tensor([alpha], device=device), torch.tensor([beta], device=device))
        w = w_dist.sample()

        x = w * mu + s * (u / u_norm)
        return x

    #@torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # Optimization args
        dependency_graph: Optional[Dict] = None,
        optimization_epoch: int = 100,
        decomposed_prompts: Optional[List[str]] = None,
        use_localized_vqa: bool = True,
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            None,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
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

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
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

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        
        # --- ND Denoising setup ---
        def denoise(latents, max_steps=None, return_latents=False, return_tweedie=False, start_step=0):
            max_steps = max_steps if max_steps is not None else num_inference_steps
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            
            with self.progress_bar(total=(max_steps - start_step)) as progress_bar:
                noise_list = {}
                for i, t in enumerate(timesteps):
                    if i < start_step:
                        continue
                    if i >= max_steps:
                        break
                    if self.interrupt:
                        continue
                        
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    noise_list[t] = noise_pred.detach()
                    latents_dtype = latents.dtype
                    current_latents = latents
                    step_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                    if hasattr(step_output, "prev_sample"):
                        latents = step_output.prev_sample
                    else:
                        latents = step_output[0]
                        
                    if latents.dtype != latents_dtype:
                        latents = latents.to(latents_dtype)

                    if return_tweedie and (i == max_steps - 1 or i == len(timesteps) - 1):
                        if getattr(step_output, "pred_original_sample", None) is not None:
                            tweedie_est = step_output.pred_original_sample
                        else:
                            alpha_prod_t = self.scheduler.alphas_cumprod[t.item() if torch.is_tensor(t) else t].to(latents.device)
                            beta_prod_t = 1 - alpha_prod_t
                            tweedie_est = (current_latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

                    if callback_on_step_end is not None:
                        callback_kwargs = {"latents": latents}
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)

                    if i == max_steps - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            if return_tweedie and return_latents: return noise_list, tweedie_est, latents
            if return_tweedie: return noise_list, tweedie_est
            if return_latents: return noise_list, latents
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

        # Initial reverse + VAE decode under no_grad (just for the initial image)
        with torch.no_grad():
            latents_rev = reverse(latents_init, noise_list)
            self.scheduler.set_timesteps(num_inference_steps)
        
        # Scaling logic for VAE in SDXL
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        
        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents_rev.device, latents_rev.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents_rev.device, latents_rev.dtype)
        
        with torch.no_grad():
            if has_latents_mean and has_latents_std:
                latents_dec = latents_rev * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents_dec = latents_rev / self.vae.config.scaling_factor
            
            if needs_upcasting:
                self.upcast_vae()
                latents_dec = latents_dec.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            
            image = self.vae.decode(latents_dec, return_dict=False)[0]
            
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            
            image = image.float()

        # -- ND optimization logic --
        dir = f"sdxl_dinot_details/{prompt[:25].replace(' ', '_')}/{generator.initial_seed()}"
        save_debug = True
        os.makedirs(dir, exist_ok=True)

        image = image.clone()

        # Entity segmentation + masks
        if dependency_graph is None:
            if decomposed_prompts is None:
                decomposed_prompts = [prompt] if isinstance(prompt, str) else prompt
            print("[Warning] No dependency_graph provided, falling back to flat list evaluation.")
            dependency_graph = {
                "nodes": [
                    {"id": f"q{i}", "type": "Entity", "concept": p, "question": p, "parent_id": None}
                    for i, p in enumerate(decomposed_prompts)
                ]
            }
        
        graph_evaluator = DependencyGraphEvaluator(dependency_graph)
        print(f"[Graph] Evaluator initialized with {len(graph_evaluator.nodes)} logical nodes.")
        
        entities = None
        use_ldino = use_localized_vqa
        
        if use_ldino:
            auto_entity_attributes = {
                n["concept"]: [] for n in graph_evaluator.nodes.values() if n.get("type", "Entity") == "Entity"
            }
            if not hasattr(self, 'ldino_optimizer') or self.ldino_optimizer is None:
                from lvqa_dino import LDINOOptimizer
                self.ldino_optimizer = LDINOOptimizer(
                    vqa_model=self.vqa_model,
                    device=self.vqa_model_device,
                    warmup_ratio=0.2,
                    lambda_ref=1.0,
                    save_visualizations=True
                )
            
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
            else:
                use_ldino = False
        
        # Aggressive memory cleanup before VQA scoring
        gc.collect()
        torch.cuda.empty_cache()

        from lvqa_dino.differentiable_blur import apply_blur_mask
        
        # --- Initial Graph Evaluation WITH gradient ---
        # Only track grad through reverse + VAE decode (not through UNet's denoise)
        print(f"\n[SDXL-DINO] Initial Graph Evaluation:")
        individual_vqa_scores_cpu = []
        gradients_cpu = []
        
        for n_id, p in graph_evaluator.questions.items():
            # Fresh gradient computation: reverse(latents_init) -> VAE decode -> VQA backward
            # This tracks grad through scheduler.step + VAE decode only (not UNet)
            latents_init_f32 = latents_init.float().detach().requires_grad_(True)
            
            # Upcast VAE decoder to float32 for clean gradients (no NaN from fp16)
            if needs_upcasting:
                self.upcast_vae()
            
            lat = reverse(latents_init_f32, noise_list)
            self.scheduler.set_timesteps(num_inference_steps)
            if has_latents_mean and has_latents_std:
                lat_d = (lat * latents_std.float() / self.vae.config.scaling_factor) + latents_mean.float()
            else:
                lat_d = lat / self.vae.config.scaling_factor
            
            # VAE is already float32 from upcast_vae(), decode in float32 for clean backward
            img_vqa = self.vae.decode(lat_d.to(next(iter(self.vae.post_quant_conv.parameters())).dtype), return_dict=False)[0]
            
            img_vqa_scaled = (img_vqa.float() / 2 + 0.5).clamp(0, 1)
            # CUDA sync to prevent deadlock before CPU copy
            torch.cuda.synchronize()
            img_vqa_cuda1 = img_vqa_scaled.detach().cpu().to(self.vqa_model_device).requires_grad_(True)
            target_image = img_vqa_cuda1
            matched_entity = None
            mask_np = None
            
            if use_ldino:
                concept = graph_evaluator.nodes[n_id].get("concept", p)
                for ent in entities:
                    if ent.name in concept: matched_entity = ent.name; break
                if matched_entity and matched_entity in initial_masks:
                    mask_np = initial_masks[matched_entity]
                    if mask_np is not None:
                        target_image = apply_blur_mask(img_vqa_cuda1, torch.from_numpy(mask_np).to(self.vqa_model_device).float().unsqueeze(0).unsqueeze(0), blur_radius=21)

            with torch.cuda.device(self.vqa_model_device):
                score = self.vqa_model(target_image, [p])
                score_val = score.item()
                individual_vqa_scores_cpu.append((n_id, score_val))
                print(f"node: {n_id}, question: {p}, score: {score_val}")
                
            score.backward()
            
            # Transfer gradient from cuda:1 back to cuda:0 image space
            grad_on_cuda0 = img_vqa_cuda1.grad.to(device=img_vqa.device, dtype=torch.float32) if img_vqa_cuda1.grad is not None else torch.zeros_like(img_vqa).float()
            grad_on_cuda0 = torch.nan_to_num(grad_on_cuda0, nan=0.0, posinf=0.0, neginf=0.0)
            img_vqa.backward(grad_on_cuda0 / 2.0)

            if latents_init_f32.grad is not None:
                grad_cpu = latents_init_f32.grad.detach().cpu().clone()
                grad_cpu = torch.nan_to_num(grad_cpu, nan=0.0, posinf=0.0, neginf=0.0)
                if use_ldino and matched_entity and matched_entity in initial_masks:
                    mask_np_loc = initial_masks[matched_entity]
                    if mask_np_loc is not None:
                        latent_h, latent_w = grad_cpu.shape[2], grad_cpu.shape[3]
                        mask_latent = np.array(Image.fromarray((mask_np_loc * 255).astype(np.uint8)).resize((latent_w, latent_h), Image.BILINEAR)) / 255.0
                        grad_cpu = grad_cpu * torch.from_numpy(mask_latent).float().unsqueeze(0).unsqueeze(0)
                gradients_cpu.append((n_id, grad_cpu))
            else:
                gradients_cpu.append((n_id, torch.zeros_like(latents_init.detach().cpu())))
            
            del score, img_vqa, target_image, latents_init_f32
            gc.collect(); torch.cuda.empty_cache()
            
            # Restore VAE to fp16
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        
        raw_scores_map = {n_id: s for n_id, s in individual_vqa_scores_cpu}
        masked_scores, root_scores, avg_vqa_score = graph_evaluator.evaluate(raw_scores_map)
        
        raw_avg = sum(v for _, v in individual_vqa_scores_cpu) / max(len(individual_vqa_scores_cpu), 1)
        max_vqa = raw_avg
        print(f"\n[SDXL-DINO] Initial raw avg VQA: {raw_avg:.4f}, hierarchical: {avg_vqa_score:.4f}")
        
        root_gradients = []
        for root_id, _ in root_scores.items():
            tree_nodes = set()
            def dfs(nid):
                if nid in tree_nodes: return
                tree_nodes.add(nid)
                for child_id, child in graph_evaluator.nodes.items():
                    parents = child.get("parent_id")
                    if (isinstance(parents, list) and nid in parents) or parents == nid: dfs(child_id)
            dfs(root_id)
            tree_grad = torch.zeros_like(gradients_cpu[0][1])
            for n_id, g in gradients_cpu:
                if n_id in tree_nodes:
                    coeff = 1.0
                    for on_id in tree_nodes:
                        if on_id != n_id: coeff *= max(raw_scores_map.get(on_id, 1e-9), 1e-9)
                    tree_grad += coeff * g
            root_gradients.append(tree_grad)
            
        final_grad = torch.mean(torch.stack(root_gradients), dim=0) if root_gradients else torch.zeros_like(latents_init.detach().cpu())
        stored_grad = final_grad.to(latents_init.device)
        print(f"[SDXL-DINO] stored_grad norm: {stored_grad.float().norm().item():.6f}")
        
        image_0 = self.image_processor.postprocess(image.detach().cpu(), output_type="pil")[0]
        image_0.save(f"{dir}/0.png")
        
        target = image.detach().clone()
        vqa_score_val = max(raw_avg, 1e-6)
        
        for i in range(optimization_epoch):
            step_lr = max(min(1 - vqa_score_val ** 0.5, 0.8), 0.01)

            grad_flat = stored_grad.detach().float().view(-1)
            grad_norm = grad_flat.norm().item()
            
            # Clip gradient to prevent blow-up in SDXL (latents have much higher variance than SD1.4)
            MAX_GRAD_NORM = 100.0
            if grad_norm > MAX_GRAD_NORM:
                stored_grad = stored_grad * (MAX_GRAD_NORM / grad_norm)
                grad_flat = stored_grad.detach().float().view(-1)
                grad_norm = grad_flat.norm().item()
                print(f"[SDXL-DINO] Clipped grad norm to {grad_norm:.2f}")

            n = grad_flat.numel()
            beta = 1.0
            alpha = math.sqrt(n)

            noise_pool = []
            pool_size = 5

            if grad_norm > 1e-6:
                for _ in range(pool_size):
                    ns = self.directional_gaussian_torch(grad_flat, alpha=alpha, beta=beta, generator=generator)
                    noise_pool.append(ns.view(stored_grad.shape).to(latents_init.dtype))
            else:
                for _ in range(pool_size):
                    ns = torch.randn(stored_grad.shape, device=stored_grad.device, dtype=latents_init.dtype, generator=generator)
                    noise_pool.append(ns)

            scores = []
            QUICK_STEPS = 12
            scheduler_snapshot = copy.deepcopy(self.scheduler)
            best_candidate_cache = None
            best_noise_idx = -1
            best_score = -float('inf')
            prompt_str = prompt[0] if isinstance(prompt, list) else prompt

            with torch.no_grad():
                for idx, noise_candidate in enumerate(noise_pool):
                    lat_std = latents_init.float().std().item()
                    latents_tmp = latents_init.detach() + step_lr * lat_std * noise_candidate
                    self.scheduler = copy.deepcopy(scheduler_snapshot)
                    noise_list_tmp, tweedie_est, latents_at_quick_steps = denoise(latents_tmp, max_steps=QUICK_STEPS, return_tweedie=True, return_latents=True)
                    
                    if has_latents_mean and has_latents_std:
                        lat_d = (tweedie_est * latents_std / self.vae.config.scaling_factor) + latents_mean
                    else:
                        lat_d = tweedie_est / self.vae.config.scaling_factor
                    
                    # Force ENTIRE VAE to float32 for correct decoding of noisy tweedie estimates
                    old_vae_dtype = self.vae.dtype
                    self.vae.to(dtype=torch.float32)
                    lat_d_up = lat_d.to(torch.float32)
                    image_fwd_raw = self.vae.decode(lat_d_up, return_dict=False)[0]
                    self.vae.to(dtype=old_vae_dtype)
                    
                    # CPU round-trip to fix PyTorch cross-GPU copy bug (P2P transfers silently return stale data)
                    # CUDA sync to prevent deadlock before CPU copy
                    torch.cuda.synchronize()
                    image_fwd = image_fwd_raw.float().cpu().to(self.vqa_model_device)
                    image_fwd = (image_fwd / 2 + 0.5).clamp(0, 1)

                    with torch.cuda.device(self.vqa_model_device):
                        s = self.vqa_model(image_fwd, [prompt_str])
                        vqa_score_tmp = s.item()

                    scores.append(vqa_score_tmp)
                    print(f"    [cand={idx}] VQA={vqa_score_tmp:.4f}")

                    if vqa_score_tmp > best_score:
                        best_score = vqa_score_tmp
                        best_noise_idx = idx
                        best_candidate_cache = {
                            "latents": latents_at_quick_steps.detach(),
                            "noise_list": {k: v.detach() for k, v in noise_list_tmp.items()},
                            "scheduler": copy.deepcopy(self.scheduler)
                        }

                    del latents_tmp, latents_at_quick_steps, tweedie_est, noise_list_tmp, lat_d, image_fwd
                    torch.cuda.empty_cache()

            print(f"[SDXL-DINO] Noise candidate scores: {scores}, best_idx: {np.argmax(scores) if scores else 'N/A'}")
            best_noise_idx = np.argmax(scores)
            best_noise = noise_pool[best_noise_idx]
            
            if best_candidate_cache is None:
                print("[SDXL-DINO] WARNING: best_candidate_cache is None, using fallback full denoise")
                lat_std = latents_init.float().std().item()
                latents_tmp = latents_init.detach() + step_lr * lat_std * best_noise
                latents_init = latents_tmp.detach()
                
                with torch.no_grad():
                    self.scheduler = copy.deepcopy(scheduler_snapshot)
                    noise_list = denoise(latents_init)
                    self.scheduler.set_timesteps(num_inference_steps)
            else:
                lat_std = latents_init.float().std().item()
                latents_tmp = latents_init.detach() + step_lr * lat_std * best_noise
                latents_init = latents_tmp.detach()
                
                with torch.no_grad():
                    self.scheduler = best_candidate_cache["scheduler"]
                    noise_list_best_second_part = denoise(best_candidate_cache["latents"], start_step=QUICK_STEPS, max_steps=num_inference_steps)
                    noise_list = {**best_candidate_cache["noise_list"], **noise_list_best_second_part}
                    self.scheduler.set_timesteps(num_inference_steps)
                
            # Decode the full image for this epoch
            with torch.no_grad():
                latents = reverse(latents_init, noise_list)
                self.scheduler.set_timesteps(num_inference_steps)
                if has_latents_mean and has_latents_std:
                    latents_dec = (latents * latents_std / self.vae.config.scaling_factor) + latents_mean
                else:
                    latents_dec = latents / self.vae.config.scaling_factor
                
                if needs_upcasting:
                    self.upcast_vae()
                    latents_dec = latents_dec.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                
                image = self.vae.decode(latents_dec, return_dict=False)[0]
                
                if needs_upcasting:
                    self.vae.to(dtype=torch.float16)
                
                image = image.float()
            
            # --- Evaluate VQA + compute gradient for this epoch ---
            gradients_cpu_epoch = []
            individual_vqa_scores_epoch = []
            
            if use_ldino and self.ldino_optimizer and hasattr(self.ldino_optimizer, 'segmenter'):
                with torch.no_grad():
                    image_pil_epoch = self.image_processor.postprocess(image.detach().cpu(), output_type="pil")[0]
                    current_masks = self.ldino_optimizer.segmenter.segment_multiple(image_pil_epoch, [e.name for e in entities])
                    
                    if save_debug:
                        mask_dir = f"{dir}/ldino_debug/epoch_{i}"
                        os.makedirs(mask_dir, exist_ok=True)
                        for e_name, m_np in current_masks.items():
                            if m_np is not None:
                                Image.fromarray((m_np * 255).astype(np.uint8)).save(f"{mask_dir}/mask_{e_name}.png")
                        if hasattr(self.ldino_optimizer.segmenter, 'visualize_masks') and current_masks:
                            combined_np = self.ldino_optimizer.segmenter.visualize_masks(image_pil_epoch, current_masks)
                            Image.fromarray(combined_np).save(f"{mask_dir}/combined_masks.png")
            else:
                current_masks = getattr(self, "initial_masks", {})

            for n_id, p in graph_evaluator.questions.items():
                # Gradient computation: reverse(latents_init) -> float32 VAE decode -> VQA backward
                latents_init_f32 = latents_init.float().detach().requires_grad_(True)
                
                if needs_upcasting:
                    self.upcast_vae()
                
                lat_ep = reverse(latents_init_f32, noise_list)
                self.scheduler.set_timesteps(num_inference_steps)
                if has_latents_mean and has_latents_std:
                    lat_d_ep = (lat_ep * latents_std.float() / self.vae.config.scaling_factor) + latents_mean.float()
                else:
                    lat_d_ep = lat_ep / self.vae.config.scaling_factor
                
                img_ep = self.vae.decode(lat_d_ep.to(next(iter(self.vae.post_quant_conv.parameters())).dtype), return_dict=False)[0]
                
                image_scaled = (img_ep.float() / 2 + 0.5).clamp(0, 1)
                # CPU round-trip to fix PyTorch cross-GPU copy bug
                # CUDA sync to prevent deadlock before CPU copy
                torch.cuda.synchronize()
                image_cuda1 = image_scaled.detach().cpu().to(self.vqa_model_device).requires_grad_(True)
                target_image = image_cuda1
                matched_entity = None
                
                if use_ldino:
                    concept = graph_evaluator.nodes[n_id].get("concept", p)
                    for ent in entities:
                        if ent.name in concept: matched_entity = ent.name; break
                    if matched_entity and matched_entity in current_masks:
                        mask_np = current_masks[matched_entity]
                        if mask_np is not None:
                            target_image = apply_blur_mask(image_cuda1, torch.from_numpy(mask_np).to(self.vqa_model_device).float().unsqueeze(0).unsqueeze(0), blur_radius=21)
                
                with torch.cuda.device(self.vqa_model_device):
                    score = self.vqa_model(target_image, [p])
                    s_val = score.item()
                    individual_vqa_scores_epoch.append((n_id, s_val))
                score.backward()
                
                grad_on_cuda0 = image_cuda1.grad.to(device=img_ep.device, dtype=torch.float32) if image_cuda1.grad is not None else torch.zeros_like(img_ep).float()
                grad_on_cuda0 = torch.nan_to_num(grad_on_cuda0, nan=0.0, posinf=0.0, neginf=0.0)
                img_ep.backward(grad_on_cuda0 / 2.0)
                
                if latents_init_f32.grad is not None:
                    grad_cpu = latents_init_f32.grad.detach().cpu().clone()
                    grad_cpu = torch.nan_to_num(grad_cpu, nan=0.0, posinf=0.0, neginf=0.0)
                    if use_ldino and matched_entity and matched_entity in current_masks:
                        mask_np_loc = current_masks[matched_entity]
                        if mask_np_loc is not None:
                            latent_h, latent_w = grad_cpu.shape[2], grad_cpu.shape[3]
                            mask_latent = np.array(Image.fromarray((mask_np_loc * 255).astype(np.uint8)).resize((latent_w, latent_h), Image.BILINEAR)) / 255.0
                            grad_cpu = grad_cpu * torch.from_numpy(mask_latent).float().unsqueeze(0).unsqueeze(0)
                    gradients_cpu_epoch.append((n_id, grad_cpu))
                else: 
                    gradients_cpu_epoch.append((n_id, torch.zeros_like(latents_init.detach().cpu())))
                
                del score, latents_init_f32, img_ep
                gc.collect(); torch.cuda.empty_cache()
                if needs_upcasting:
                    self.vae.to(dtype=torch.float16)
            
            raw_scores_map_epoch = {n_id: s for n_id, s in individual_vqa_scores_epoch}
            masked_scores_epoch, root_scores_epoch, avg_score_epoch = graph_evaluator.evaluate(raw_scores_map_epoch)
            raw_avg_ep = sum(v for _, v in individual_vqa_scores_epoch) / max(len(individual_vqa_scores_epoch), 1)
            vqa_score_val = max(raw_avg_ep, 1e-6)
            
            root_gradients_epoch = []
            for root_id, tree_score in root_scores_epoch.items():
                tree_nodes = set()
                def dfs(nid):
                    if nid in tree_nodes: return
                    tree_nodes.add(nid)
                    for child_id, child in graph_evaluator.nodes.items():
                        parents = child.get("parent_id")
                        if (isinstance(parents, list) and nid in parents) or parents == nid: dfs(child_id)
                dfs(root_id)
                tree_grad = torch.zeros_like(gradients_cpu_epoch[0][1])
                for n_id, g in gradients_cpu_epoch:
                    if n_id in tree_nodes:
                        coeff = 1.0
                        for on_id in tree_nodes:
                            if on_id != n_id: coeff *= max(raw_scores_map_epoch.get(on_id, 1e-9), 1e-9)
                        tree_grad += coeff * g
                root_gradients_epoch.append(tree_grad)
            
            avg_gradient_epoch = torch.mean(torch.stack(root_gradients_epoch), dim=0) if len(root_gradients_epoch) > 0 else gradients_cpu_epoch[0][1]
            stored_grad = avg_gradient_epoch.to(latents_init.device)
            
            if vqa_score_val > max_vqa:
                max_vqa = vqa_score_val
                target = image.detach().clone()
                
            print(f"  [Epoch {i+1}] VQA={vqa_score_val:.4f} (best={max_vqa:.4f}), lr={step_lr:.4f}")
            
            ep_pil = self.image_processor.postprocess(image.detach().cpu(), output_type="pil")[0]
            ep_pil.save(f'{dir}/{i + 1}.png')
            del noise_pool, scores; gc.collect(); torch.cuda.empty_cache()

        print(f"\n[SDXL-DINO] Optimization complete! Best VQA: {max_vqa:.4f}")
        
        target = self.image_processor.postprocess(target.detach().cpu(), output_type=output_type)[0]
        target.save(f'{dir}/target.png')

        self.maybe_free_model_hooks()
        if not return_dict: return (target,)
        return StableDiffusionXLPipelineOutput(images=[target])
