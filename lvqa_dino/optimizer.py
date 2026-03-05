"""
L-DINO Optimizer - Complete Implementation

Implements the full L-DINO-CoT algorithm with:
- Grounded-SAM for high-quality text-guided segmentation
- Reflection VQA scoring (target attributes)
- Visualization saving for debugging
- Localized gradient computation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
import os

from .prompt_decomposer import EntityInfo, StructuredCoTDecomposer
from .vqa_scorer import LocalizedVQAScorer, FallbackVQAScorer
from .daam_attention import DAAMExtractor, SimpleAttentionLocalizer


class LDINOOptimizer:
    """
    Complete L-DINO-CoT optimizer implementing Algorithm 5.1 from the paper.
    
    Combines:
    1. Structured CoT decomposition with Reflection questions
    2. Hybrid localization (DAAM early -> CLIPSeg/SAM later)
    3. Dual-branch VQA scoring per entity
    4. Directional Gaussian noise updates
    5. Visualization saving for analysis
    """
    
    def __init__(
        self,
        vqa_model=None,
        device: str = "cuda",
        warmup_ratio: float = 0.2,
        lambda_ref: float = 1.0,
        use_grounded_sam: bool = False,
        use_blip2: bool = False,
        mask_cache_interval: int = 3,
        save_visualizations: bool = True
    ):
        """
        Initialize the L-DINO optimizer.
        
        Args:
            vqa_model: Existing VQA model from t2v_metrics (fallback)
            device: Device for computation
            warmup_ratio: Fraction of steps to use soft (DAAM) masks
            lambda_ref: Weight for reflection loss (target attributes)
            use_grounded_sam: Whether to use Grounded-SAM for segmentation
            use_blip2: Whether to use BLIP-2 (vs existing VQA model)
            mask_cache_interval: How often to recompute masks
            save_visualizations: Whether to save intermediate images
        """
        self.device = device
        self.warmup_ratio = warmup_ratio
        self.lambda_ref = lambda_ref
        self.mask_cache_interval = mask_cache_interval
        self.save_visualizations = save_visualizations
        
        # Initialize components
        self.decomposer = StructuredCoTDecomposer()
        
        # Always try Grounded-SAM first (best quality masks)
        from .segmentation import GroundedSAMSegmenter, SimpleMaskGenerator, GROUNDED_SAM_AVAILABLE
        
        self.segmenter = None
        if GROUNDED_SAM_AVAILABLE:
            try:
                self.segmenter = GroundedSAMSegmenter(device=device)
                print("[L-DINO] Using Grounded-SAM for segmentation ✓")
            except Exception as e:
                print(f"[L-DINO] Grounded-SAM init error: {e}")
        
        if self.segmenter is None:
            print("[L-DINO] Grounded-SAM not available, using fallback masks")
            print("[L-DINO] Install with: pip install autodistill-grounded-sam")
        
        self.simple_mask_gen = SimpleMaskGenerator(device=device)
        self.simple_attention = SimpleAttentionLocalizer(device=device)
        
        if use_blip2:
            self.vqa_scorer = LocalizedVQAScorer(device=device)
        elif vqa_model is not None:
            self.vqa_scorer = FallbackVQAScorer(vqa_model, device=device)
        else:
            self.vqa_scorer = None
        
        self.vqa_model = vqa_model
        
        # Cache for masks
        self._mask_cache: Dict[str, np.ndarray] = {}
        self._last_cache_step = -1
        
        # Current state
        self.entities = []
        self.prompt = ""
        self.output_dir = ""
    
    def setup_entities(
        self,
        prompt: str,
        entity_attributes: Dict[str, List[str]],
        output_dir: str = ""
    ) -> List[EntityInfo]:
        """
        Set up entities from hardcoded input.
        
        Args:
            prompt: Full prompt string
            entity_attributes: Dict mapping entity names to attributes
            output_dir: Directory for saving visualizations
        
        Returns:
            List of EntityInfo objects with questions
        """
        entities_dict = [
            {"name": name, "attributes": attrs}
            for name, attrs in entity_attributes.items()
        ]
        self.entities = self.decomposer.decompose_from_dict(entities_dict)
        self.prompt = prompt
        self.output_dir = output_dir
        
        # Create visualization subdirectory
        if self.save_visualizations and output_dir:
            os.makedirs(os.path.join(output_dir, "ldino_debug"), exist_ok=True)
        
        return self.entities
    
    def compute_localized_loss(
        self,
        image: torch.Tensor,
        image_processor,
        step: int,
        total_steps: int,
        entities: Optional[List[EntityInfo]] = None,
        save_prefix: str = "",
        all_masks: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        Compute total localized loss across all entities with full L-DINO-CoT.
        
        Uses soft masks during warmup, hard masks afterwards.
        Saves visualizations if enabled.
        
        Args:
            image: Generated image tensor (B, C, H, W)
            image_processor: Pipeline's image processor for conversion
            step: Current optimization step
            total_steps: Total optimization steps
            entities: Entity list (uses self.entities if not provided)
            save_prefix: Prefix for saved visualization files
            all_masks: Optional pre-computed masks to use instead of segmenting
            
        Returns:
            Total loss tensor, list of per-entity info dicts
        """
        if entities is None:
            entities = self.entities
        
        if not entities:
            return torch.tensor(0.0, device=self.device, requires_grad=True), []
        
        # Determine mask type based on step (Algorithm 5.1 Step 3.2)
        use_hard_masks = step >= int(total_steps * self.warmup_ratio)
        should_update_cache = (
            step - self._last_cache_step >= self.mask_cache_interval or
            step == 0
        )
        
        # Convert image tensor to PIL for segmentation
        image_pil = self._tensor_to_pil(image, image_processor)
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        all_info = []
        
        # Get masks for each entity (Step 3.2)
        if all_masks is not None:
            masks = all_masks
        else:
            masks = self._get_masks(
                image, image_pil, entities,
                use_hard_masks, should_update_cache, step
            )
        
        # Save mask visualization
        if self.save_visualizations and self.output_dir and save_prefix:
            self._save_mask_visualization(
                image_pil, masks, entities,
                os.path.join(self.output_dir, "ldino_debug"),
                f"{save_prefix}_step{step}"
            )
        
        # Compute per-entity loss (Step 3.3)
        for entity in entities:
            mask = masks.get(entity.name)
            if mask is None:
                continue
            
            # Convert mask for cropping
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze().cpu().numpy()
            else:
                mask_np = mask
            
            # Crop image with blur outside mask (Section 4.1)
            cropped_pil = self._crop_and_blur(image_pil, mask_np)
            
            # Save cropped visualization
            if self.save_visualizations and self.output_dir and save_prefix:
                crop_path = os.path.join(
                    self.output_dir, "ldino_debug",
                    f"{save_prefix}_step{step}_{entity.name}_crop.png"
                )
                cropped_pil.save(crop_path)
            
            # Compute VQA scores (Section 3.3)
            ref_score = self._compute_entity_vqa_scores(
                cropped_pil, entity
            )
            
            # Loss: λ_ref(1 - s_ref) (Equation from Section 3.3)
            entity_loss = self.lambda_ref * (1 - ref_score)
            
            info = {
                'entity': entity.name,
                'reflection_score': ref_score,
                'loss': entity_loss,
                'mask_type': 'hard' if use_hard_masks else 'soft'
            }
            
            total_loss = total_loss + entity_loss
            all_info.append(info)
        
        return total_loss, all_info
    
    def _get_masks(
        self,
        image: torch.Tensor,
        image_pil: Image.Image,
        entities: List[EntityInfo],
        use_hard_masks: bool,
        should_update_cache: bool,
        step: int
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Get masks for all entities using Grounded-SAM or fallback."""
        masks = {}
        H, W = image_pil.size[1], image_pil.size[0]
        
        if use_hard_masks and self.segmenter is not None:
            # Use Grounded-SAM for high-quality masks
            if should_update_cache:
                entity_names = [e.name for e in entities]
                print(f"[L-DINO] Computing Grounded-SAM masks for: {entity_names}")
                
                try:
                    # Use batch segmentation for efficiency
                    if hasattr(self.segmenter, 'segment_multiple'):
                        batch_masks = self.segmenter.segment_multiple(image_pil, entity_names)
                        self._mask_cache = batch_masks
                    else:
                        # Fallback to individual segmentation
                        for entity in entities:
                            mask = self.segmenter.segment(image_pil, entity.name)
                            self._mask_cache[entity.name] = mask
                except Exception as e:
                    print(f"[L-DINO] Segmentation error: {e}")
                    # Fallback to simple masks
                    fallback_masks = self.simple_mask_gen._generate_simple_masks(H, W, entity_names) if hasattr(self.simple_mask_gen, '_generate_simple_masks') else {}
                    for i, entity in enumerate(entities):
                        if entity.name not in fallback_masks:
                            all_masks = self.simple_mask_gen.generate_quadrant_masks(H, W, len(entities))
                            self._mask_cache[entity.name] = all_masks[i] if i < len(all_masks) else self.simple_mask_gen.generate_center_mask(H, W)
                        else:
                            self._mask_cache[entity.name] = fallback_masks[entity.name]
                
                self._last_cache_step = step
            
            masks = self._mask_cache.copy()
        else:
            # Use simple quadrant masks during warmup
            for i, entity in enumerate(entities):
                all_masks = self.simple_mask_gen.generate_quadrant_masks(H, W, len(entities))
                if i < len(all_masks):
                    masks[entity.name] = all_masks[i]
                else:
                    masks[entity.name] = self.simple_mask_gen.generate_center_mask(H, W)
        
        return masks
    
    def _compute_entity_vqa_scores(
        self,
        cropped_image: Image.Image,
        entity: EntityInfo
    ) -> float:
        """
        Compute reflection VQA score for an entity.
        
        Implements Section 3.3 - Reflection VQA Scoring.
        Leakage is now handled via dependency graph nodes in the pipeline.
        """
        if self.vqa_model is None:
            return 0.5
        
        # Convert PIL to tensor (normalized to [0, 1])
        import torchvision.transforms as transforms
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(cropped_image).unsqueeze(0).to(self.device)
        
        # Reflection: average P("Yes") for target attribute questions
        ref_scores = []
        for q in entity.reflection_questions:
            try:
                statement = self._question_to_statement(q, entity.name)
                # VQA model expects tensor input
                score = self.vqa_model(img_tensor, [statement])
                if isinstance(score, torch.Tensor):
                    ref_scores.append(score.item())
                else:
                    ref_scores.append(float(score))
            except Exception as e:
                print(f"VQA error for reflection '{q}': {e}")
                ref_scores.append(0.5)
        
        s_ref = np.mean(ref_scores) if ref_scores else 0.5
        
        return s_ref
    
    def _question_to_statement(self, question: str, entity_name: str) -> str:
        """Convert a yes/no question to a statement for VQA scoring."""
        # "Is the cat fluffy?" -> "a fluffy cat"
        q = question.replace("Is the ", "").replace("?", "").strip()
        q = q.replace(f"{entity_name} ", "")
        if q.startswith("there a "):
            return f"a {entity_name}"
        return f"a {q} {entity_name}"
    
    def _crop_and_blur(
        self,
        image_pil: Image.Image,
        mask: np.ndarray,
        blur_radius: int = 20
    ) -> Image.Image:
        """
        Crop image with Gaussian blur outside mask (Section 4.1).
        Maintains context while focusing on object.
        """
        from PIL import ImageFilter
        
        W, H = image_pil.size
        
        # Resize mask if needed
        if mask.shape != (H, W):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((W, H), Image.NEAREST)
            mask = np.array(mask_img) / 255.0
        
        # Create blurred version
        blurred = image_pil.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Composite: sharp inside mask, blurred outside
        mask_3ch = np.stack([mask] * 3, axis=-1)
        img_np = np.array(image_pil).astype(float)
        blur_np = np.array(blurred).astype(float)
        
        result = img_np * mask_3ch + blur_np * (1 - mask_3ch)
        result = result.clip(0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def _save_mask_visualization(
        self,
        image_pil: Image.Image,
        masks: Dict[str, np.ndarray],
        entities: List[EntityInfo],
        output_dir: str,
        prefix: str
    ):
        """Save visualization of masks overlaid on image."""
        import random
        
        # Create a copy for visualization
        vis_img = image_pil.copy().convert("RGBA")
        W, H = vis_img.size
        
        # Colors for different entities
        colors = [
            (255, 0, 0, 128),    # Red
            (0, 255, 0, 128),    # Green
            (0, 0, 255, 128),    # Blue
            (255, 255, 0, 128),  # Yellow
            (255, 0, 255, 128),  # Magenta
            (0, 255, 255, 128),  # Cyan
        ]
        
        for i, entity in enumerate(entities):
            if entity.name not in masks:
                continue
            
            mask = masks[entity.name]
            if isinstance(mask, torch.Tensor):
                mask = mask.squeeze().cpu().numpy()
            
            # Resize mask if needed
            if mask.shape != (H, W):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize((W, H), Image.NEAREST)
                mask = np.array(mask_img) / 255.0
            
            # Create colored overlay
            color = colors[i % len(colors)]
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            overlay_np = np.array(overlay)
            
            mask_bool = mask > 0.5
            overlay_np[mask_bool] = color
            
            overlay = Image.fromarray(overlay_np)
            vis_img = Image.alpha_composite(vis_img, overlay)
        
        # Save
        vis_path = os.path.join(output_dir, f"{prefix}_masks.png")
        vis_img.convert("RGB").save(vis_path)
    
    def _tensor_to_pil(
        self,
        tensor: torch.Tensor,
        image_processor=None
    ) -> Image.Image:
        """Convert image tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        # Normalize to [0, 1]
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
        
        # Convert to numpy
        np_img = tensor.detach().cpu().permute(1, 2, 0).numpy()
        np_img = (np_img * 255).astype(np.uint8)
        
        return Image.fromarray(np_img)
    
    def clear_cache(self):
        """Clear mask cache."""
        self._mask_cache.clear()
        self._last_cache_step = -1
    
    def print_entity_summary(self, entities: Optional[List[EntityInfo]] = None):
        """Print summary of entities and their questions."""
        if entities is None:
            entities = self.entities
        
        print("\n" + "="*60)
        print("L-DINO-CoT Entity Summary")
        print("="*60)
        
        for entity in entities:
            print(f"\n[Entity: {entity.name}]")
            print(f"  Target attributes: {entity.target_attributes}")
            print(f"  Distractor attributes: {entity.distractor_attributes}")
            print(f"  Reflection questions ({len(entity.reflection_questions)}):")
            for q in entity.reflection_questions[:3]:
                print(f"    - {q}")
            if len(entity.reflection_questions) > 3:
                print(f"    ... and {len(entity.reflection_questions) - 3} more")

        
        print("\n" + "="*60 + "\n")

    def print_entity_scores(self, entity_info: List[dict], step: int):
        """Print L-VQA scores for all entities at a specific step."""
        print(f"\n[L-DINO-CoT] Step {step} entity scores:")
        for info in entity_info:
            entity_name = info.get('entity', 'Unknown')
            ref = info.get('reflection_score', 0)
            # Handle both scalar and list scores
            ref_val = ref.mean().item() if hasattr(ref, 'mean') else ref
            print(f"  {entity_name}: ref={ref_val:.3f}")
