"""
L-VQA Scoring Module for DINO-CoT

Implements Localized VQA scoring (Section 4.3) with:
- Reflection questions: Verify target attributes are present
- Leakage questions: Verify distractor attributes are absent
- Crop-and-blur localization for accurate scoring
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageFilter


class LVQAScorer:
    """
    Localized VQA Scorer for entity-level evaluation.
    
    Computes:
    - s_refl: P("Yes" | crop_i, "Is the [entity] [target_attr]?")
    - s_leak: P("Yes" | crop_i, "Is the [entity] [other_attr]?") - want LOW
    """
    
    def __init__(
        self,
        vqa_model,
        device: str = "cuda",
        lambda_ref: float = 1.0,
        lambda_leak: float = 1.5
    ):
        self.vqa_model = vqa_model
        self.device = device
        self.lambda_ref = lambda_ref
        self.lambda_leak = lambda_leak
    
    def compute_entity_score(
        self,
        cropped_image: Image.Image,
        entity_name: str,
        target_attributes: List[str],
        distractor_attributes: List[str]
    ) -> Tuple[float, float, float]:
        """
        Compute L-VQA score for a single entity.
        
        Args:
            cropped_image: Entity crop (blur-background)
            entity_name: Name of entity
            target_attributes: Attributes that SHOULD be present
            distractor_attributes: Attributes that should NOT be present
            
        Returns:
            (reflection_score, leakage_score, total_loss)
        """
        import torchvision.transforms as transforms
        to_tensor = transforms.ToTensor()
        
        ref_scores = []
        leak_scores = []
        
        try:
            img_tensor = to_tensor(cropped_image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"[L-VQA] Tensor error: {e}")
            return 0.5, 0.0, 0.5
        
        # Reflection: target attributes (want HIGH)
        for attr in target_attributes:
            try:
                statement = f"a {attr} {entity_name}"
                score = self.vqa_model(img_tensor, [statement])
                if isinstance(score, torch.Tensor):
                    ref_scores.append(score.mean().item())
                else:
                    ref_scores.append(float(score))
            except Exception as e:
                print(f"[L-VQA] Reflection error: {e}")
                ref_scores.append(0.5)
            
            # Memory cleanup
            torch.cuda.empty_cache()
        
        # Leakage: distractor attributes (want LOW)
        for attr in distractor_attributes:
            try:
                statement = f"a {attr} {entity_name}"
                score = self.vqa_model(img_tensor, [statement])
                if isinstance(score, torch.Tensor):
                    leak_scores.append(score.mean().item())
                else:
                    leak_scores.append(float(score))
            except Exception as e:
                print(f"[L-VQA] Leakage error: {e}")
                leak_scores.append(0.0)
            
            torch.cuda.empty_cache()
        
        # Aggregate
        ref_avg = np.mean(ref_scores) if ref_scores else 0.5
        leak_avg = np.mean(leak_scores) if leak_scores else 0.0
        
        # Loss: λ_r*(1 - ref) + λ_l*leak
        loss = self.lambda_ref * (1 - ref_avg) + self.lambda_leak * leak_avg
        
        return ref_avg, leak_avg, loss
    
    def crop_and_blur(
        self,
        image: Image.Image,
        mask: np.ndarray,
        blur_radius: int = 20,
        min_crop_ratio: float = 0.1
    ) -> Image.Image:
        """
        Crop image with blurred background (Section 4.2.2).
        
        x_i = Crop(x * M + Blur(x) * (1-M))
        
        Args:
            image: Full image
            mask: Binary mask (H, W) with 1 = entity, 0 = background
            blur_radius: Gaussian blur radius for background
            min_crop_ratio: Minimum crop size as ratio of image
        """
        W, H = image.size
        
        # Ensure mask is correct shape
        if mask.shape != (H, W):
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((W, H), PILImage.NEAREST)
            mask = np.array(mask_pil) / 255.0
        
        # Create blurred background
        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Composite: foreground where mask=1, blurred where mask=0
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        
        img_np = np.array(image).astype(np.float32)
        blur_np = np.array(blurred).astype(np.float32)
        
        composite = img_np * mask_3ch + blur_np * (1 - mask_3ch)
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        
        result = Image.fromarray(composite)
        
        # Find bounding box of mask for cropping
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # Empty mask, return center crop
            margin = int(min(W, H) * 0.25)
            return result.crop((margin, margin, W-margin, H-margin))
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Add margin
        margin = int(max((y_max - y_min), (x_max - x_min)) * 0.15)
        y_min = max(0, y_min - margin)
        y_max = min(H, y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(W, x_max + margin)
        
        # Ensure minimum size
        min_size = int(min(W, H) * min_crop_ratio)
        if (x_max - x_min) < min_size:
            center_x = (x_min + x_max) // 2
            x_min = max(0, center_x - min_size // 2)
            x_max = min(W, center_x + min_size // 2)
        if (y_max - y_min) < min_size:
            center_y = (y_min + y_max) // 2
            y_min = max(0, center_y - min_size // 2)
            y_max = min(H, center_y + min_size // 2)
        
        return result.crop((x_min, y_min, x_max, y_max))


class SimpleSegmenter:
    """
    Simple fallback segmenter using quadrant masks.
    Used when Grounded-SAM is not available or OOM.
    """
    
    def segment_multiple(
        self,
        image: Image.Image,
        entities: List[str]
    ) -> Dict[str, np.ndarray]:
        """Generate simple masks for entities."""
        W, H = image.size
        masks = {}
        n = len(entities)
        
        if n == 1:
            # Center mask
            mask = np.zeros((H, W), dtype=np.float32)
            h_margin, w_margin = H // 4, W // 4
            mask[h_margin:H-h_margin, w_margin:W-w_margin] = 1.0
            masks[entities[0]] = mask
        elif n == 2:
            # Left/right split
            mask_left = np.zeros((H, W), dtype=np.float32)
            mask_left[:, :W//2] = 1.0
            mask_right = np.zeros((H, W), dtype=np.float32)
            mask_right[:, W//2:] = 1.0
            masks[entities[0]] = mask_left
            masks[entities[1]] = mask_right
        else:
            # Quadrant split
            for i, entity in enumerate(entities):
                mask = np.zeros((H, W), dtype=np.float32)
                row, col = i // 2, i % 2
                h_start = (H * row) // 2
                h_end = (H * (row + 1)) // 2
                w_start = (W * col) // 2
                w_end = (W * (col + 1)) // 2
                mask[h_start:h_end, w_start:w_end] = 1.0
                masks[entity] = mask
        
        return masks


def compute_lvqa_loss(
    image: Image.Image,
    entities: List,
    vqa_model,
    segmenter=None,
    device: str = "cuda",
    lambda_ref: float = 1.0,
    lambda_leak: float = 1.5,
    verbose: bool = True
) -> Tuple[float, Dict]:
    """
    Compute full L-VQA loss for all entities.
    
    Args:
        image: Generated image
        entities: List of EntityInfo with name, target_attributes, distractor_attributes
        vqa_model: VQA model for scoring
        segmenter: Segmenter (uses SimpleSegmenter if None)
        
    Returns:
        (total_loss, info_dict)
    """
    import gc
    
    if segmenter is None:
        segmenter = SimpleSegmenter()
    
    scorer = LVQAScorer(vqa_model, device, lambda_ref, lambda_leak)
    
    # Get masks for all entities
    entity_names = [e.name for e in entities]
    
    try:
        masks = segmenter.segment_multiple(image, entity_names)
    except Exception as e:
        if verbose:
            print(f"[L-VQA] Segmentation failed: {e}, using fallback")
        masks = SimpleSegmenter().segment_multiple(image, entity_names)
    
    # Memory cleanup after segmentation
    gc.collect()
    torch.cuda.empty_cache()
    
    total_loss = 0.0
    info = {"entities": {}, "total_loss": 0.0}
    
    for entity in entities:
        mask = masks.get(entity.name)
        if mask is None:
            continue
        
        # Crop with blur background
        crop = scorer.crop_and_blur(image, mask)
        
        # Compute scores
        ref_score, leak_score, entity_loss = scorer.compute_entity_score(
            crop,
            entity.name,
            entity.target_attributes,
            entity.distractor_attributes
        )
        
        total_loss += entity_loss
        info["entities"][entity.name] = {
            "reflection": ref_score,
            "leakage": leak_score,
            "loss": entity_loss
        }
        
        if verbose:
            print(f"  [{entity.name}] ref={ref_score:.3f}, leak={leak_score:.3f}")
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    info["total_loss"] = total_loss
    return total_loss, info


# Export
__all__ = ["LVQAScorer", "SimpleSegmenter", "compute_lvqa_loss"]
