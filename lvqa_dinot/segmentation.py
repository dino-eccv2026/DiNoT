"""
Segmentation module for L-DINO-CoT using autodistill Grounded-SAM

Uses GroundedSAM from autodistill for text-guided segmentation.
This provides high-quality masks for localized VQA scoring.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import os
import tempfile
import gc

# Try to import autodistill Grounded-SAM
GROUNDED_SAM_AVAILABLE = False
try:
    from autodistill_grounded_sam import GroundedSAM
    from autodistill.detection import CaptionOntology
    import supervision as sv
    GROUNDED_SAM_AVAILABLE = True
    print("[Segmentation] autodistill Grounded-SAM available!")
except ImportError as e:
    print(f"[Segmentation] autodistill Grounded-SAM not available: {e}")
    print("Install with: pip install autodistill-grounded-sam")


class GroundedSAMSegmenter:
    """
    Text-guided segmentation using Grounded-SAM via autodistill.
    Provides pixel-perfect masks for accurate localized VQA scoring.
    
    Features memory-efficient loading and better fallback handling.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._loaded = False
        self._temp_dir = tempfile.mkdtemp()
        self._current_entities = []
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_model(self, entities: List[str]):
        """
        Load Grounded-SAM with ontology for the given entities.
        Includes memory cleanup before loading.
        """
        if not GROUNDED_SAM_AVAILABLE:
            raise ImportError("autodistill-grounded-sam not installed!")
        
        ontology_dict = {entity: entity for entity in entities}
        
        if self.model is None:
            print(f"[Grounded-SAM] Loading model for entities: {entities}")
            try:
                self.model = GroundedSAM(
                    ontology=CaptionOntology(ontology_dict)
                )
                self._loaded = True
                self._current_entities = entities
                print("[Grounded-SAM] Model loaded successfully!")
            except Exception as e:
                print(f"[Grounded-SAM] Model load error: {e}")
                self._cleanup_memory()
                raise
        else:
            self.model.ontology = CaptionOntology(ontology_dict)
            self._current_entities = entities
    
    def segment(
        self,
        image: Union[Image.Image, np.ndarray, str],
        text_prompt: str,
        threshold: float = 0.3
    ) -> np.ndarray:
        """
        Segment the image based on a single text prompt.
        
        Args:
            image: PIL Image, numpy array, or path to image
            text_prompt: Text describing the object to segment
            threshold: Confidence threshold (used internally by Grounded-SAM)
            
        Returns:
            Binary mask as numpy array (H, W) with values in {0, 1}
        """
        # Load model if not loaded or entities changed
        if not self._loaded or text_prompt not in self._current_entities:
            try:
                self._load_model([text_prompt])
            except Exception as e:
                print(f"[Grounded-SAM] Could not load model: {e}")
                return self._generate_attention_based_mask(image, text_prompt)
        
        # Get image dimensions
        if isinstance(image, Image.Image):
            W, H = image.size
            temp_path = os.path.join(self._temp_dir, "temp_image.jpg")
            image.save(temp_path)
            image_path = temp_path
        elif isinstance(image, np.ndarray):
            H, W = image.shape[:2]
            temp_path = os.path.join(self._temp_dir, "temp_image.jpg")
            Image.fromarray(image).save(temp_path)
            image_path = temp_path
        else:
            image_path = image
            img = Image.open(image_path)
            W, H = img.size
        
        # Run prediction
        try:
            self._cleanup_memory()
            results = self.model.predict(image_path)
            
            # Extract mask for the requested entity
            if results.mask is not None and len(results.mask) > 0:
                # Find the mask for our text_prompt
                prompts = self.model.ontology.prompts()
                try:
                    entity_idx = prompts.index(text_prompt)
                    matching_indices = np.where(results.class_id == entity_idx)[0]
                    
                    if len(matching_indices) > 0:
                        # Use the first matching mask (highest confidence)
                        mask = results.mask[matching_indices[0]]
                        self._cleanup_memory()
                        return mask.astype(np.float32)
                except ValueError:
                    pass
                
                # Fallback: use the first mask regardless of class
                if len(results.mask) > 0:
                    mask = results.mask[0]
                    self._cleanup_memory()
                    return mask.astype(np.float32)
            
            # No mask found
            print(f"[Grounded-SAM] No mask found for '{text_prompt}'")
            return self._generate_attention_based_mask(image, text_prompt)
            
        except Exception as e:
            print(f"[Grounded-SAM] Prediction error: {e}")
            self._cleanup_memory()
            return self._generate_attention_based_mask(image, text_prompt)
    
    def segment_multiple(
        self,
        image: Union[Image.Image, np.ndarray, str],
        entities: List[str],
        threshold: float = 0.3
    ) -> Dict[str, np.ndarray]:
        """
        Segment multiple objects in the image at once.
        More efficient than calling segment() multiple times.
        
        For memory efficiency, tries segmentation one entity at a time
        if batch fails.
        """
        # Get image dimensions
        if isinstance(image, Image.Image):
            W, H = image.size
        elif isinstance(image, np.ndarray):
            H, W = image.shape[:2]
        else:
            img = Image.open(image)
            W, H = img.size
        
        if not GROUNDED_SAM_AVAILABLE:
            print("[Grounded-SAM] Not available, using attention-based masks")
            return self._generate_attention_masks(image, entities, H, W)
        
        masks = {}
        
        # First try batch segmentation
        try:
            self._cleanup_memory()
            self._load_model(entities)
            
            # Save image to temp file
            if isinstance(image, Image.Image):
                temp_path = os.path.join(self._temp_dir, "temp_image.jpg")
                image.save(temp_path)
                image_path = temp_path
            elif isinstance(image, np.ndarray):
                temp_path = os.path.join(self._temp_dir, "temp_image.jpg")
                Image.fromarray(image).save(temp_path)
                image_path = temp_path
            else:
                image_path = image
            results = self.model.predict(image_path)
            prompts = self.model.ontology.prompts()
            
            if results.mask is not None and len(results.mask) > 0:
                # Temporary storage: entity -> (mask, confidence)
                masks_with_conf = {}
                
                for entity in entities:
                    try:
                        # Autodistill's predict() builds class_ids based on prompts() iteration order,
                        # not classes() order. So we must index against prompts() or the corresponding prompt.
                        # Since ontology sets {entity: entity}, prompt == class == entity.
                        # Just testing index against prompts() for safety
                        entity_idx = prompts.index(entity)
                        matching_indices = np.where(results.class_id == entity_idx)[0]
                        
                        if len(matching_indices) > 0:
                            # Combine all masks for this entity
                            combined_mask = np.zeros((H, W), dtype=np.float32)
                            max_confidence = 0.0
                            for idx in matching_indices:
                                mask = results.mask[idx]
                                # Get confidence for this detection
                                if results.confidence is not None and len(results.confidence) > idx:
                                    conf = results.confidence[idx]
                                    max_confidence = max(max_confidence, conf)
                                if mask.shape == (H, W):
                                    combined_mask = np.maximum(combined_mask, mask.astype(np.float32))
                                else:
                                    # Resize if needed
                                    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                                    mask_pil = mask_pil.resize((W, H), Image.NEAREST)
                                    combined_mask = np.maximum(combined_mask, np.array(mask_pil) / 255.0)
                            
                            masks_with_conf[entity] = (combined_mask, max_confidence)
                            print(f"[Grounded-SAM] Found mask for '{entity}' (confidence: {max_confidence:.3f}, area: {combined_mask.mean()*100:.1f}%)")
                        else:
                            print(f"[Grounded-SAM] No detection for '{entity}' (confidence: 0.000)")
                    except ValueError:
                        pass
                
                # Detect and resolve duplicate/overlapping masks using IoU
                IOU_THRESHOLD = 0.7  # Masks with IoU > 0.7 are considered duplicates
                entities_to_fallback = set()
                
                entity_list = list(masks_with_conf.keys())
                for i in range(len(entity_list)):
                    for j in range(i + 1, len(entity_list)):
                        e1, e2 = entity_list[i], entity_list[j]
                        if e1 in entities_to_fallback or e2 in entities_to_fallback:
                            continue
                        
                        mask1, conf1 = masks_with_conf[e1]
                        mask2, conf2 = masks_with_conf[e2]
                        
                        # Compute IoU and containment
                        mask1_area = (mask1 > 0.5).sum()
                        mask2_area = (mask2 > 0.5).sum()
                        intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5).sum()
                        union = np.logical_or(mask1 > 0.5, mask2 > 0.5).sum()
                        iou = intersection / (union + 1e-8)
                        
                        # OVERLAP LOGIC
                        COVERAGE_THRESHOLD = 0.8
                        if iou > 0.4: # significant overlap
                            # choose one with high probability
                            if conf1 >= conf2:
                                high_e, high_mask, high_conf, high_area = e1, mask1, conf1, mask1_area
                                low_e, low_mask, low_conf, low_area = e2, mask2, conf2, mask2_area
                            else:
                                high_e, high_mask, high_conf, high_area = e2, mask2, conf2, mask2_area
                                low_e, low_mask, low_conf, low_area = e1, mask1, conf1, mask1_area
                                
                            if iou > 0.9:
                                # They are nearly perfectly identical. Cannot subtract without zeroing out mask.
                                print(f"[Grounded-SAM] Nearly identical masks for '{high_e}' and '{low_e}' (IoU: {iou:.2f}). Choosing higher-prob '{high_e}'.")
                                entities_to_fallback.add(low_e)
                            elif intersection / (low_area + 1e-8) > COVERAGE_THRESHOLD:
                                # then check if 1 mask is inside other and based on that decrease area of other mask
                                print(f"[Grounded-SAM] '{low_e}' is heavily inside higher-prob '{high_e}'. Subtracting '{low_e}' from '{high_e}'.")
                                masks_with_conf[high_e] = (np.clip(high_mask - low_mask, 0.0, 1.0), high_conf)
                            elif intersection / (high_area + 1e-8) > COVERAGE_THRESHOLD:
                                print(f"[Grounded-SAM] Higher-prob '{high_e}' is heavily inside '{low_e}'. Subtracting '{high_e}' from '{low_e}'.")
                                masks_with_conf[low_e] = (np.clip(low_mask - high_mask, 0.0, 1.0), low_conf)
                            elif iou > 0.7:
                                print(f"[Grounded-SAM] Large conflict but neither is completely contained. Dropping lower-prob '{low_e}'.")
                                entities_to_fallback.add(low_e)
                
                # Build final masks dict, excluding duplicates
                for entity, (mask, conf) in masks_with_conf.items():
                    if entity not in entities_to_fallback:
                        masks[entity] = mask
                
                self._cleanup_memory()
                
                # Fill in missing entities and duplicates with attention-based masks
                missing = [e for e in entities if e not in masks]
                if missing:
                    print(f"[Grounded-SAM] Using fallback for: {missing}")
                    fallback = self._generate_attention_masks(image, missing, H, W)
                    masks.update(fallback)
                
                return masks
                    
        except Exception as e:
            print(f"[Grounded-SAM] Batch error: {e}")
            self._cleanup_memory()
        
        # Batch failed, try individual segmentation
        print("[Grounded-SAM] Trying individual segmentation...")
        for entity in entities:
            if entity not in masks:
                try:
                    self._cleanup_memory()
                    mask = self.segment(image, entity, threshold)
                    masks[entity] = mask
                except Exception as e:
                    print(f"[Grounded-SAM] Individual segment failed for '{entity}': {e}")
        
        # Fill remaining with attention-based masks
        missing = [e for e in entities if e not in masks]
        if missing:
            fallback = self._generate_attention_masks(image, missing, H, W)
            masks.update(fallback)
        
        return masks
    
    def _generate_attention_based_mask(
        self,
        image: Union[Image.Image, np.ndarray, str],
        entity: str
    ) -> np.ndarray:
        """
        Generate a better fallback mask based on image saliency/center-prior.
        Uses Gaussian blob at center with size based on entity type.
        """
        if isinstance(image, Image.Image):
            W, H = image.size
        elif isinstance(image, np.ndarray):
            H, W = image.shape[:2]
        else:
            img = Image.open(image)
            W, H = img.size
        
        # Create smooth Gaussian blob at center
        y, x = np.ogrid[:H, :W]
        center_y, center_x = H // 2, W // 2
        
        # Vary size based on entity (smaller objects get smaller masks)
        small_objects = ['fly', 'pen', 'ball', 'hat', 'cap', 'key', 'ring']
        large_objects = ['elephant', 'car', 'house', 'room', 'bathroom', 'sofa']
        
        if any(obj in entity.lower() for obj in small_objects):
            sigma_y, sigma_x = H * 0.15, W * 0.15
        elif any(obj in entity.lower() for obj in large_objects):
            sigma_y, sigma_x = H * 0.4, W * 0.4
        else:
            sigma_y, sigma_x = H * 0.25, W * 0.25
        
        # Gaussian blob
        mask = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + 
                        (y - center_y)**2 / (2 * sigma_y**2)))
        
        # Threshold to get binary mask
        mask = (mask > 0.3).astype(np.float32)
        
        print(f"[Fallback] Generated center-prior mask for '{entity}' (area: {mask.mean()*100:.1f}%)")
        return mask
    
    def _generate_attention_masks(
        self, 
        image: Union[Image.Image, np.ndarray, str],
        entities: List[str],
        H: int, W: int
    ) -> Dict[str, np.ndarray]:
        """
        Generate attention-based masks for multiple entities.
        Uses spatial distribution to avoid overlap.
        """
        masks = {}
        n = len(entities)
        
        if n == 0:
            return masks
        
        if n == 1:
            masks[entities[0]] = self._generate_attention_based_mask(image, entities[0])
        elif n == 2:
            # Distribute left and right but with Gaussian blobs, not hard splits
            for i, entity in enumerate(entities):
                y, x = np.ogrid[:H, :W]
                
                if i == 0:
                    center_x = W // 3
                else:
                    center_x = 2 * W // 3
                center_y = H // 2
                
                sigma_y, sigma_x = H * 0.3, W * 0.25
                mask = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + 
                                (y - center_y)**2 / (2 * sigma_y**2)))
                mask = (mask > 0.3).astype(np.float32)
                masks[entity] = mask
                print(f"[Fallback] Generated distributed mask for '{entity}' (area: {mask.mean()*100:.1f}%)")
        else:
            # Grid distribution
            rows = 1 if n <= 2 else 2
            cols = (n + rows - 1) // rows
            
            for i, entity in enumerate(entities):
                row = i // cols
                col = i % cols
                
                center_x = int((col + 0.5) * W / cols)
                center_y = int((row + 0.5) * H / rows)
                
                y, x = np.ogrid[:H, :W]
                sigma_y = H / (2 * rows) * 0.6
                sigma_x = W / (2 * cols) * 0.6
                
                mask = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + 
                                (y - center_y)**2 / (2 * sigma_y**2)))
                mask = (mask > 0.3).astype(np.float32)
                masks[entity] = mask
                print(f"[Fallback] Generated grid mask for '{entity}' at ({row},{col}) (area: {mask.mean()*100:.1f}%)")
        
        return masks
    
    def visualize_masks(
        self,
        image: Union[Image.Image, np.ndarray],
        masks: Dict[str, np.ndarray],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize masks overlaid on image with different colors.
        """
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image.copy()
        
        # Colors for different entities (RGB)
        colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
        ]
        
        H, W = img_np.shape[:2]
        overlay = img_np.astype(np.float32).copy()
        
        for i, (entity, mask) in enumerate(masks.items()):
            color = np.array(colors[i % len(colors)], dtype=np.float32)
            
            # Ensure mask is correct size
            if mask.shape != (H, W):
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((W, H), Image.NEAREST)
                mask = np.array(mask_pil) / 255.0
            
            # Apply colored overlay
            mask_3d = np.stack([mask, mask, mask], axis=-1)
            overlay = overlay * (1 - 0.4 * mask_3d) + color * 0.4 * mask_3d
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        if output_path:
            Image.fromarray(overlay).save(output_path)
        
        return overlay


class SimpleMaskGenerator:
    """Fallback mask generator when Grounded-SAM isn't available."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def generate_center_mask(self, H: int, W: int, size_ratio: float = 0.5) -> np.ndarray:
        """Generate a smooth center Gaussian mask."""
        y, x = np.ogrid[:H, :W]
        center_y, center_x = H // 2, W // 2
        sigma_y = H * size_ratio * 0.4
        sigma_x = W * size_ratio * 0.4
        
        mask = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + 
                        (y - center_y)**2 / (2 * sigma_y**2)))
        return (mask > 0.3).astype(np.float32)
    
    def generate_quadrant_masks(self, H: int, W: int, num_objects: int) -> List[np.ndarray]:
        """Generate Gaussian blob masks at different positions."""
        masks = []
        
        if num_objects == 1:
            masks.append(self.generate_center_mask(H, W, size_ratio=0.6))
        elif num_objects == 2:
            # Left and right blobs
            for center_x in [W // 3, 2 * W // 3]:
                y, x = np.ogrid[:H, :W]
                sigma_y, sigma_x = H * 0.3, W * 0.25
                mask = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + 
                                (y - H//2)**2 / (2 * sigma_y**2)))
                masks.append((mask > 0.3).astype(np.float32))
        elif num_objects <= 4:
            # Quadrant blobs
            positions = [
                (H // 3, W // 3),
                (H // 3, 2 * W // 3),
                (2 * H // 3, W // 3),
                (2 * H // 3, 2 * W // 3),
            ]
            for i in range(num_objects):
                center_y, center_x = positions[i]
                y, x = np.ogrid[:H, :W]
                sigma_y, sigma_x = H * 0.2, W * 0.2
                mask = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + 
                                (y - center_y)**2 / (2 * sigma_y**2)))
                masks.append((mask > 0.3).astype(np.float32))
        else:
            for i in range(num_objects):
                masks.append(self.generate_center_mask(H, W, size_ratio=0.3))
        
        return masks


# Factory function
def get_segmenter(device: str = "cuda") -> GroundedSAMSegmenter:
    """Get the Grounded-SAM segmenter (with fallback built-in)."""
    return GroundedSAMSegmenter(device=device)
