"""
DAAM (Diffusion Attention Attribution Maps) Extractor

Extracts cross-attention maps from UNet for soft differentiable localization,
used during early optimization steps before hard segmentation is reliable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Tuple
from contextlib import contextmanager


class DAAMExtractor:
    """
    Extracts and aggregates cross-attention maps from UNet for soft localization.
    
    During early optimization steps (when the object may not exist yet),
    we use these attention maps as differentiable soft masks instead of
    hard SAM masks.
    """
    
    def __init__(
        self,
        unet: nn.Module,
        tokenizer,
        device: str = "cuda"
    ):
        """
        Initialize the extractor.
        
        Args:
            unet: The UNet model from the diffusion pipeline
            tokenizer: Tokenizer for converting text to tokens
            device: Device for computation
        """
        self.unet = unet
        self.tokenizer = tokenizer
        self.device = device
        
        self.attention_maps: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._registered = False
    
    def register_hooks(self) -> None:
        """Register forward hooks on cross-attention layers."""
        if self._registered:
            return
            
        for name, module in self.unet.named_modules():
            # Look for cross-attention layers (usually named attn2)
            if 'attn2' in name and hasattr(module, 'processor'):
                hook = module.register_forward_hook(
                    self._create_hook(name)
                )
                self.hooks.append(hook)
        
        self._registered = True
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._registered = False
    
    def _create_hook(self, name: str) -> Callable:
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # Store attention weights if available
            # Note: This depends on the specific attention implementation
            if hasattr(module, 'attention_probs'):
                self.attention_maps[name] = module.attention_probs.detach()
        return hook_fn
    
    def clear_maps(self) -> None:
        """Clear stored attention maps."""
        self.attention_maps.clear()
    
    def get_token_indices(
        self,
        prompt: str,
        word: str
    ) -> List[int]:
        """
        Get token indices for a word in the prompt.
        
        Args:
            prompt: Full prompt text
            word: Word to find indices for
            
        Returns:
            List of token indices
        """
        # Tokenize the full prompt and the word
        prompt_tokens = self.tokenizer.encode(prompt)
        word_tokens = self.tokenizer.encode(word)
        
        # Find word tokens in prompt (skip special tokens)
        word_token_ids = word_tokens[1:-1] if len(word_tokens) > 2 else word_tokens
        
        indices = []
        for i, token in enumerate(prompt_tokens):
            if token in word_token_ids:
                indices.append(i)
        
        return indices if indices else [1]  # Default to first real token
    
    def get_soft_mask(
        self,
        prompt: str,
        entity_name: str,
        resolution: Tuple[int, int] = (512, 512),
        temperature: float = 0.1,
        aggregate_layers: List[str] = None
    ) -> torch.Tensor:
        """
        Generate a soft differentiable mask from attention maps.
        
        Args:
            prompt: Full prompt text
            entity_name: Entity to create mask for
            resolution: Output resolution (H, W)
            temperature: Temperature for softmax (lower = sharper)
            aggregate_layers: Which layers to use (None = all)
            
        Returns:
            Soft mask tensor (1, 1, H, W) with values in [0, 1]
        """
        if not self.attention_maps:
            # Return uniform mask if no attention maps available
            return torch.ones(1, 1, resolution[0], resolution[1], device=self.device) * 0.5
        
        # Get token indices for the entity
        token_indices = self.get_token_indices(prompt, entity_name)
        
        # Aggregate attention maps
        aggregated = None
        count = 0
        
        for layer_name, attn_map in self.attention_maps.items():
            if aggregate_layers is not None and layer_name not in aggregate_layers:
                continue
            
            # attn_map shape: (batch, heads, spatial, tokens)
            # Extract attention to entity tokens
            entity_attn = attn_map[..., token_indices].mean(dim=-1)  # Average over entity tokens
            entity_attn = entity_attn.mean(dim=1)  # Average over heads
            
            # Reshape to spatial dimensions
            spatial_size = int(entity_attn.shape[-1] ** 0.5)
            if spatial_size * spatial_size == entity_attn.shape[-1]:
                entity_attn = entity_attn.view(-1, spatial_size, spatial_size)
                
                # Resize to target resolution
                entity_attn = F.interpolate(
                    entity_attn.unsqueeze(1).float(),
                    size=resolution,
                    mode='bilinear',
                    align_corners=False
                )
                
                if aggregated is None:
                    aggregated = entity_attn
                else:
                    aggregated = aggregated + entity_attn
                count += 1
        
        if aggregated is None or count == 0:
            return torch.ones(1, 1, resolution[0], resolution[1], device=self.device) * 0.5
        
        aggregated = aggregated / count
        
        # Apply temperature-controlled sigmoid for soft thresholding
        # This keeps the mask differentiable
        threshold = aggregated.mean()
        soft_mask = torch.sigmoid((aggregated - threshold) / temperature)
        
        return soft_mask
    
    @contextmanager
    def capture_attention(self):
        """Context manager for capturing attention during forward pass."""
        self.register_hooks()
        self.clear_maps()
        try:
            yield self
        finally:
            pass  # Keep hooks registered for potential reuse


class SimpleAttentionLocalizer:
    """
    Simplified attention-based localization without hooks.
    
    This is a fallback when hook-based extraction is not available.
    It estimates object locations based on prompt structure.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def estimate_entity_regions(
        self,
        prompt: str,
        entity_names: List[str],
        image_size: Tuple[int, int] = (512, 512)
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate entity regions based on prompt structure.
        
        Uses heuristics like word order to guess spatial layout.
        
        Args:
            prompt: Full prompt
            entity_names: List of entity names
            image_size: Output size (H, W)
            
        Returns:
            Dict mapping entity names to soft masks
        """
        H, W = image_size
        masks = {}
        
        num_entities = len(entity_names)
        
        if num_entities == 1:
            # Single entity: center of image
            masks[entity_names[0]] = self._create_gaussian_mask(
                H, W, center=(H//2, W//2), sigma=H//3
            )
        elif num_entities == 2:
            # Two entities: left and right
            masks[entity_names[0]] = self._create_gaussian_mask(
                H, W, center=(H//2, W//3), sigma=W//4
            )
            masks[entity_names[1]] = self._create_gaussian_mask(
                H, W, center=(H//2, 2*W//3), sigma=W//4
            )
        else:
            # Multiple entities: distribute across image
            for i, name in enumerate(entity_names):
                angle = 2 * 3.14159 * i / num_entities
                cx = int(W//2 + W//4 * torch.cos(torch.tensor(angle)))
                cy = int(H//2 + H//4 * torch.sin(torch.tensor(angle)))
                masks[name] = self._create_gaussian_mask(
                    H, W, center=(cy, cx), sigma=min(H, W)//4
                )
        
        return masks
    
    def _create_gaussian_mask(
        self,
        H: int,
        W: int,
        center: Tuple[int, int],
        sigma: float
    ) -> torch.Tensor:
        """Create a 2D Gaussian soft mask."""
        y = torch.arange(H, device=self.device).float()
        x = torch.arange(W, device=self.device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        cy, cx = center
        gaussian = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.max()  # Normalize to [0, 1]
        
        return gaussian.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
