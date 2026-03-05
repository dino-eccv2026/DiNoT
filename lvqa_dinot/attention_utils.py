"""
Attention utilities for L-DINO-CoT pipeline.
Implements Attend-and-Excite style attention loss for missing objects.

Based on: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_attend_and_excite
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple


class GaussianSmoothing(torch.nn.Module):
    """
    Apply Gaussian smoothing on a 2D tensor.
    Filtering is performed separately for each channel.
    """
    def __init__(self, channels: int = 1, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # Register as buffer
        self.register_buffer('kernel', kernel_2d.view(1, 1, kernel_size, kernel_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to input tensor."""
        return F.conv2d(x, self.kernel.to(x.device), padding=self.padding)


class AttentionStore:
    """
    Stores cross-attention maps during UNet forward pass.
    Used to extract attention for specific tokens.
    """
    
    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}
    
    def __call__(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str):
        """Called by attention processor to store attention maps."""
        if self.cur_att_layer >= 0 and is_cross:
            # Only store attention maps at target resolution
            if attn.shape[1] == np.prod(self.attn_res):
                self.step_store[place_in_unet].append(attn)
        
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()
    
    def between_steps(self):
        """Called after processing all attention layers in one step."""
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()
    
    def get_average_attention(self) -> Dict[str, List[torch.Tensor]]:
        """Returns the stored attention maps."""
        return self.attention_store
    
    def aggregate_attention(self, from_where: List[str] = ("up", "down", "mid")) -> torch.Tensor:
        """
        Aggregates attention across layers and heads at target resolution.
        
        Returns:
            Tensor of shape [H, W, num_tokens] with aggregated attention
        """
        out = []
        attention_maps = self.get_average_attention()
        
        for location in from_where:
            for item in attention_maps.get(location, []):
                # Reshape: [batch*heads, H*W, tokens] -> [batch*heads, H, W, tokens]
                cross_maps = item.reshape(-1, self.attn_res[0], self.attn_res[1], item.shape[-1])
                out.append(cross_maps)
        
        if len(out) == 0:
            return None
            
        out = torch.cat(out, dim=0)
        # Average across all heads and layers
        out = out.sum(0) / out.shape[0]
        return out
    
    def reset(self):
        """Reset attention store for new generation."""
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
    
    def __init__(self, attn_res: Tuple[int, int] = (16, 16)):
        """
        Initialize AttentionStore.
        
        Args:
            attn_res: Resolution of attention maps to capture (typically 16x16 for SD)
        """
        self.num_att_layers = -1  # Set by register_attention_control
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.attn_res = attn_res


class AttendExciteAttnProcessor:
    """
    Custom attention processor that captures cross-attention maps.
    """
    
    def __init__(self, attnstore: AttentionStore, place_in_unet: str):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attn.to_q(hidden_states)
        
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # Store attention maps only when gradients are needed (during optimization)
        if attention_probs.requires_grad:
            self.attnstore(attention_probs, is_cross, self.place_in_unet)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # Linear proj and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states


def get_token_indices(tokenizer, prompt: str, entity: str) -> List[int]:
    """
    Get token indices for an entity within a prompt.
    
    Args:
        tokenizer: CLIP tokenizer
        prompt: Full prompt string
        entity: Entity name to find
        
    Returns:
        List of token indices for the entity (1-indexed, excluding BOS/EOS)
    """
    # Tokenize the prompt
    tokens = tokenizer.tokenize(prompt.lower())
    entity_tokens = tokenizer.tokenize(entity.lower())
    
    # Find matching token indices
    indices = []
    for i, token in enumerate(tokens):
        for ent_token in entity_tokens:
            # Match tokens (handle </w> suffix)
            if token.replace("</w>", "") == ent_token.replace("</w>", ""):
                # Add 1 to account for BOS token
                indices.append(i + 1)
                break
    
    return indices if indices else None


def compute_attention_loss(
    attention_maps: torch.Tensor,
    token_indices: List[int],
    smoothing_lambda: float = 0.5,
    spatial_mask: Optional[torch.Tensor] = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    Compute max-attention loss for neglected tokens (Attend-and-Excite style).
    
    L_neglect = -max(A) - λ_smooth * mean(GaussianSmooth(A))
    
    Args:
        attention_maps: Aggregated attention [H, W, num_tokens]
        token_indices: Indices of tokens to maximize attention for
        smoothing_lambda: Weight for Gaussian smoothing term
        spatial_mask: Optional spatial mask for phantom bounding boxes [H, W]
        device: Target device
        
    Returns:
        Attention loss (negative, to be minimized)
    """
    if attention_maps is None or len(token_indices) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    H, W = attention_maps.shape[:2]
    device = attention_maps.device if device is None else device
    
    # Extract attention for target tokens (excluding BOS/EOS)
    attention_for_text = attention_maps[:, :, 1:-1]
    
    # Apply softmax to normalize
    attention_for_text = attention_for_text * 100
    attention_for_text = F.softmax(attention_for_text, dim=-1)
    
    # Shift indices since we removed BOS token
    shifted_indices = [idx - 1 for idx in token_indices if idx > 0]
    
    if len(shifted_indices) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute max attention for each target token
    max_attention_list = []
    smoothing = GaussianSmoothing(kernel_size=3, sigma=0.5).to(device)
    
    for idx in shifted_indices:
        if idx >= attention_for_text.shape[-1]:
            continue
            
        token_attention = attention_for_text[:, :, idx]  # [H, W]
        
        # Apply spatial mask if provided (phantom bounding box)
        if spatial_mask is not None:
            spatial_mask = spatial_mask.to(device)
            token_attention = token_attention * spatial_mask
        
        # Apply Gaussian smoothing
        token_attn_4d = token_attention.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        padded = F.pad(token_attn_4d, (1, 1, 1, 1), mode="reflect")
        smoothed = smoothing(padded).squeeze(0).squeeze(0)  # [H, W]
        
        # Max attention value (smoothed)
        max_val = smoothed.max()
        max_attention_list.append(max_val)
    
    if len(max_attention_list) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Loss: minimize negative max attention (maximize attention)
    # L = -max(A) for each token, take the worst one (min of max)
    min_max_attention = min(max_attention_list)
    
    # Target: attention should be at least 0.8 (from Attend-and-Excite threshold)
    loss = max(0, 1.0 - min_max_attention)
    
    return loss


def register_attention_control(unet, attention_store: AttentionStore) -> Dict:
    """
    Register custom attention processors on UNet to capture attention maps.
    
    Args:
        unet: UNet2DConditionModel
        attention_store: AttentionStore instance
        
    Returns:
        Original attention processors (to restore later)
    """
    original_processors = unet.attn_processors
    attn_procs = {}
    cross_att_count = 0
    
    for name in unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"  
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue
        
        cross_att_count += 1
        attn_procs[name] = AttendExciteAttnProcessor(
            attnstore=attention_store,
            place_in_unet=place_in_unet
        )
    
    unet.set_attn_processor(attn_procs)
    attention_store.num_att_layers = cross_att_count
    
    return original_processors


def restore_attention_processors(unet, original_processors: Dict):
    """Restore original attention processors on UNet."""
    unet.set_attn_processor(original_processors)
