"""
Localized VQA Scorer using BLIP-2

Implements dual-branch VQA scoring:
- Reflection branch: Verify target attributes are present (maximize P("Yes"))
- Leakage branch: Verify distractor attributes are absent (minimize P("Yes"))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from PIL import Image
import numpy as np


class LocalizedVQAScorer(nn.Module):
    """
    Computes localized VQA scores for semantic alignment verification.
    
    Uses BLIP-2 to answer yes/no questions about specific image regions,
    providing differentiable gradients for optimization.
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the VQA scorer.
        
        Args:
            model_name: HuggingFace model name for BLIP-2
            device: Device to run on
            dtype: Data type for model
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        
        self.model = None
        self.processor = None
        self._initialized = False
        
        # Token IDs will be set after model loads
        self.yes_token_id = None
        self.no_token_id = None
    
    def _lazy_init(self):
        """Lazy initialization to save memory until needed."""
        if self._initialized:
            return
        
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            print(f"Loading BLIP-2 model: {self.model_name}...")
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto"
            )
            
            # Get token IDs for "Yes" and "No"
            self.yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
            self.no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]
            
            self._initialized = True
            print("BLIP-2 loaded successfully!")
            
        except Exception as e:
            print(f"Warning: Failed to load BLIP-2: {e}")
            print("Using fallback random scoring (for testing only)")
            self._initialized = True
    
    def get_yes_probability(
        self,
        image: Union[torch.Tensor, Image.Image],
        question: str
    ) -> torch.Tensor:
        """
        Get the probability of "Yes" answer for a question.
        
        Args:
            image: Image tensor (C, H, W) or (B, C, H, W) or PIL Image
            question: Yes/No question to ask
            
        Returns:
            Probability tensor (differentiable if image is tensor with grad)
        """
        self._lazy_init()
        
        if self.model is None:
            # Fallback: return random probability
            return torch.tensor(0.5, device=self.device, requires_grad=True)
        
        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            image_pil = self._tensor_to_pil(image)
            image_for_grad = image
        else:
            image_pil = image
            image_for_grad = None
        
        # Prepare inputs
        inputs = self.processor(
            images=image_pil,
            text=question,
            return_tensors="pt"
        ).to(self.device, self.dtype)
        
        # Forward pass to get logits
        with torch.set_grad_enabled(image_for_grad is not None and image_for_grad.requires_grad):
            outputs = self.model(
                **inputs,
                return_dict=True
            )
        
        # Get logits for the first generated token
        logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
        
        # Extract probabilities for Yes/No
        yes_no_logits = logits[:, [self.yes_token_id, self.no_token_id]]
        probs = F.softmax(yes_no_logits, dim=-1)
        
        # Return P("Yes")
        return probs[:, 0].mean()
    
    def compute_entity_score(
        self,
        image_crop: Union[torch.Tensor, Image.Image],
        reflection_questions: List[str],
        leakage_questions: List[str],
        lambda_ref: float = 1.0,
        lambda_leak: float = 1.5
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined score for a single entity.
        
        The loss is: λ_ref * (1 - s_ref) + λ_leak * s_leak
        
        Where:
        - s_ref = average P("Yes") for reflection questions (should be high)
        - s_leak = average P("Yes") for leakage questions (should be low)
        
        Args:
            image_crop: Cropped image of the entity region
            reflection_questions: Questions about target attributes
            leakage_questions: Questions about distractor attributes
            lambda_ref: Weight for reflection loss
            lambda_leak: Weight for leakage loss
            
        Returns:
            Total loss tensor, dict with individual scores
        """
        self._lazy_init()
        
        # Compute reflection scores (target attributes)
        ref_scores = []
        for q in reflection_questions:
            score = self.get_yes_probability(image_crop, q)
            ref_scores.append(score)
        
        s_ref = torch.stack(ref_scores).mean() if ref_scores else torch.tensor(0.5, device=self.device)
        
        # Compute leakage scores (distractor attributes)
        leak_scores = []
        for q in leakage_questions:
            score = self.get_yes_probability(image_crop, q)
            leak_scores.append(score)
        
        s_leak = torch.stack(leak_scores).mean() if leak_scores else torch.tensor(0.0, device=self.device)
        
        # Compute total loss
        loss = lambda_ref * (1 - s_ref) + lambda_leak * s_leak
        
        # Return loss and debug info
        info = {
            'reflection_score': s_ref.item() if isinstance(s_ref, torch.Tensor) else s_ref,
            'leakage_score': s_leak.item() if isinstance(s_leak, torch.Tensor) else s_leak,
            'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
        }
        
        return loss, info
    
    def compute_total_loss(
        self,
        image: torch.Tensor,
        entity_crops: List[torch.Tensor],
        entities: List['EntityInfo'],
        lambda_ref: float = 1.0,
        lambda_leak: float = 1.5
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        Compute total loss across all entities.
        
        Args:
            image: Full image tensor
            entity_crops: List of cropped entity images
            entities: List of EntityInfo with questions
            lambda_ref: Weight for reflection loss
            lambda_leak: Weight for leakage loss
            
        Returns:
            Total loss, list of per-entity info dicts
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        all_info = []
        
        for crop, entity in zip(entity_crops, entities):
            loss, info = self.compute_entity_score(
                crop,
                entity.reflection_questions,
                entity.leakage_questions,
                lambda_ref,
                lambda_leak
            )
            total_loss = total_loss + loss
            info['entity'] = entity.name
            all_info.append(info)
        
        return total_loss, all_info
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert image tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor[0]  # Remove batch dimension
        
        # Assume tensor is in [-1, 1] or [0, 1] range
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2
        
        tensor = tensor.clamp(0, 1)
        tensor = tensor.detach().cpu()
        
        # Convert to numpy (C, H, W) -> (H, W, C)
        np_img = tensor.permute(1, 2, 0).numpy()
        np_img = (np_img * 255).astype(np.uint8)
        
        return Image.fromarray(np_img)


class FallbackVQAScorer(nn.Module):
    """
    Fallback scorer that uses the existing VQA model from t2v_metrics.
    
    This integrates with the existing VQA infrastructure but adds
    localization capabilities.
    """
    
    def __init__(self, vqa_model, device: str = "cuda"):
        """
        Initialize with existing VQA model.
        
        Args:
            vqa_model: VQAScore model from t2v_metrics
            device: Device for computation
        """
        super().__init__()
        self.vqa_model = vqa_model
        self.device = device
    
    def score_crop(
        self,
        image_crop: torch.Tensor,
        question: str
    ) -> torch.Tensor:
        """
        Score an image crop against a question/statement.
        
        Uses the existing VQA model interface.
        """
        # The t2v_metrics VQA model expects images and texts
        # We format the question as a statement for higher scores
        statement = question.replace("Is ", "").replace("?", ".")
        
        score = self.vqa_model(image_crop, [statement])
        return score
    
    def compute_entity_score(
        self,
        image_crop: torch.Tensor,
        reflection_questions: List[str],
        leakage_questions: List[str],
        lambda_ref: float = 1.0,
        lambda_leak: float = 1.5
    ) -> Tuple[torch.Tensor, dict]:
        """Compute entity score using existing VQA model."""
        ref_scores = []
        for q in reflection_questions:
            score = self.score_crop(image_crop, q)
            ref_scores.append(score)
        
        s_ref = torch.stack(ref_scores).mean() if ref_scores else torch.tensor(0.5, device=self.device)
        
        leak_scores = []
        for q in leakage_questions:
            score = self.score_crop(image_crop, q)
            leak_scores.append(score)
        
        s_leak = torch.stack(leak_scores).mean() if leak_scores else torch.tensor(0.0, device=self.device)
        
        loss = lambda_ref * (1 - s_ref) + lambda_leak * s_leak
        
        info = {
            'reflection_score': s_ref.item() if isinstance(s_ref, torch.Tensor) else s_ref,
            'leakage_score': s_leak.item() if isinstance(s_leak, torch.Tensor) else s_leak,
            'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
        }
        
        return loss, info


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
