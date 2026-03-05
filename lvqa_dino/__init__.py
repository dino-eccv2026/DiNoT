"""
L-DINO-CoT: Localized Directional Noise Optimization with Chain-of-Thought

This module implements localized VQA scoring for improved semantic alignment
in text-to-image diffusion models.
"""

from .prompt_decomposer import EntityInfo, StructuredCoTDecomposer, create_entities_from_simple_format
from .segmentation import GroundedSAMSegmenter
from .vqa_scorer import LocalizedVQAScorer, DependencyGraphEvaluator
from .daam_attention import DAAMExtractor
from .optimizer import LDINOOptimizer
from .lvqa_scoring import LVQAScorer, SimpleSegmenter, compute_lvqa_loss

__all__ = [
    'EntityInfo',
    'StructuredCoTDecomposer',
    'create_entities_from_simple_format',
    'GroundedSAMSegmenter',
    'LocalizedVQAScorer',
    'DependencyGraphEvaluator',
    'DAAMExtractor',
    'LDINOOptimizer',
    # L-VQA scoring
    'LVQAScorer',
    'SimpleSegmenter',
    'compute_lvqa_loss',
]
