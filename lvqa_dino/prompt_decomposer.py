"""
Structured Chain-of-Thought Prompt Decomposer

Takes hardcoded entity-attribute mappings as input and generates
Reflection (target) and Leakage (distractor) questions for VQA scoring.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class EntityInfo:
    """Represents an entity with its attributes and VQA questions."""
    name: str  # e.g., "cat", "car", "robot"
    target_attributes: List[str] = field(default_factory=list)  # e.g., ["red", "fluffy"]
    distractor_attributes: List[str] = field(default_factory=list)  # attributes from OTHER entities
    reflection_questions: List[str] = field(default_factory=list)  # "Is the cat fluffy?"
    leakage_questions: List[str] = field(default_factory=list)  # "Is the cat metallic?"


class StructuredCoTDecomposer:
    """
    Decomposes prompts into Entity-Attribute tuples.
    
    This version takes hardcoded input rather than using LLM parsing.
    The entity information should be provided directly by the caller.
    """
    
    def __init__(self):
        pass
    
    def decompose_from_dict(
        self, 
        entities_dict: List[Dict[str, any]]
    ) -> List[EntityInfo]:
        """
        Creates EntityInfo objects from a list of dictionaries.
        
        Args:
            entities_dict: List of dicts with keys:
                - 'name': str - entity name (e.g., "cat")
                - 'attributes': List[str] - target attributes (e.g., ["fluffy", "white"])
        
        Returns:
            List of EntityInfo with reflection/leakage questions generated
        
        Example:
            entities_dict = [
                {"name": "cat", "attributes": ["fluffy"]},
                {"name": "robot", "attributes": ["metallic"]}
            ]
        """
        # First pass: collect all entities
        entities = []
        all_attributes = set()
        
        for entity_data in entities_dict:
            entity = EntityInfo(
                name=entity_data['name'],
                target_attributes=entity_data.get('attributes', [])
            )
            entities.append(entity)
            all_attributes.update(entity.target_attributes)
        
        # Second pass: assign distractors and generate questions
        for entity in entities:
            # Distractor attributes are those belonging to OTHER entities
            entity.distractor_attributes = [
                attr for attr in all_attributes 
                if attr not in entity.target_attributes
            ]
            
            # Generate Reflection questions (target attributes)
            entity.reflection_questions = self._generate_reflection_questions(entity)
            
            # Generate Leakage questions (distractor attributes)
            entity.leakage_questions = self._generate_leakage_questions(entity)
        
        return entities
    
    def _generate_reflection_questions(self, entity: EntityInfo) -> List[str]:
        """
        Generate questions to verify target attributes are present.
        
        These should return "Yes" for a correctly generated image.
        """
        questions = []
        for attr in entity.target_attributes:
            # Different question formats for different attribute types
            if self._is_color(attr):
                questions.append(f"Is the {entity.name} {attr}?")
            elif self._is_texture(attr):
                questions.append(f"Is the {entity.name} {attr}?")
            elif self._is_material(attr):
                questions.append(f"Is the {entity.name} made of {attr}?")
            else:
                questions.append(f"Is the {entity.name} {attr}?")
        
        # Add existence question
        questions.append(f"Is there a {entity.name}?")
        
        return questions
    
    def _generate_leakage_questions(self, entity: EntityInfo) -> List[str]:
        """
        Generate questions to verify distractor attributes are NOT present.
        
        These should return "No" for a correctly generated image (no attribute leakage).
        """
        questions = []
        for attr in entity.distractor_attributes:
            if self._is_color(attr):
                questions.append(f"Is the {entity.name} {attr}?")
            elif self._is_texture(attr):
                questions.append(f"Is the {entity.name} {attr}?")
            elif self._is_material(attr):
                questions.append(f"Is the {entity.name} made of {attr}?")
            else:
                questions.append(f"Is the {entity.name} {attr}?")
        
        return questions
    
    def _is_color(self, attr: str) -> bool:
        """Check if attribute is a color."""
        colors = {
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
            'black', 'white', 'gray', 'grey', 'brown', 'gold', 'golden',
            'silver', 'bronze', 'cyan', 'magenta', 'turquoise', 'crimson'
        }
        return attr.lower() in colors
    
    def _is_texture(self, attr: str) -> bool:
        """Check if attribute is a texture."""
        textures = {
            'fluffy', 'smooth', 'rough', 'soft', 'hard', 'furry', 'hairy',
            'silky', 'velvety', 'glossy', 'matte', 'shiny', 'dull', 'bumpy',
            'scaly', 'feathery', 'woolly', 'spiky'
        }
        return attr.lower() in textures
    
    def _is_material(self, attr: str) -> bool:
        """Check if attribute is a material."""
        materials = {
            'metallic', 'metal', 'wooden', 'wood', 'glass', 'plastic',
            'stone', 'brick', 'crystal', 'diamond', 'rubber', 'leather',
            'fabric', 'cloth', 'paper', 'ceramic', 'porcelain', 'marble',
            'steel', 'iron', 'copper', 'aluminum', 'titanium'
        }
        return attr.lower() in materials


def create_entities_from_simple_format(
    prompt: str,
    entity_attributes: Dict[str, List[str]]
) -> List[EntityInfo]:
    """
    Convenience function to create entities from a simple dictionary format.
    
    Args:
        prompt: The full prompt string (for reference)
        entity_attributes: Dict mapping entity names to their attributes
            e.g., {"cat": ["fluffy", "white"], "robot": ["metallic", "tall"]}
    
    Returns:
        List of EntityInfo objects with questions generated
    
    Example:
        entities = create_entities_from_simple_format(
            prompt="a fluffy white cat and a metallic tall robot",
            entity_attributes={
                "cat": ["fluffy", "white"],
                "robot": ["metallic", "tall"]
            }
        )
    """
    decomposer = StructuredCoTDecomposer()
    entities_dict = [
        {"name": name, "attributes": attrs}
        for name, attrs in entity_attributes.items()
    ]
    return decomposer.decompose_from_dict(entities_dict)
