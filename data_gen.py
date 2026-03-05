"""
Dependency Graph Data Generator

Uses OpenAI API to decompose prompts into the dependency_graph format
required by the L-DINO-CoT pipeline.

Output format (per prompt):
{
  "prompt": "...",
  "dependency_graph": {
    "nodes": [
      {"id": "q1", "type": "Entity", "concept": "cat", "question": "Is there a cat in the image?", "parent_id": null},
      {"id": "q2", "type": "Attribute", "concept": "blue", "question": "Is the cat blue?", "parent_id": "q1"},
      {"id": "q3", "type": "Relation", "concept": "on", "question": "Is the cat on the table?", "parent_id": ["q1", "q4"]},
      ...
    ]
  }
}

Usage:
    python data_gen.py --input prompts.txt --output data.json --api_key YOUR_KEY
    python data_gen.py -i prompts.txt -o data.json  # uses OPENAI_API_KEY env var
"""

import os
import json
import argparse
import time
from typing import Dict, Optional
from openai import OpenAI


SYSTEM_PROMPT = """You are a prompt decomposition assistant for text-to-image generation evaluation.

Given a text prompt, decompose it into a **dependency graph** of yes/no VQA questions.

Node types:
- **Entity**: A physical object/subject. question: "Is there a <entity> in the image?"  parent_id: null (root)
- **Attribute**: A property of an entity (color, material, texture, size, style, etc.). question: "Is the <entity> <attribute>?"  parent_id: the entity's id
- **Relation**: A spatial or logical relationship between entities. question: Formulate a yes/no question about the relationship.  parent_id: list of the involved entity ids (e.g. ["q1", "q3"])

Rules:
1. Every entity is a root node (parent_id: null)
2. An entity can have **multiple** attribute nodes — create one Attribute node per distinct property (e.g. color, material, size, shape are each separate nodes)
3. Attributes always have exactly one parent (the entity they describe)
4. Relations always have multiple parents (the entities involved)
5. IDs are sequential: "q1", "q2", "q3", ...
6. Entity nodes come first, then their attributes, then relations
7. Only include things explicitly stated in the prompt
8. Questions must be answerable with yes/no from an image

Return ONLY a JSON object with a single key "nodes" containing the array of node objects. Example:

{
  "nodes": [
    {"id": "q1", "type": "Entity", "concept": "cube", "question": "Is there a cube in the image?", "parent_id": null},
    {"id": "q2", "type": "Attribute", "concept": "red", "question": "Is the cube red?", "parent_id": "q1"},
    {"id": "q3", "type": "Entity", "concept": "sphere", "question": "Is there a sphere in the image?", "parent_id": null},
    {"id": "q4", "type": "Attribute", "concept": "blue", "question": "Is the sphere blue?", "parent_id": "q3"},
    {"id": "q5", "type": "Relation", "concept": "on top of", "question": "Is the cube on top of the sphere?", "parent_id": ["q1", "q3"]}
  ]
}"""


def generate_dependency_graph(
    client: OpenAI,
    prompt: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3
) -> Optional[Dict]:
    """
    Use OpenAI to decompose a prompt into a dependency graph.
    
    Args:
        client: OpenAI client
        prompt: The input prompt to decompose
        model: OpenAI model to use
        max_retries: Number of retries on failure
        
    Returns:
        Dict with "nodes" key containing the dependency graph
    """
    user_message = f'Decompose this text-to-image prompt into a dependency graph:\n\n"{prompt}"'
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Validate structure
            if "nodes" not in result:
                print(f"  [Retry {attempt+1}] Missing 'nodes' key, got keys: {list(result.keys())}")
                continue
            
            nodes = result["nodes"]
            if not isinstance(nodes, list) or len(nodes) == 0:
                print(f"  [Retry {attempt+1}] Invalid nodes: expected non-empty list")
                continue
            
            # Validate each node has required fields
            valid = True
            for node in nodes:
                for field in ["id", "type", "concept", "question", "parent_id"]:
                    if field not in node:
                        print(f"  [Retry {attempt+1}] Node missing field '{field}': {node}")
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Ensure at least one Entity (root) node exists
            entity_count = sum(1 for n in nodes if n["type"] == "Entity")
            if entity_count == 0:
                print(f"  [Retry {attempt+1}] No Entity nodes found")
                continue
            
            print(f"  ✓ Generated {len(nodes)} nodes ({entity_count} entities)")
            return result
            
        except json.JSONDecodeError as e:
            print(f"  [Retry {attempt+1}] JSON parse error: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"  [Retry {attempt+1}] API error: {e}")
            time.sleep(2)
    
    # Fallback: create a minimal single-entity graph
    print(f"  [FALLBACK] Creating minimal graph for: {prompt[:50]}...")
    return {
        "nodes": [
            {"id": "q1", "type": "Entity", "concept": prompt, "question": f"Does the image match: {prompt}?", "parent_id": None}
        ]
    }


def process_prompts(
    input_file: str,
    output_file: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_prompts: Optional[int] = None,
    delay: float = 0.3
) -> Dict:
    """
    Process a file of prompts and generate dependency graphs.
    Supports resumption: if output_file already exists, skips already-processed prompts.
    
    Args:
        input_file: Path to txt file with one prompt per line
        output_file: Path to output JSON file
        api_key: OpenAI API key
        model: OpenAI model to use
        max_prompts: Maximum number of prompts to process (None = all)
        delay: Delay between API calls in seconds
        
    Returns:
        Dict with all processed prompts
    """
    client = OpenAI(api_key=api_key)
    
    # Read input prompts
    with open(input_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(prompts)} prompts from {input_file}")
    
    # Load existing output for resumption
    results = {"prompts": []}
    start_idx = 0
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
            start_idx = len(results["prompts"])
            print(f"Resuming from {start_idx} existing entries")
        except (json.JSONDecodeError, KeyError):
            print(f"Could not parse existing {output_file}, starting fresh")
    
    # Determine range
    end_idx = len(prompts)
    if max_prompts is not None:
        end_idx = min(start_idx + max_prompts, len(prompts))
    
    if start_idx >= end_idx:
        print("All prompts already processed!")
        return results
    
    print(f"Processing prompts {start_idx + 1} to {end_idx}...\n")
    
    for i in range(start_idx, end_idx):
        prompt = prompts[i]
        print(f"[{i+1}/{end_idx}] \"{prompt[:70]}{'...' if len(prompt) > 70 else ''}\"")
        
        graph = generate_dependency_graph(client, prompt, model)
        
        entry = {
            "prompt": prompt,
            "dependency_graph": graph
        }
        
        results["prompts"].append(entry)
        
        # Save after each prompt for resumption
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Rate limiting
        if delay > 0 and i < end_idx - 1:
            time.sleep(delay)
    
    print(f"\n✓ Saved {len(results['prompts'])} prompts to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate dependency graph decompositions for L-DINO-CoT pipeline using OpenAI"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input file with prompts (one per line)")
    parser.add_argument("--output", "-o", type=str, default="data.json",
                        help="Output JSON file (default: data.json)")
    parser.add_argument("--api_key", "-k", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--max", "-n", type=int, default=None,
                        help="Maximum number of prompts to process")
    parser.add_argument("--delay", "-d", type=float, default=0.1,
                        help="Delay between API calls in seconds (default: 0.3)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Use --api_key or set OPENAI_API_KEY env var.")
        return
    
    process_prompts(
        input_file=args.input,
        output_file=args.output,
        api_key=api_key,
        model=args.model,
        max_prompts=args.max,
        delay=args.delay
    )


if __name__ == "__main__":
    main()
