"""Workflow for removing target objects from trajectory images."""
import re
from typing import Tuple
import textwrap
import re
from google.genai import types

from pydantic import BaseModel
class TargetObject(BaseModel):
    object: str


def extract_target_object_regex(instruction: str) -> str:
    """Extract the target object from language instruction.
    
    :param instruction: Language instruction from dataset
    :returns: Target object to remove
    """
    instruction_lower = instruction.lower().strip()

    # Early return for non-English or unclear instructions
    if not any(word in instruction_lower for word in ['the', 'a', 'an', 'put', 'move', 'place', 'take', 'turn', 'flip', 'knock']):
        return "object"

    # Helper function to clean extracted objects
    def clean_extracted_object(obj):
        # Remove common trailing action phrases
        cleaned = re.sub(r'\s+(and|on|near|to|into|from|with|over).*$', '', obj)
        # Remove trailing verbs
        cleaned = re.sub(r'\s+(put|place|move|take|remove|reach|reachs|use|unfold|fold|open|close).*$', '', cleaned)
        return cleaned.strip()

    # Expanded list of common objects
    object_words = {'cloth', 'towel', 'fabric', 'napkin', 'spoon', 'spatula', 'brush',
                    'pot', 'pan', 'lid', 'cover', 'cup', 'bowl', 'plate', 'fork',
                    'knife', 'bottle', 'can', 'box', 'cube', 'cylinder', 'sphere',
                    'block', 'drawer', 'microwave', 'sink', 'stove', 'lever',
                    'switch', 'zipper', 'bag', 'figure', 'item', 'object',
                    'almonds', 'broccoli', 'banana', 'potato', 'faucet', 'oven',
                    'chip', 'juice', 'sushi', 'pear', 'carrot', 'sweet potato',
                    'eggplant', 'cheese', 'knife', 'plunger', 'stick', 'figure'}

    # Pattern matching with priority order, handling multi-word objects and action boundaries
    patterns = [
        # Patterns with specific action boundaries
        (r"pick up (?:the )?([\w\s]+?)\s+(?:and put|and place|and move)", 1),
        (r"put (?:the )?([\w\s]+?)\s+(?:on|in|into|and put|and place)", 1),
        (r"move (?:the )?([\w\s]+?)\s+(?:to|on|into|and place)", 1),
        (r"place (?:the )?([\w\s]+?)\s+(?:behind|in|on|near|to|into)", 1),
        (r"take (?:the )?([\w\s]+?)\s+(?:out of|off|from)", 1),
        (r"close (?:the )?([\w\s]+?)\s+(?:and|with)", 1),
        (r"open (?:the )?([\w\s]+?)\s+(?:and|with)", 1),
        
        # General patterns for actions
        (r"turn (?:the )?([\w\s]+?)\s+(?:left|right|vertical)", 1),
        (r"flip (?:the )?([\w\s]+?)\s+upright", 1),
        (r"knock (?:the )?([\w\s]+?)\s+over", 1),
        (r"unfold (?:the )?([\w\s]+?)\s+(?:from|to)", 1),
        (r"fold (?:the )?([\w\s]+?)\s+(?:in half|from|to)", 1),
        (r"pour (?:the )?([\w\s]+?)\s+(?:in|into)", 1),
        (r"reaching (?:the )?([\w\s]+?)$", 1),
        (r"zip (?:the )?([\w\s]+?)$", 1),
        (r"close (?:the )?([\w\s]+?)$", 1),
        (r"open (?:the )?([\w\s]+?)$", 1),
        (r"pick (?:up )?(?:the )?([\w\s]+?)$", 1),
    ]
    
    for pattern, group_idx in patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            obj = match.group(group_idx).strip()
            # Clean up articles and common descriptors
            obj = re.sub(r'^(the|a|an)\s+', '', obj)
            # Apply cleaning helper function
            cleaned_obj = clean_extracted_object(obj)
            
            # Handle color extraction logic
            color_match = re.match(r'(red|blue|green|yellow|violet|purple|orange|pink|brown|black|white|gray|grey)\s+(.+)', cleaned_obj)
            if color_match:
                object_part = color_match.group(2)
                # Only return the object part if it's a known object, otherwise keep the full phrase
                if any(word in object_part for word in object_words):
                    return object_part
            
            return cleaned_obj

    # Fallback: look for common objects by elimination
    action_verbs = {'move', 'put', 'place', 'bring', 'fold', 'unfold', 'turn', 
                    'close', 'open', 'pick', 'grasp', 'pour', 'reach', 'reaching',
                    'take', 'lift', 'push', 'pull', 'slide', 'rotate', 'flip', 'knock',
                    'sweep', 'zip', 'removed', 'removed', 'placed', 'poured', 'arranged',
                    'straighened'}
    
    skip_words = {'the', 'a', 'an', 'in', 'on', 'to', 'from', 'left', 'right', 
                  'top', 'bottom', 'half', 'edge', 'front', 'back', 'or', 'and',
                  'up', 'down', 'center', 'middle', 'side', 'of', 'behind', 'near',
                  'table', 'surface', 'lower', 'upper', 'into', 'onto', 'which', 'is',
                  'it', 'that', 'with', 'using', 'so', 'did', 'not', 'move', 'they', 'a',
                  'which', 'is', 'in', 'front', 'back', 'which'}
    
    words = instruction_lower.split()
    
    # First pass: look for known objects, prioritizing the end of the sentence
    for i in range(len(words) - 1, -1, -1):
        word = words[i]
        if word in object_words:
            # Check for a two-word object (e.g., 'sweet potato')
            if i > 0 and words[i-1] + ' ' + word in object_words:
                return words[i-1] + ' ' + word
            return word

    # Second pass: look for words that aren't verbs or skip words, again from the end
    for i in range(len(words) - 1, -1, -1):
        word = words[i]
        if word not in action_verbs and word not in skip_words and len(word) > 2:
            return word
    
    return "object"  # Default fallback

def create_inpaint_prompts(instruction: str) -> Tuple[str, str]:
    """Create inpainting and negative prompts from instruction.
    
    :param instruction: Language instruction
    :returns: (inpaint_prompt, negative_prompt)
    """
    target_obj = extract_target_object_regex(instruction)
    
    # Create focused removal prompt
    inpaint_prompt = f"Remove the {target_obj} from the scene. Show empty table surface or background where the {target_obj} was."
    
    # Negative prompt to avoid artifacts
    negative_prompt = "robot arm, gripper, mechanical parts, blur, artifacts"
    
    return inpaint_prompt, negative_prompt

def extract_target_object_gemini(gemini_client, robot_instruction: str, model_type: str = "gemini-2.5-flash"):    
    prompt = textwrap.dedent(f"""
            Extract the target object(s) from the following robot instruction.
            If no object is mentioned, just return 'object'. Robot Instruction: '{robot_instruction}' Reply ONLY with the object.
            """).strip()
    response = gemini_client.models.generate_content(
        model=model_type,
        contents=prompt,
        # config={
        #     "response_mime_type": "application/json",
        #     "response_schema": TargetObject,
        # },
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TargetObject,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    )
    return response.parsed.object
