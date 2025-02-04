"""Utility functions for similarity calculations and saliency computation."""
from typing import List, Any
import numpy as np
from numpy.typing import NDArray

def compute_similarity(embedding1: NDArray, embedding2: NDArray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

# Compute saliency for a single item based on an utterance.
def compute_saliency(
    item: Any,
    utterance: Any,
    decay_factor: float = 1.0,
    b: float = 1.0,
    c: float = 1.0
) -> float:
    """Compute saliency for a single item based on an utterance.
    
    Saliency = max(b * similarity(item, interpretation), c * similarity(item, text)) * weight * decay
    where decay = decay_factor^(current_turn - item_turn) if decay_factor != 1
    
    Args:
        item: Item to compute saliency for (must have embedding, weight, last_accessed_turn attributes)
        utterance: Utterance to compute similarity against (must have embedding, turn_number attributes)
        decay_factor: Factor for time-based decay (default: 1.0)
        b: Weight for interpretation similarity (default: 1.0)
        c: Weight for text similarity (default: 1.0)
        
    Returns:
        Computed saliency value
    """
    # Compute similarities
    similarity_interpretation = compute_similarity(
        item.embedding,
        utterance.interpretation.embedding if hasattr(utterance, 'interpretation') else utterance.embedding
    )
    similarity_text = compute_similarity(item.embedding, utterance.embedding)
    
    # Compute time-based decay using turn numbers
    turns_elapsed = max(0, utterance.turn_number - item.last_accessed_turn)
    decay = decay_factor ** turns_elapsed if decay_factor != 1.0 else 1.0
        
    # Compute saliency
    # if no weight is provided, use the default weight of 1.0
    item_weight = item.weight if hasattr(item, 'weight') else 1.0
    saliency = max(b * similarity_interpretation, c * similarity_text) * item_weight * decay
    return saliency

# Recalibrate saliency for a collection of items (e.g. thoughts or memories) based on an utterance.
def recalibrate_all_saliency(
    items: List[Any],
    utterance: Any,
    decay_factor: float = 1.0,
    b: float = 1.0,
    c: float = 1.0
) -> None:
    """Recalibrate saliency for a collection of items based on an utterance.
    
    Updates the saliency attribute of each item in place based on similarity to the utterance
    and time decay since the item was created/last accessed.

    if utterance.turn_number is smaller than the last_accessed_turn of any item, skip the item
    
    Args:
        items: List of items to recalibrate (each must have embedding, weight, turn_number attributes)
        utterance: Utterance to compute similarity against (must have embedding, turn_number attributes)
        decay_factor: Factor for time-based decay (default: 1.0)
        b: Weight for interpretation similarity (default: 1.0)
        c: Weight for text similarity (default: 1.0)
    """
    for item in items:
        if utterance.turn_number < item.last_accessed_turn:
            continue
        item.saliency = compute_saliency(item, utterance, decay_factor, b, c)