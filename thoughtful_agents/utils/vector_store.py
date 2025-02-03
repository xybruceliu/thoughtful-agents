"""Vector store for efficient embedding storage and retrieval."""
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import faiss
from dataclasses import dataclass

@dataclass
class StoredItem:
    """Item stored in the vector store."""
    id: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    turn_number: int

class VectorStore:
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self, dimension: int):
        """Initialize vector store.
        
        Args:
            dimension: Dimensionality of the vectors to store
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for cosine similarity after normalization
        self.stored_items: Dict[int, StoredItem] = {}  # Map FAISS ids to items
        self.current_id = 0
    
    def add_item(
        self,
        item_id: str,
        embedding: List[float],
        content: Dict[str, Any],
        metadata: Dict[str, Any],
        turn_number: int
    ) -> None:
        """Add an item to the store.
        
        Args:
            item_id: Unique identifier for the item
            embedding: Vector embedding of the item
            content: Content of the item (e.g., text and its embedding)
            metadata: Additional metadata about the item
            turn_number: Turn number when this item was created
        """
        # Convert to numpy and normalize
        vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)
        
        # Add to FAISS index
        self.index.add(vector)
        
        # Store item data
        self.stored_items[self.current_id] = StoredItem(
            id=item_id,
            content=content,
            metadata=metadata,
            turn_number=turn_number
        )
        self.current_id += 1
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        min_similarity: float = 0.7,
        turn_window: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[StoredItem, float]]:
        """Search for most similar items.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            min_similarity: Minimum similarity threshold
            turn_window: Optional (start_turn, end_turn) to filter results
            
        Returns:
            List of (item, similarity_score) tuples
        """
        if not self.stored_items:
            return []
            
        # Convert to numpy and normalize
        vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)
        
        # Search in FAISS index
        distances, indices = self.index.search(vector, k)
        
        # Convert L2 distances to cosine similarities and filter results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS padding
                continue
                
            item = self.stored_items[idx]
            
            # Skip if outside turn window
            if turn_window:
                start_turn, end_turn = turn_window
                if not (start_turn <= item.turn_number <= end_turn):
                    continue
            
            # Convert L2 distance to cosine similarity
            # similarity = 1 - (dist / 2)  # For normalized vectors
            similarity = 1 - (dist / 4)  # Adjust scale for better range
            
            if similarity >= min_similarity:
                results.append((item, similarity))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def update_metadata(self, item_id: str, metadata_updates: Dict[str, Any]) -> None:
        """Update metadata for an item.
        
        Args:
            item_id: ID of item to update
            metadata_updates: Dictionary of metadata updates
        """
        for idx, item in self.stored_items.items():
            if item.id == item_id:
                item.metadata.update(metadata_updates)
                break
    
    def get_item_by_id(self, item_id: str) -> Optional[StoredItem]:
        """Retrieve an item by its ID.
        
        Args:
            item_id: ID of item to retrieve
            
        Returns:
            StoredItem if found, None otherwise
        """
        for item in self.stored_items.values():
            if item.id == item_id:
                return item
        return None 