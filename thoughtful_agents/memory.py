"""Memory module containing MemoryItem and MemoryStore classes."""
from typing import Dict, List, Optional, Any, Tuple
from .utils.vector_store import VectorStore, StoredItem

class MemoryItem:
    """A single memory item that can be stored and retrieved."""
    
    def __init__(
        self,
        id: str,
        content: Dict[str, Any],
        weight: float = 1.0,
        memory_type: str = "working",
        turn_number: int = 0
    ):
        """Initialize a memory item.
        
        Args:
            id: Unique identifier for the memory
            content: Dictionary containing text and embedding
            weight: Predefined importance weight
            memory_type: Type of memory ("working" or "long_term")
            turn_number: Turn number when this memory was created
        """
        self.id = id
        self.content = content
        self.weight = weight
        self.memory_type = memory_type
        self.turn_number = turn_number
        self.saliency = 1.0
        self.last_accessed_turn = turn_number
        self.retrieval_count = 0

class MemoryStore:
    """Manages storage and retrieval of memories using FAISS."""
    
    def __init__(self, embedding_dim: int = 1536):  # Default for text-embedding-3-small
        """Initialize memory store with vector stores for each memory type.
        
        Args:
            embedding_dim: Dimension of embeddings (default: 1536 for text-embedding-3-small)
        """
        self.working_memory = VectorStore(embedding_dim)
        self.long_term_memory = VectorStore(embedding_dim)
        
    def add(self, memory_item: MemoryItem) -> None:
        """Add a memory item to the appropriate store.
        
        Args:
            memory_item: Memory item to add
        """
        store = self.working_memory if memory_item.memory_type == "working" else self.long_term_memory
        
        store.add_item(
            item_id=memory_item.id,
            embedding=memory_item.content["embedding"],
            content=memory_item.content,
            metadata={
                "weight": memory_item.weight,
                "saliency": memory_item.saliency,
                "last_accessed_turn": memory_item.last_accessed_turn,
                "retrieval_count": memory_item.retrieval_count
            },
            turn_number=memory_item.turn_number
        )
        
    def retrieve(
        self,
        query_embedding: List[float],
        threshold: float = 0.7,
        memory_type: Optional[str] = None,
        turn_window: Optional[Tuple[int, int]] = None,
        top_k: int = 5
    ) -> List[MemoryItem]:
        """Retrieve relevant memories based on embedding similarity.
        
        Args:
            query_embedding: Embedding vector to search with
            threshold: Minimum similarity threshold
            memory_type: Optional type to filter by ("working" or "long_term")
            turn_window: Optional (start_turn, end_turn) to filter results
            top_k: Maximum number of results to return
            
        Returns:
            List of relevant memory items
        """
        results = []
        
        # Search appropriate stores
        stores = []
        if memory_type == "working" or memory_type is None:
            stores.append(self.working_memory)
        if memory_type == "long_term" or memory_type is None:
            stores.append(self.long_term_memory)
            
        # Collect results from each store
        for store in stores:
            items = store.search(
                query_embedding=query_embedding,
                k=top_k,
                min_similarity=threshold,
                turn_window=turn_window
            )
            
            # Convert StoredItems back to MemoryItems
            for item, similarity in items:
                memory = MemoryItem(
                    id=item.id,
                    content=item.content,
                    weight=item.metadata["weight"],
                    memory_type="working" if store == self.working_memory else "long_term",
                    turn_number=item.turn_number
                )
                memory.saliency = item.metadata["saliency"] * similarity  # Adjust saliency by similarity
                memory.last_accessed_turn = item.metadata["last_accessed_turn"]
                memory.retrieval_count = item.metadata["retrieval_count"] + 1
                
                # Update metadata in store
                store.update_metadata(item.id, {
                    "retrieval_count": memory.retrieval_count,
                    "saliency": memory.saliency
                })
                
                results.append(memory)
        
        # Sort by saliency and return top_k
        results.sort(key=lambda x: x.saliency, reverse=True)
        return results[:top_k]
        
    def update_saliency(self, memory_id: str) -> None:
        """Update saliency and access time for a memory item."""
        pass 