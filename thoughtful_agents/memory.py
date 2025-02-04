"""Memory module containing MemoryItem and MemoryStore classes."""
from typing import List, Optional, Dict, Any
import uuid
import numpy as np
from numpy.typing import NDArray
from .utils import get_embedding

class MemoryItem:
    """A single memory item.
    
    Attributes:
        id: Auto-generated unique identifier for the memory (format: "memory-{uuid}")
        agent_id: ID of the agent who owns this memory
        text: The text content of the memory
        embedding: Vector representation of the text as numpy array (generated from text)
        weight: Importance score of the memory predefined by user
        memory_type: Type of memory ("working" or "long_term")
        turn_number: Turn number when this memory was created
        saliency: Importance score of the memory
        last_accessed_turn: Last turn number when this memory was retrieved
        retrieval_count: Number of times this memory has been retrieved
    """
    
    def __init__(
        self,
        agent_id: str,
        text: str,
        weight: float,
        embedding: NDArray[np.float32],
        memory_type: str = "long_term",
        turn_number: int = 0
    ) -> None:
        """Initialize a memory item.
        
        Args:
            agent_id: ID of the agent who owns this memory
            text: The text content of the memory
            weight: Importance score predefined by user
            embedding: Vector representation of the text
            memory_type: Type of memory ("working" or "long_term")
            turn_number: Turn number when this memory was created
        """
        self.id = f"memory-{uuid.uuid4()}"
        self.agent_id = agent_id
        self.weight = weight
        self.text = text
        self.embedding = embedding
        self.turn_number = turn_number
        self.memory_type = memory_type

        # Tracking metrics
        self.saliency = 1.0  # default saliency
        self.last_accessed_turn = turn_number
        self.retrieval_count = 0

class MemoryStore:
    """Manages storage and retrieval of memories.
    
    Maintains separate lists for working and long-term memories,
    with methods for adding, retrieving, and searching memories.
    
    Attributes:
        working_memory: List of working (temporary) memories
        long_term_memory: List of long-term (permanent) memories
    """
    
    def __init__(self) -> None:
        """Initialize memory store."""
        self.working_memory: List[MemoryItem] = []
        self.long_term_memory: List[MemoryItem] = []
        
    async def add_memory(self, agent_id: str, text: str, weight: float = 1.0, memory_type: str = "long_term", turn_number: int = 0) -> None:
        """Add a new memory item with generated embedding.
        
        Args:
            agent_id: ID of the agent who owns this memory
            text: The text content of the memory
            weight: Importance score predefined by user
            memory_type: Type of memory ("working" or "long_term")
            turn_number: Turn number when this memory was created
            
        Raises:
            ValueError: If text is empty
            LLMAPIError: If embedding generation fails
        """
        if not text.strip():
            raise ValueError("Empty text provided for memory")
            
        # Generate embedding from text
        embedding_list = await get_embedding(text)
        embedding = np.array(embedding_list, dtype=np.float32)
        
        # Create memory item
        memory = MemoryItem(
            agent_id=agent_id,
            text=text,
            weight=weight,
            embedding=embedding,
            memory_type=memory_type,
            turn_number=turn_number
        )
        
        # Add to appropriate memory store
        if memory.memory_type == "working":
            self.working_memory.append(memory)
        else:
            self.long_term_memory.append(memory)

    # TODO: Implement retrieval methods