"""Conversation module containing Conversation and Utterance classes."""
from typing import Dict, List, Optional, Any
from .agent import Agent
from .utils.vector_store import VectorStore, StoredItem

class Utterance:
    """A single utterance in a conversation."""
    
    def __init__(
        self,
        id: str,
        agent_id: str,
        content: Dict[str, Any],
        turn_number: int,
        interpretation: Optional[Dict] = None,
        thought_id: Optional[str] = None
    ):
        """Initialize an utterance.
        
        Args:
            id: Unique identifier for the utterance
            agent_id: ID of the agent who made the utterance
            content: Dictionary containing text and embedding
            turn_number: The turn number when this utterance was made
            interpretation: Optional interpretation of the utterance
            thought_id: Optional ID of the thought that generated this utterance
        """
        self.id = id
        self.agent_id = agent_id
        self.content = content
        self.turn_number = turn_number
        self.interpretation = interpretation
        self.thought_id = thought_id
        self.saliency = 1.0
        self.last_accessed_turn = turn_number
        self.retrieval_count = 0

class Conversation:
    """Manages a conversation between multiple agents."""
    
    def __init__(self, agents: List['Agent'], topic: str, embedding_dim: int = 1536):
        """Initialize a conversation.
        
        Args:
            agents: List of participating agents
            topic: Topic of the conversation
            embedding_dim: Dimension of embeddings (default: 1536 for text-embedding-3-small)
        """
        self.agents = agents
        self.topic = topic
        self.current_turn = 0
        self.vector_store = VectorStore(embedding_dim)
        
    def add_utterance(self, utterance: Utterance) -> None:
        """Add a new utterance to the conversation.
        
        Args:
            utterance: Utterance to add
        """
        self.current_turn += 1
        utterance.turn_number = self.current_turn
        
        # Store in vector store
        self.vector_store.add_item(
            item_id=utterance.id,
            embedding=utterance.content["embedding"],
            content=utterance.content,
            metadata={
                "agent_id": utterance.agent_id,
                "interpretation": utterance.interpretation,
                "thought_id": utterance.thought_id,
                "saliency": utterance.saliency,
                "last_accessed_turn": utterance.last_accessed_turn,
                "retrieval_count": utterance.retrieval_count
            },
            turn_number=utterance.turn_number
        )
        
    def get_context(
        self,
        query_embedding: Optional[List[float]] = None,
        window_size: Optional[int] = None,
        min_similarity: float = 0.7,
        top_k: int = 5
    ) -> List[Utterance]:
        """Get conversation context, optionally filtered by relevance to a query.
        
        Args:
            query_embedding: Optional embedding to find relevant utterances
            window_size: Optional number of most recent turns to include
            min_similarity: Minimum similarity threshold for semantic search
            top_k: Maximum number of results to return
            
        Returns:
            List of utterances in context window
        """
        # Calculate turn window if specified
        turn_window = None
        if window_size is not None:
            start_turn = max(1, self.current_turn - window_size + 1)
            turn_window = (start_turn, self.current_turn)
            
        if query_embedding is None:
            # Return most recent utterances if no query
            items = []
            for idx, item in self.vector_store.stored_items.items():
                if turn_window is None or (turn_window[0] <= item.turn_number <= turn_window[1]):
                    items.append((item, 1.0))  # Default similarity of 1.0
            items.sort(key=lambda x: x[0].turn_number, reverse=True)
            items = items[:top_k]
        else:
            # Search by similarity
            items = self.vector_store.search(
                query_embedding=query_embedding,
                k=top_k,
                min_similarity=min_similarity,
                turn_window=turn_window
            )
            
        # Convert to Utterances
        utterances = []
        for item, similarity in items:
            utterance = Utterance(
                id=item.id,
                agent_id=item.metadata["agent_id"],
                content=item.content,
                turn_number=item.turn_number,
                interpretation=item.metadata["interpretation"],
                thought_id=item.metadata["thought_id"]
            )
            utterance.saliency = item.metadata["saliency"] * similarity
            utterance.last_accessed_turn = self.current_turn
            utterance.retrieval_count = item.metadata["retrieval_count"] + 1
            
            # Update metadata
            self.vector_store.update_metadata(item.id, {
                "last_accessed_turn": utterance.last_accessed_turn,
                "retrieval_count": utterance.retrieval_count,
                "saliency": utterance.saliency
            })
            
            utterances.append(utterance)
            
        return utterances
        
    def get_current_speaker(self) -> Optional['Agent']:
        """Determine which agent currently has the conversation turn."""
        if self.current_turn == 0:
            return self.agents[0]  # First agent starts
            
        # Get last speaker from vector store
        items = list(self.vector_store.stored_items.values())
        if not items:
            return self.agents[0]
            
        last_speaker_id = max(items, key=lambda x: x.turn_number).metadata["agent_id"]
        current_idx = (next(i for i, a in enumerate(self.agents) if a.id == last_speaker_id) + 1) % len(self.agents)
        return self.agents[current_idx]
        
