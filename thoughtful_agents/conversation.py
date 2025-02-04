"""Conversation module containing Conversation and Utterance classes."""
from typing import Dict, List, Optional, Any, Tuple
import uuid
import numpy as np
from numpy.typing import NDArray
from .utils import get_embedding, get_completion
from .participant import Participant

class Utterance:
    """A single utterance in a conversation.
    
    Attributes:
        id: Auto-generated unique identifier for the utterance (format: "utterance-{uuid}")
        participant_id: ID of the participant who made the utterance
        text: The text content of the utterance
        embedding: Vector representation of the text as numpy array
        weight: Importance score of the utterance predefined by user
        turn_number: The turn number when this utterance was made
        interpretation: Optional interpretation of the utterance
        thought_id: Optional ID of the thought that generated this utterance
    """
    
    def __init__(
        self,
        participant_id: str,
        text: str,
        embedding: NDArray[np.float32],
        turn_number: int,
        weight: float = 1.0,
        thought_id: Optional[str] = None,
        interpretation: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize an utterance.
        
        Args:
            participant_id: ID of the participant who made the utterance
            text: The text content of the utterance
            embedding: Vector representation of the text
            turn_number: The turn number when this utterance was made
            weight: Importance score predefined by user (default: 1.0)
            thought_id: Optional ID of the thought that generated this utterance
            interpretation: Optional interpretation of the utterance
        """
        self.id = f"utterance-{uuid.uuid4()}"
        self.participant_id = participant_id
        self.text = text
        self.embedding = embedding
        self.turn_number = turn_number
        self.weight = weight
        self.thought_id = thought_id
        self.interpretation = interpretation
        
    def __repr__(self) -> str:
        """Return a string representation of the utterance for debugging."""
        return f"Utterance(id='{self.id}', participant_id='{self.participant_id}', turn_number={self.turn_number}, text='{self.text[:30]}...')"


class Conversation:
    """Manages a conversation between multiple participants (humans and AI agents).
    
    The conversation maintains an ordered list of utterances and provides
    methods for adding new utterances and retrieving relevant context.
    
    Attributes:
        participants: List of participating humans and AI agents
        topic: Topic of the conversation
        current_turn: Current turn number
        utterances: List of all utterances in the conversation
    """
    
    def __init__(
        self, 
        participants: List[Participant], 
        topic: str
    ) -> None:
        """Initialize a conversation. Do not use directly, use create() instead.
        
        Args:
            participants: List of participating humans and AI agents
            topic: Topic of the conversation
        """
        self.participants = participants
        self.topic = topic
        self.current_turn = 0
        self.utterances: List[Utterance] = []
        
    @classmethod
    def create(
        cls,
        participants: List[Participant],
        topic: str
    ) -> "Conversation":
        """Create a new conversation.
        
        Args:
            participants: List of participating humans and AI agents
            topic: Topic of the conversation
            
        Returns:
            Initialized Conversation instance
            
        Raises:
            ValueError: If no participants provided or topic is empty
        """
        if not participants:
            raise ValueError("At least one participant is required")
        if not topic.strip():
            raise ValueError("Topic cannot be empty")
            
        return cls(participants=participants, topic=topic)
        
    def __repr__(self) -> str:
        """Return a string representation of the conversation for debugging."""
        participant_names = ', '.join(p.name for p in self.participants)
        return f"Conversation(topic='{self.topic}', current_turn={self.current_turn}, participants=[{participant_names}])"
        
    async def add_utterance(self, participant_id: str, text: str, weight: float = 1.0) -> None:
        """Add a new utterance to the conversation with interpretation.
        
        Args:
            participant_id: ID of the participant who made the utterance
            text: The text content of the utterance
            weight: Importance score predefined by user (default: 1.0)
            
        Raises:
            ValueError: If text is empty
        """
        if not text.strip():
            raise ValueError("Empty text provided for utterance")
            
        self.current_turn += 1
        
        # Generate embedding from text
        embedding_list = await get_embedding(text)
        embedding = np.array(embedding_list, dtype=np.float32)
        
        # Create utterance
        utterance = Utterance(
            participant_id=participant_id,
            text=text,
            embedding=embedding,
            turn_number=self.current_turn,
            weight=weight
        )
        
        # Generate interpretation
        interpretation = await self.interpret_utterance(utterance)
        utterance.interpretation = interpretation
        
        # Add to conversation
        self.utterances.append(utterance)
        
    async def interpret_utterance(self, utterance: Utterance, last_n: int = 10) -> Dict[str, Any]:
        """Interpret the utterance in the context of the conversation.
        
        Args:
            utterance: The utterance to interpret
            last_n: Number of recent utterances to consider for context
            
        Returns:
            Dictionary containing interpretation text and embedding
        """
        # Create prompts for interpretation
        system_prompt, user_prompt = self.create_prompt_interpretation(utterance, last_n)
        
        # Call GPT for interpretation
        interpretation_text = await get_completion(system_prompt, user_prompt)
        interpretation_embedding = await get_embedding(interpretation_text['text'])
        
        return {
            "text": interpretation_text['text'],
            "embedding": interpretation_embedding
        }
        
    def create_prompt_interpretation(self, utterance: Utterance, last_n: int) -> Tuple[str, str]:
        """Create a prompt for GPT to interpret the utterance.
        
        Args:
            utterance: The utterance to interpret
            last_n: Number of recent utterances to consider for context
            
        Returns:
            Tuple containing system and user prompts
        """
        participant_name = next((p.name for p in self.participants if p.id == utterance.participant_id), "Unknown")
        recent_utterances = self.retrieve(last_n)
        conversation_context = "\n".join(f"{u.participant_id}: {u.text}" for u in recent_utterances)
        
        system_prompt = f"You are playing a role as a participant in a conversation. Your name in the conversation is {participant_name}.\n" \
                        f"Given the last few lines of the conversation:\n{conversation_context}\n" \
                        f"Interpret what {participant_name} just said in the context of the conversation and what {participant_name} might be thinking. Be as succinct as possible and use a single sentence."
        
        user_prompt = f"Utterance: {participant_name}: {utterance.text}\nInterpretation: "
        
        return system_prompt, user_prompt
        
    def retrieve(self, last_n_turns: int = 20) -> List[Utterance]:
        """Retrieve the last n turns of conversation.
        
        Args:
            last_n_turns: Number of turns to retrieve
            
        Returns:      
            List of utterances from the last n turns
        """
        return self.utterances[-last_n_turns:]
        