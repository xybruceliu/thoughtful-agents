"""Agent module containing the core Agent class."""
from typing import Dict, List, Optional, Tuple

from .memory import MemoryStore
from .thought import ThoughtReservoir, Thought
from .conversation import Utterance

class Agent:
    """An agent that can generate thoughts and participate in conversations."""
    
    def __init__(
        self,
        id: str,
        name: str,
        persona: str,
        config: Dict,
    ):
        """Initialize an agent.
        
        Args:
            id: Unique identifier for the agent
            name: Display name of the agent
            persona: Persona of the agent, will be split into memory using text splitters like RecursiveCharacterTextSplitter
            config: Configuration dictionary containing thresholds and parameters
        """
        self.id = id
        self.name = name
        self.persona = persona
        self.config = config
        
        self.memory_store = MemoryStore()
        self.thought_reservoir = ThoughtReservoir()
        self.turns_since_last_speak = 0
        self.last_turn_spoken = 0  # Track the last turn this agent spoke

        
    async def generate_thoughts(self, trigger_event: Dict) -> List['Thought']:
        """Generate new thoughts based on a trigger event.
        
        Args:
            trigger_event: Dictionary containing event details and context
            
        Returns:
            List of generated thoughts
        """
        pass
        
    async def evaluate_thoughts(self) -> List[Tuple[str, float]]:
        """Evaluate current thoughts and return their reasonings (text) and scores (float).
        
        Returns:
            List of tuples (reasoning, score) for each thought
        """
        pass
        
    async def select_and_participate(self, conversation_state: Dict) -> Optional['Utterance']:
        """Decide whether to participate and generate an utterance if appropriate.
        
        Args:
            conversation_state: Current state of the conversation
            
        Returns:
            Optional utterance if agent decides to participate
        """
        pass
        
    def update_turns_since_last_speak(self, current_turn: int) -> None:
        """Update the number of turns since the agent last spoke.
        
        Args:
            current_turn: The current turn number in the conversation
        """
        self.turns_since_last_speak = current_turn - self.last_turn_spoken
        self.last_turn_spoken = current_turn


