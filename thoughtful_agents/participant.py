"""Base classes for conversation participants."""
from typing import Dict, Optional
from abc import ABC, abstractmethod

class Participant(ABC):
    """Abstract base class for all conversation participants (AI agents and humans).
    
    This class defines the common interface that all participants must implement,
    ensuring they can be handled uniformly in conversations regardless of their type.
    
    Attributes:
        id: Unique identifier for the participant
        name: Display name of the participant
        type: Type of participant ("human" or "agent")
        turns_since_last_speak: Number of turns since participant last spoke
        last_turn_spoken: Last turn number when participant spoke
    """
    
    def __init__(
        self,
        id: str,
        name: str,
        type: str,
    ) -> None:
        """Initialize a participant.
        
        Args:
            id: Unique identifier
            name: Display name
            type: Type of participant ("human" or "agent")
        """
        self.id = id
        self.name = name
        self.type = type
        
        # Turn tracking (common to all participants)
        self.turns_since_last_speak = 0
        self.last_turn_spoken = 0
    
    def __repr__(self) -> str:
        """Return a technical string representation of the participant."""
        return f"""Participant(name='{self.name}', id='{self.id}', type='{self.type}')
State:
- turns_since_last_speak: {self.turns_since_last_speak}
- last_turn_spoken: {self.last_turn_spoken}"""