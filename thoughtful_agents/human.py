"""Human participant implementation."""
from typing import Dict, Optional
import asyncio
import uuid
from .participant import Participant

class Human(Participant):
    """Represents a human participant in a conversation.
    
    This class handles human input in conversations, allowing humans to
    interact with AI agents naturally.
    """
    
    def __init__(
        self,
        name: str,
    ) -> None:
        """Initialize a human participant."""
        # Generate unique ID using UUID
        human_id = f"human-{uuid.uuid4()}"
        
        super().__init__(id=human_id, name=name, type="human")
    
    @classmethod
    def create(
        cls,
        name: str,
    ) -> "Human":
        """Create a new human participant.
        
        Args:
            name: Display name
            
        Returns:
            Initialized Human instance with auto-generated ID
        """
        return cls(name=name)
    
    async def can_speak(self, conversation_state: Dict) -> bool:
        """Humans can always choose to speak.
        
        Args:
            conversation_state: Current state of the conversation
            
        Returns:
            Always True for humans
        """
        return True
    
    async def get_response(self, conversation_state: Dict) -> Optional[str]:
        """Get response from human input.
        
        Args:
            conversation_state: Current state of the conversation
            
        Returns:
            Human's input text or None if they choose not to speak
        """
        # Get input from human (can be customized based on UI/interface)
        response = input(f"{self.name}, your response (press Enter to skip): ").strip()
        
        if response:
            self.turns_since_last_speak = 0
            self.last_turn_spoken = conversation_state.get("current_turn", 0)
        else:
            self.turns_since_last_speak += 1
            
        return response if response else None