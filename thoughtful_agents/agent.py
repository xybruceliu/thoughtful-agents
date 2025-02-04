"""Agent module containing the core Agent class."""
from typing import Dict, List, Optional, Tuple
import logging
import uuid
from .memory import MemoryStore, MemoryItem
from .thought import ThoughtReservoir, Thought
from .utils import SentenceTextSplitter, get_embedding, LLMAPIError
from .participant import Participant

logger = logging.getLogger(__name__)

class Agent(Participant):
    """An AI agent that can generate thoughts and participate in conversations.
    
    The agent maintains its own memory store for long-term and working memories,
    can generate and evaluate thoughts, and make decisions about when to participate
    in conversations.
    
    Attributes:
        id: Auto-generated unique identifier for the agent (format: "agent-{uuid}")
        name: Display name of the agent
        persona: Persona description that defines the agent's characteristics
        config: Configuration containing behavior thresholds
        memory_store: Storage for working and long-term memories
        thought_reservoir: Storage for generated thoughts
        turns_since_last_speak: Number of turns since agent last spoke
        last_turn_spoken: Last turn number when agent spoke
    """
    
    # Class variable to track number of agents created
    _agent_counter = 0
    
    def __init__(
        self,
        name: str,
        persona: str,
        config: Dict,
        memory_store: MemoryStore,
        thought_reservoir: ThoughtReservoir
    ) -> None:
        """Initialize an agent. Do not use directly, use create() instead."""
        # Generate unique ID using UUID
        agent_id = f"agent-{uuid.uuid4()}"
        
        super().__init__(id=agent_id, name=name, type="agent")
        self.persona = persona.strip()
        self.config = config
        self.memory_store = memory_store
        self.thought_reservoir = thought_reservoir
    
        
    def __repr__(self) -> str:
        """Return a technical string representation of the agent for debugging."""
        # Get memory statistics
        working_memories = len(self.memory_store.working_memory)
        long_term_memories = len(self.memory_store.long_term_memory)
        thoughts = len(self.thought_reservoir.thoughts)
        
        # Format config values
        config_str = "\n".join(f"- {k}: {v}" for k, v in self.config.items())
        
        # Format persona (dedent and normalize whitespace)
        persona_lines = [line.strip() for line in self.persona.split('\n')]
        persona_str = ' '.join(line for line in persona_lines if line)
        
        # Get base representation
        base_repr = super().__repr__()
        
        return f"""{base_repr}
Persona: {persona_str}
Config:
{config_str}
Memory Counts:
- working_memories: {working_memories}
- long_term_memories: {long_term_memories}
- thoughts: {thoughts}"""
        
    @classmethod
    async def create(
        agent_cls,  # This is the class itself (Agent)
        name: str,
        persona: str,
        config: Dict
    ) -> "Agent":
        """Create a new agent with initialized memories.
        
        This is a factory method that handles memory initialization and returns a fully
        initialized Agent instance.
        
        Args:
            name: Display name of the agent
            persona: Persona description that defines the agent's characteristics
            config: Configuration containing behavior thresholds:
                   - im_threshold: Threshold for intrinsic motivation
                   - system1_prob: Probability of using system 1 thinking
                   - interrupt_threshold: Threshold for interrupting conversations
                   - proactive_tone: True or False
            
        Returns:
            Initialized Agent instance with auto-generated ID
            
        Raises:
            ValueError: If persona is empty or invalid
            LLMAPIError: If embedding generation fails
            RuntimeError: If memory initialization fails
        """
        if not isinstance(persona, str) or not persona.strip():
            raise ValueError("Persona must be a non-empty string")
            
        # Initialize components
        memory_store = MemoryStore()
        thought_reservoir = ThoughtReservoir()
        
        # Create agent instance (ID will be auto-generated in __init__)
        agent = agent_cls(
            name=name,
            persona=persona,
            config=config,
            memory_store=memory_store,
            thought_reservoir=thought_reservoir
        )
        
        # Initialize persona memories to memory store
        try:
            # Split persona into chunks
            chunks = SentenceTextSplitter().split_text(agent.persona)
            if not chunks:
                raise ValueError("No valid chunks extracted from persona text")
                
            # Store each chunk as a long-term memory
            for i in range(len(chunks)):
                try:
                    # Add memory item with embedding
                    await agent.memory_store.add_memory(
                        agent_id=agent.id,
                        text=chunks[i],
                        weight=1.0,  # Default weight for persona memories
                        memory_type="long_term",
                        turn_number=0
                    )

                except LLMAPIError:
                    logger.error("Failed to create memory for chunk: %s", chunks[i])
                    raise
                except Exception as e:
                    logger.error("Failed to process chunk: %s", str(e))
                    raise RuntimeError(f"Failed to process persona chunk: {str(e)}") from e
                    
            logger.info("Initialized %d persona memories for agent %s", len(chunks), agent.id)
            return agent
            
        except Exception as e:
            logger.error("Failed to initialize agent: %s", str(e))
            raise
        
    # Retrieve the most relevant memories from memory store based on saliency at this 
    async def retrieve_memories(self, turn_number: int) -> List[MemoryItem]:
        pass

    # Thought related methods
    async def generate_thoughts(self, trigger_event: Dict) -> List[Thought]:
        """Generate new thoughts based on a trigger event.
        
        Args:
            trigger_event: Dictionary containing:
                         - type: Type of trigger ("utterance", "memory", etc.)
                         - content: Text content to process
                         - context: Additional context information
            
        Returns:
            List of generated thoughts
        """
        # TODO: Implement thought generation logic
        pass
        
    async def evaluate_thoughts(self) -> List[Tuple[str, float]]:
        """Evaluate current thoughts and return their reasonings and scores.
        
        Returns:
            List of tuples (reasoning, score) for each thought
        """
        # TODO: Implement thought evaluation logic
        pass