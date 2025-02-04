"""
A Python framework for building proactive LLM agents that simulate human-like cognitive processes.
"""

__version__ = "0.1.0"

from .agent import Agent
from .conversation import Conversation, Utterance
from .memory import MemoryItem, MemoryStore
from .thought import Thought, ThoughtReservoir
from .utils.llm_api import get_completion, get_embedding, LLMAPIError
from .utils.text_splitter import SentenceTextSplitter

__all__ = [
    # Core classes
    'Agent',
    'Conversation',
    'Utterance',
    'MemoryItem',
    'MemoryStore',
    'Thought',
    'ThoughtReservoir',
    
    # Utilities
    'get_completion',
    'get_embedding',
    'LLMAPIError',
    'SentenceTextSplitter'
] 