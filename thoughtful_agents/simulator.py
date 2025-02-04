"""Simulator module for running agent conversations."""
from typing import Dict, List, Optional
from .conversation import Conversation
from .agent import Agent    

class Simulator:
    """Manages the simulation of agent conversations."""

    # Create a simulator, takes in a conversation object
    def __init__(self, conversation: Conversation) -> None:
        self.conversation = conversation

    # Run a single turn of the conversation
    def run_turn(self) -> None:
        pass