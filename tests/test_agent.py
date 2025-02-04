"""Tests for the Agent class."""
import pytest
import pytest_asyncio
import numpy as np
from typing import Dict, Any

from thoughtful_agents.agent import Agent

@pytest.mark.asyncio
async def test_create_agent():
    """Test creating an agent with persona memories."""
    agent = await Agent.create(
        id="test_agent",
        name="Test Agent",
        persona="Test Agent Persona",
        config={
            'im_threshold': 0.5,
            'system1_prob': 0.5,
            'interrupt_threshold': 0.5,
            'proactive_tone': True
        }
    )
    assert isinstance(agent, Agent)
    assert agent.id == "test_agent"
    assert agent.name == "Test Agent"
    assert agent.persona == "Test Agent Persona"
    assert len(agent.memory_store.long_term_memory) == 1
    assert len(agent.memory_store.working_memory) == 0
    assert agent.memory_store.long_term_memory[0].id == "test_agent-long_term-0"
    assert agent.memory_store.long_term_memory[0].text == "Test Agent Persona"
    assert agent.memory_store.long_term_memory[0].weight == 1.0
    assert agent.memory_store.long_term_memory[0].embedding is not None
    assert agent.memory_store.long_term_memory[0].memory_type == "long_term"
    assert agent.memory_store.long_term_memory[0].turn_number == 0
    assert len(agent.thought_reservoir.thoughts) == 0

@pytest.mark.asyncio
async def test_create_agent_empty_persona():
    """Test that creating an agent with empty persona raises ValueError."""
    with pytest.raises(ValueError, match="Persona must be a non-empty string"):
        await Agent.create(
            id="test_agent",
            name="Test Agent",
            persona="",
            config={
                'im_threshold': 0.5,
                'system1_prob': 0.5,
                'interrupt_threshold': 0.5,
                'proactive_tone': True
            }
        )

    
