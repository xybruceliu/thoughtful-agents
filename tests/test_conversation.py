"""Tests for the Conversation and Utterance classes."""
import pytest
from datetime import datetime
from thoughtful_agents.conversation import Conversation, Utterance
from thoughtful_agents.agent import Agent

@pytest.fixture
def test_utterance():
    """Create a test utterance fixture."""
    return Utterance(
        id="test_utterance_1",
        agent_id="test_agent_1",
        content={"text": "Hello", "embedding": [0.1, 0.2, 0.3]}
    )

@pytest.fixture
def test_agents():
    """Create test agents fixture."""
    config = {"im_threshold": 0.5}
    return [
        Agent(id="agent_1", name="Agent 1", config=config),
        Agent(id="agent_2", name="Agent 2", config=config)
    ]

@pytest.fixture
def test_conversation(test_agents):
    """Create a test conversation fixture."""
    return Conversation(agents=test_agents)

def test_utterance_initialization(test_utterance):
    """Test utterance initialization."""
    assert test_utterance.id == "test_utterance_1"
    assert test_utterance.agent_id == "test_agent_1"
    assert "text" in test_utterance.content

def test_conversation_initialization(test_conversation):
    """Test conversation initialization."""
    assert len(test_conversation.agents) == 2
    assert len(test_conversation.utterances) == 0

def test_add_utterance(test_conversation, test_utterance):
    """Test adding utterance to conversation."""
    pass

def test_get_context(test_conversation):
    """Test retrieving conversation context."""
    pass

def test_get_current_speaker(test_conversation):
    """Test current speaker determination."""
    pass 