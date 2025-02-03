"""Tests for the Agent class."""
import pytest
from thoughtful_agents.agent import Agent

@pytest.fixture
def test_agent():
    """Create a test agent fixture."""
    config = {
        "im_threshold": 0.5,
        "system1_prob": 0.3,
        "interrupt_threshold": 0.8,
        "proactive_tone": 0.6
    }
    return Agent(
        id="test_agent_1",
        name="Test Agent",
        config=config
    )

def test_agent_initialization(test_agent):
    """Test agent initialization."""
    assert test_agent.id == "test_agent_1"
    assert test_agent.name == "Test Agent"
    assert test_agent.config["im_threshold"] == 0.5

@pytest.mark.asyncio
async def test_generate_thoughts(test_agent):
    """Test thought generation."""
    pass

@pytest.mark.asyncio
async def test_evaluate_thoughts(test_agent):
    """Test thought evaluation."""
    pass

@pytest.mark.asyncio
async def test_select_and_participate(test_agent):
    """Test participation decision."""
    pass 