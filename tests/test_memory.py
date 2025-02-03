"""Tests for the MemoryItem and MemoryStore classes."""
import pytest
from datetime import datetime, timedelta
from thoughtful_agents.memory import MemoryItem, MemoryStore

@pytest.fixture
def test_memory_item():
    """Create a test memory item fixture."""
    return MemoryItem(
        id="memory_1",
        content={"text": "Test memory", "embedding": [0.1, 0.2, 0.3]},
        weight=1.0,
        memory_type="working"
    )

@pytest.fixture
def test_memory_store():
    """Create a test memory store fixture."""
    return MemoryStore()

def test_memory_item_initialization(test_memory_item):
    """Test memory item initialization."""
    assert test_memory_item.id == "memory_1"
    assert test_memory_item.weight == 1.0
    assert test_memory_item.memory_type == "working"
    assert test_memory_item.saliency == 1.0

def test_memory_store_initialization(test_memory_store):
    """Test memory store initialization."""
    assert len(test_memory_store.working) == 0
    assert len(test_memory_store.long_term) == 0

def test_add_memory(test_memory_store, test_memory_item):
    """Test adding memory to store."""
    pass

def test_retrieve_memories(test_memory_store):
    """Test memory retrieval."""
    pass

def test_update_saliency(test_memory_store, test_memory_item):
    """Test saliency updates."""
    pass

def test_memory_decay(test_memory_item):
    """Test memory decay over time."""
    pass 