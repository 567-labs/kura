"""Tests for MultiCheckpointManager functionality."""

import os
import tempfile
import pytest
from typing import List
from pydantic import BaseModel
from unittest.mock import Mock, patch

from kura.v1 import CheckpointManager, MultiCheckpointManager


class TestModel(BaseModel):
    """Test model for checkpoint testing."""
    id: str
    value: int


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as temp1, tempfile.TemporaryDirectory() as temp2:
        yield temp1, temp2


@pytest.fixture
def checkpoint_managers(temp_dirs):
    """Create test checkpoint managers."""
    temp1, temp2 = temp_dirs
    manager1 = CheckpointManager(temp1, enabled=True)
    manager2 = CheckpointManager(temp2, enabled=True)
    return manager1, manager2


@pytest.fixture
def sample_data():
    """Sample test data."""
    return [
        TestModel(id="1", value=10),
        TestModel(id="2", value=20),
        TestModel(id="3", value=30)
    ]


class TestMultiCheckpointManagerInit:
    """Test MultiCheckpointManager initialization."""
    
    def test_empty_managers_raises_error(self):
        """Test that empty managers list raises ValueError."""
        with pytest.raises(ValueError, match="At least one checkpoint manager must be provided"):
            MultiCheckpointManager([])
    
    def test_valid_initialization(self, checkpoint_managers):
        """Test valid initialization."""
        manager1, manager2 = checkpoint_managers
        multi_manager = MultiCheckpointManager([manager1, manager2])
        
        assert len(multi_manager.managers) == 2
        assert multi_manager.load_strategy == "first_found"
        assert multi_manager.save_strategy == "all_enabled"
    
    def test_custom_strategies(self, checkpoint_managers):
        """Test initialization with custom strategies."""
        manager1, manager2 = checkpoint_managers
        multi_manager = MultiCheckpointManager(
            [manager1, manager2],
            load_strategy="priority",
            save_strategy="primary_only"
        )
        
        assert multi_manager.load_strategy == "priority"
        assert multi_manager.save_strategy == "primary_only"
    
    def test_invalid_load_strategy(self, checkpoint_managers):
        """Test invalid load strategy raises error."""
        manager1, manager2 = checkpoint_managers
        with pytest.raises(ValueError, match="Invalid load_strategy"):
            MultiCheckpointManager([manager1, manager2], load_strategy="invalid")
    
    def test_invalid_save_strategy(self, checkpoint_managers):
        """Test invalid save strategy raises error."""
        manager1, manager2 = checkpoint_managers
        with pytest.raises(ValueError, match="Invalid save_strategy"):
            MultiCheckpointManager([manager1, manager2], save_strategy="invalid")


class TestMultiCheckpointManagerProperties:
    """Test MultiCheckpointManager properties."""
    
    def test_enabled_with_all_enabled(self, checkpoint_managers):
        """Test enabled property when all managers are enabled."""
        manager1, manager2 = checkpoint_managers
        multi_manager = MultiCheckpointManager([manager1, manager2])
        
        assert multi_manager.enabled is True
    
    def test_enabled_with_some_disabled(self, temp_dirs):
        """Test enabled property when some managers are disabled."""
        temp1, temp2 = temp_dirs
        manager1 = CheckpointManager(temp1, enabled=True)
        manager2 = CheckpointManager(temp2, enabled=False)
        multi_manager = MultiCheckpointManager([manager1, manager2])
        
        assert multi_manager.enabled is True
    
    def test_enabled_with_all_disabled(self, temp_dirs):
        """Test enabled property when all managers are disabled."""
        temp1, temp2 = temp_dirs
        manager1 = CheckpointManager(temp1, enabled=False)
        manager2 = CheckpointManager(temp2, enabled=False)
        multi_manager = MultiCheckpointManager([manager1, manager2])
        
        assert multi_manager.enabled is False
    
    def test_get_checkpoint_path(self, checkpoint_managers):
        """Test get_checkpoint_path returns path from first manager."""
        manager1, manager2 = checkpoint_managers
        multi_manager = MultiCheckpointManager([manager1, manager2])
        
        path = multi_manager.get_checkpoint_path("test.jsonl")
        expected = manager1.get_checkpoint_path("test.jsonl")
        
        assert path == expected


class TestMultiCheckpointManagerLoad:
    """Test MultiCheckpointManager load functionality."""
    
    def test_load_disabled_returns_none(self, temp_dirs):
        """Test load returns None when all managers are disabled."""
        temp1, temp2 = temp_dirs
        manager1 = CheckpointManager(temp1, enabled=False)
        manager2 = CheckpointManager(temp2, enabled=False)
        multi_manager = MultiCheckpointManager([manager1, manager2])
        
        result = multi_manager.load_checkpoint("test.jsonl", TestModel)
        assert result is None
    
    def test_load_first_found_strategy(self, checkpoint_managers, sample_data):
        """Test load with first_found strategy."""
        manager1, manager2 = checkpoint_managers
        
        # Save data to second manager only
        manager2.save_checkpoint("test.jsonl", sample_data)
        
        multi_manager = MultiCheckpointManager([manager1, manager2], load_strategy="first_found")
        result = multi_manager.load_checkpoint("test.jsonl", TestModel)
        
        assert result is not None
        assert len(result) == 3
        assert result[0].id == "1"
    
    def test_load_priority_strategy(self, checkpoint_managers, sample_data):
        """Test load with priority strategy."""
        manager1, manager2 = checkpoint_managers
        
        # Save different data to both managers
        data1 = [TestModel(id="priority", value=100)]
        data2 = sample_data
        
        manager1.save_checkpoint("test.jsonl", data1)
        manager2.save_checkpoint("test.jsonl", data2)
        
        multi_manager = MultiCheckpointManager([manager1, manager2], load_strategy="priority")
        result = multi_manager.load_checkpoint("test.jsonl", TestModel)
        
        # Should load from first manager (priority order)
        assert result is not None
        assert len(result) == 1
        assert result[0].id == "priority"
    
    def test_load_with_errors(self, checkpoint_managers, sample_data):
        """Test load behavior when some managers fail."""
        manager1, manager2 = checkpoint_managers
        
        # Save data to second manager only
        manager2.save_checkpoint("test.jsonl", sample_data)
        
        # Mock first manager to raise exception
        with patch.object(manager1, 'load_checkpoint', side_effect=Exception("Test error")):
            multi_manager = MultiCheckpointManager([manager1, manager2])
            result = multi_manager.load_checkpoint("test.jsonl", TestModel)
            
            # Should still load from second manager
            assert result is not None
            assert len(result) == 3
    
    def test_load_no_data_found(self, checkpoint_managers):
        """Test load when no data is found in any manager."""
        manager1, manager2 = checkpoint_managers
        multi_manager = MultiCheckpointManager([manager1, manager2])
        
        result = multi_manager.load_checkpoint("nonexistent.jsonl", TestModel)
        assert result is None


class TestMultiCheckpointManagerSave:
    """Test MultiCheckpointManager save functionality."""
    
    def test_save_disabled_does_nothing(self, temp_dirs, sample_data):
        """Test save does nothing when all managers are disabled."""
        temp1, temp2 = temp_dirs
        manager1 = CheckpointManager(temp1, enabled=False)
        manager2 = CheckpointManager(temp2, enabled=False)
        multi_manager = MultiCheckpointManager([manager1, manager2])
        
        # Should not raise any exceptions
        multi_manager.save_checkpoint("test.jsonl", sample_data)
        
        # Verify no files were created
        assert not os.path.exists(os.path.join(temp1, "test.jsonl"))
        assert not os.path.exists(os.path.join(temp2, "test.jsonl"))
    
    def test_save_all_enabled_strategy(self, checkpoint_managers, sample_data):
        """Test save with all_enabled strategy."""
        manager1, manager2 = checkpoint_managers
        multi_manager = MultiCheckpointManager([manager1, manager2], save_strategy="all_enabled")
        
        multi_manager.save_checkpoint("test.jsonl", sample_data)
        
        # Verify data was saved to both managers
        result1 = manager1.load_checkpoint("test.jsonl", TestModel)
        result2 = manager2.load_checkpoint("test.jsonl", TestModel)
        
        assert result1 is not None and len(result1) == 3
        assert result2 is not None and len(result2) == 3
    
    def test_save_primary_only_strategy(self, checkpoint_managers, sample_data):
        """Test save with primary_only strategy."""
        manager1, manager2 = checkpoint_managers
        multi_manager = MultiCheckpointManager([manager1, manager2], save_strategy="primary_only")
        
        multi_manager.save_checkpoint("test.jsonl", sample_data)
        
        # Verify data was saved to first manager only
        result1 = manager1.load_checkpoint("test.jsonl", TestModel)
        result2 = manager2.load_checkpoint("test.jsonl", TestModel)
        
        assert result1 is not None and len(result1) == 3
        assert result2 is None
    
    def test_save_with_some_disabled(self, temp_dirs, sample_data):
        """Test save behavior with some managers disabled."""
        temp1, temp2 = temp_dirs
        manager1 = CheckpointManager(temp1, enabled=True)
        manager2 = CheckpointManager(temp2, enabled=False)
        multi_manager = MultiCheckpointManager([manager1, manager2], save_strategy="all_enabled")
        
        multi_manager.save_checkpoint("test.jsonl", sample_data)
        
        # Verify data was saved to enabled manager only
        result1 = manager1.load_checkpoint("test.jsonl", TestModel)
        assert result1 is not None and len(result1) == 3
        assert not os.path.exists(os.path.join(temp2, "test.jsonl"))
    
    def test_save_with_errors(self, checkpoint_managers, sample_data):
        """Test save behavior when some managers fail."""
        manager1, manager2 = checkpoint_managers
        multi_manager = MultiCheckpointManager([manager1, manager2], save_strategy="all_enabled")
        
        # Mock first manager to raise exception
        with patch.object(manager1, 'save_checkpoint', side_effect=Exception("Test error")):
            multi_manager.save_checkpoint("test.jsonl", sample_data)
            
            # Should still save to second manager
            result2 = manager2.load_checkpoint("test.jsonl", TestModel)
            assert result2 is not None and len(result2) == 3


class TestMultiCheckpointManagerIntegration:
    """Integration tests for MultiCheckpointManager."""
    
    def test_full_workflow(self, checkpoint_managers, sample_data):
        """Test complete save and load workflow."""
        manager1, manager2 = checkpoint_managers
        multi_manager = MultiCheckpointManager([manager1, manager2])
        
        # Save data
        multi_manager.save_checkpoint("workflow.jsonl", sample_data)
        
        # Create new multi-manager to simulate fresh load
        new_multi_manager = MultiCheckpointManager([manager1, manager2])
        result = new_multi_manager.load_checkpoint("workflow.jsonl", TestModel)
        
        assert result is not None
        assert len(result) == 3
        assert [item.id for item in result] == ["1", "2", "3"]
        assert [item.value for item in result] == [10, 20, 30]
    
    def test_redundancy_scenario(self, checkpoint_managers, sample_data):
        """Test redundancy scenario where one checkpoint becomes corrupted."""
        manager1, manager2 = checkpoint_managers
        multi_manager = MultiCheckpointManager([manager1, manager2])
        
        # Save data to both managers
        multi_manager.save_checkpoint("redundancy.jsonl", sample_data)
        
        # Corrupt first manager's data
        corrupted_path = manager1.get_checkpoint_path("redundancy.jsonl")
        with open(corrupted_path, "w") as f:
            f.write("corrupted data\n")
        
        # Should still load from second manager
        result = multi_manager.load_checkpoint("redundancy.jsonl", TestModel)
        assert result is not None
        assert len(result) == 3