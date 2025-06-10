"""
Tests for MultiCheckpointManager functionality.
"""

import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch
from typing import List

from kura.v1 import CheckpointManager, MultiCheckpointManager
from kura.types import ConversationSummary, Conversation, Message
from datetime import datetime
from uuid import uuid4


class TestMultiCheckpointManager:
    """Test suite for MultiCheckpointManager."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories for testing
        self.temp_dirs = []
        for i in range(3):
            temp_dir = tempfile.mkdtemp(prefix=f"kura_test_checkpoint_{i}_")
            self.temp_dirs.append(temp_dir)
            
        # Create individual checkpoint managers
        self.managers = [
            CheckpointManager(temp_dir, enabled=True)
            for temp_dir in self.temp_dirs
        ]
        
        # Create test data
        self.test_conversations = [
            Conversation(
                id=str(uuid4()),
                created_at=datetime.now(),
                messages=[
                    Message(
                        created_at=str(datetime.now()),
                        role="user",
                        content=f"Test message {i}"
                    )
                ]
            ) for i in range(3)
        ]
        
        self.test_summaries = [
            ConversationSummary(
                conversation_id=conv.id,
                summary=f"Summary for conversation {i}",
                conversation=conv
            ) for i, conv in enumerate(self.test_conversations)
        ]

    def teardown_method(self):
        """Clean up test fixtures."""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_init_requires_managers(self):
        """Test that MultiCheckpointManager requires at least one manager."""
        with pytest.raises(ValueError, match="At least one checkpoint manager must be provided"):
            MultiCheckpointManager([])

    def test_init_sets_up_enabled_managers(self):
        """Test that initialization sets up directories for enabled managers."""
        multi_manager = MultiCheckpointManager(self.managers)
        
        # All directories should exist
        for temp_dir in self.temp_dirs:
            assert os.path.exists(temp_dir)

    def test_enabled_property(self):
        """Test the enabled property."""
        # All enabled
        multi_manager = MultiCheckpointManager(self.managers)
        assert multi_manager.enabled is True
        
        # Some disabled
        self.managers[1].enabled = False
        multi_manager = MultiCheckpointManager(self.managers)
        assert multi_manager.enabled is True
        
        # All disabled
        for manager in self.managers:
            manager.enabled = False
        multi_manager = MultiCheckpointManager(self.managers)
        assert multi_manager.enabled is False

    def test_checkpoint_dir_property(self):
        """Test the checkpoint_dir property returns first enabled manager's dir."""
        multi_manager = MultiCheckpointManager(self.managers)
        assert multi_manager.checkpoint_dir == self.temp_dirs[0]
        
        # Disable first manager
        self.managers[0].enabled = False
        multi_manager = MultiCheckpointManager(self.managers)
        assert multi_manager.checkpoint_dir == self.temp_dirs[1]

    def test_save_all_enabled_strategy(self):
        """Test saving to all enabled managers."""
        multi_manager = MultiCheckpointManager(
            self.managers,
            save_strategy="all_enabled"
        )
        
        # Save test data
        multi_manager.save_checkpoint("test.jsonl", self.test_summaries)
        
        # Verify all managers have the file
        for temp_dir in self.temp_dirs:
            test_file = os.path.join(temp_dir, "test.jsonl")
            assert os.path.exists(test_file)
            
            # Verify content
            with open(test_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == len(self.test_summaries)

    def test_save_primary_only_strategy(self):
        """Test saving to primary manager only."""
        multi_manager = MultiCheckpointManager(
            self.managers,
            save_strategy="primary_only"
        )
        
        # Save test data
        multi_manager.save_checkpoint("test.jsonl", self.test_summaries)
        
        # Verify only first manager has the file
        test_file_0 = os.path.join(self.temp_dirs[0], "test.jsonl")
        assert os.path.exists(test_file_0)
        
        # Other managers should not have the file
        for i in range(1, len(self.temp_dirs)):
            test_file = os.path.join(self.temp_dirs[i], "test.jsonl")
            assert not os.path.exists(test_file)

    def test_load_first_found_strategy(self):
        """Test loading with first_found strategy."""
        multi_manager = MultiCheckpointManager(
            self.managers,
            load_strategy="first_found"
        )
        
        # Save data to second manager only
        self.managers[1].save_checkpoint("test.jsonl", self.test_summaries)
        
        # Load should find it in second manager
        loaded = multi_manager.load_checkpoint("test.jsonl", ConversationSummary)
        assert loaded is not None
        assert len(loaded) == len(self.test_summaries)
        assert loaded[0].conversation_id == self.test_summaries[0].conversation_id

    def test_load_priority_strategy(self):
        """Test loading with priority strategy (same as first_found)."""
        multi_manager = MultiCheckpointManager(
            self.managers,
            load_strategy="priority"
        )
        
        # Save different data to different managers
        summary_1 = [self.test_summaries[0]]  # First summary only
        summary_2 = self.test_summaries[:2]   # First two summaries
        
        self.managers[0].save_checkpoint("test.jsonl", summary_1)
        self.managers[1].save_checkpoint("test.jsonl", summary_2)
        
        # Should load from first manager (priority order)
        loaded = multi_manager.load_checkpoint("test.jsonl", ConversationSummary)
        assert loaded is not None
        assert len(loaded) == 1  # Should get data from first manager
        assert loaded[0].conversation_id == self.test_summaries[0].conversation_id

    def test_load_returns_none_when_not_found(self):
        """Test that load returns None when checkpoint not found."""
        multi_manager = MultiCheckpointManager(self.managers)
        
        result = multi_manager.load_checkpoint("nonexistent.jsonl", ConversationSummary)
        assert result is None

    def test_save_error_handling_continue_on_error(self):
        """Test save error handling when not failing on errors."""
        # Create a manager that will fail (invalid directory)
        failing_manager = CheckpointManager("/invalid/path/that/does/not/exist", enabled=True)
        managers_with_failure = self.managers + [failing_manager]
        
        multi_manager = MultiCheckpointManager(
            managers_with_failure,
            save_strategy="all_enabled",
            fail_on_save_error=False
        )
        
        # Should not raise exception, but should log warnings
        with patch('kura.v1.kura.logger') as mock_logger:
            multi_manager.save_checkpoint("test.jsonl", self.test_summaries)
            
            # Should have error logs
            mock_logger.error.assert_called()
            mock_logger.warning.assert_called()

    def test_save_error_handling_fail_on_error(self):
        """Test save error handling when failing on errors."""
        # Create a manager that will fail
        failing_manager = CheckpointManager("/invalid/path/that/does/not/exist", enabled=True)
        managers_with_failure = self.managers + [failing_manager]
        
        multi_manager = MultiCheckpointManager(
            managers_with_failure,
            save_strategy="all_enabled",
            fail_on_save_error=True
        )
        
        # Should raise exception
        with pytest.raises(RuntimeError, match="Some checkpoint saves failed"):
            multi_manager.save_checkpoint("test.jsonl", self.test_summaries)

    def test_save_all_fail_raises_exception(self):
        """Test that save raises exception when all managers fail."""
        # Create managers that will all fail
        failing_managers = [
            CheckpointManager(f"/invalid/path/{i}", enabled=True)
            for i in range(2)
        ]
        
        multi_manager = MultiCheckpointManager(
            failing_managers,
            save_strategy="all_enabled",
            fail_on_save_error=False
        )
        
        # Should raise exception when all saves fail
        with pytest.raises(RuntimeError, match="Failed to save to any checkpoint manager"):
            multi_manager.save_checkpoint("test.jsonl", self.test_summaries)

    def test_load_error_handling(self):
        """Test load error handling when managers fail."""
        # Mock a manager to raise an exception
        with patch.object(self.managers[0], 'load_checkpoint', side_effect=Exception("Load failed")):
            multi_manager = MultiCheckpointManager(self.managers)
            
            # Save data to second manager
            self.managers[1].save_checkpoint("test.jsonl", self.test_summaries)
            
            # Should still succeed by loading from second manager
            with patch('kura.v1.kura.logger') as mock_logger:
                loaded = multi_manager.load_checkpoint("test.jsonl", ConversationSummary)
                
                assert loaded is not None
                assert len(loaded) == len(self.test_summaries)
                
                # Should have warning about first manager failure
                mock_logger.warning.assert_called()

    def test_disabled_manager_operations(self):
        """Test that disabled multi-manager doesn't perform operations."""
        # Disable all managers
        for manager in self.managers:
            manager.enabled = False
            
        multi_manager = MultiCheckpointManager(self.managers)
        
        # Operations should be no-ops
        multi_manager.save_checkpoint("test.jsonl", self.test_summaries)
        result = multi_manager.load_checkpoint("test.jsonl", ConversationSummary)
        
        assert result is None
        
        # No files should be created
        for temp_dir in self.temp_dirs:
            test_file = os.path.join(temp_dir, "test.jsonl")
            assert not os.path.exists(test_file)

    def test_invalid_strategies_raise_errors(self):
        """Test that invalid strategies raise appropriate errors."""
        multi_manager = MultiCheckpointManager(self.managers)
        
        # Invalid load strategy
        multi_manager.load_strategy = "invalid_strategy"
        with pytest.raises(ValueError, match="Unknown load strategy"):
            multi_manager.load_checkpoint("test.jsonl", ConversationSummary)
            
        # Invalid save strategy
        multi_manager.save_strategy = "invalid_strategy"
        with pytest.raises(ValueError, match="Unknown save strategy"):
            multi_manager.save_checkpoint("test.jsonl", self.test_summaries)

    def test_integration_with_pipeline_functions(self):
        """Test that MultiCheckpointManager works with pipeline functions."""
        # This is more of an integration test - we'll just verify the interface
        multi_manager = MultiCheckpointManager(self.managers)
        
        # Should have same interface as regular CheckpointManager
        assert hasattr(multi_manager, 'enabled')
        assert hasattr(multi_manager, 'checkpoint_dir')
        assert hasattr(multi_manager, 'setup_checkpoint_dir')
        assert hasattr(multi_manager, 'get_checkpoint_path')
        assert hasattr(multi_manager, 'load_checkpoint')
        assert hasattr(multi_manager, 'save_checkpoint')
        
        # These methods should be compatible with existing pipeline function signatures
        assert callable(multi_manager.load_checkpoint)
        assert callable(multi_manager.save_checkpoint)