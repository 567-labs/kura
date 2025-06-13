"""Tests for MultiCheckpointManager functionality."""

import os
import tempfile
import shutil
import pytest

from kura.checkpoint import CheckpointManager
from kura.checkpoints import MultiCheckpointManager
from kura.types import ConversationSummary, Cluster


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    dirs = []
    for _ in range(3):
        dirs.append(tempfile.mkdtemp())
    yield dirs
    # Cleanup
    for dir_path in dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


@pytest.fixture
def sample_summaries():
    """Create sample conversation summaries for testing."""
    return [
        ConversationSummary(
            chat_id=f"conv_{i}",
            summary=f"Summary of conversation {i}",
            metadata={"source": "test"},
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_clusters():
    """Create sample clusters for testing."""
    return [
        Cluster(
            id=f"cluster_{i}",
            name=f"Cluster {i}",
            description=f"Summary of cluster {i}",
            chat_ids=[f"conv_{j}" for j in range(i, i + 3)],
            slug=f"cluster-{i}",
            parent_id=None,
        )
        for i in range(3)
    ]


class TestMultiCheckpointManager:
    """Test suite for MultiCheckpointManager."""

    def test_initialization(self, temp_dirs):
        """Test MultiCheckpointManager initialization."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]
        multi_mgr = MultiCheckpointManager(managers)

        assert multi_mgr.enabled
        assert len(multi_mgr.managers) == 2
        assert multi_mgr.save_strategy == "all_enabled"
        assert multi_mgr.load_strategy == "first_found"

    def test_initialization_with_disabled_managers(self, temp_dirs):
        """Test initialization with some disabled managers."""
        managers = [
            CheckpointManager(temp_dirs[0], enabled=True),
            CheckpointManager(temp_dirs[1], enabled=False),
            CheckpointManager(temp_dirs[2], enabled=True),
        ]
        multi_mgr = MultiCheckpointManager(managers)

        assert multi_mgr.enabled  # Should be enabled if any manager is enabled
        assert len(multi_mgr.managers) == 3

    def test_initialization_all_disabled(self, temp_dirs):
        """Test initialization when all managers are disabled."""
        managers = [CheckpointManager(d, enabled=False) for d in temp_dirs]
        multi_mgr = MultiCheckpointManager(managers)

        assert not multi_mgr.enabled

    def test_save_all_enabled_strategy(self, temp_dirs, sample_summaries):
        """Test saving with all_enabled strategy."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]
        multi_mgr = MultiCheckpointManager(managers, save_strategy="all_enabled")

        multi_mgr.save_checkpoint("summaries.jsonl", sample_summaries)

        # Check that both managers have the file
        for mgr in managers:
            assert os.path.exists(os.path.join(mgr.checkpoint_dir, "summaries.jsonl"))

    def test_save_primary_only_strategy(self, temp_dirs, sample_summaries):
        """Test saving with primary_only strategy."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]
        multi_mgr = MultiCheckpointManager(managers, save_strategy="primary_only")

        multi_mgr.save_checkpoint("summaries.jsonl", sample_summaries)

        # Only first manager should have the file
        assert os.path.exists(
            os.path.join(managers[0].checkpoint_dir, "summaries.jsonl")
        )
        assert not os.path.exists(
            os.path.join(managers[1].checkpoint_dir, "summaries.jsonl")
        )

    def test_load_first_found_strategy(self, temp_dirs, sample_summaries):
        """Test loading with first_found strategy."""
        managers = [CheckpointManager(d) for d in temp_dirs[:3]]

        # Save to second manager only
        managers[1].save_checkpoint("summaries.jsonl", sample_summaries)

        multi_mgr = MultiCheckpointManager(managers, load_strategy="first_found")
        loaded = multi_mgr.load_checkpoint("summaries.jsonl", ConversationSummary)

        assert loaded is not None
        assert len(loaded) == len(sample_summaries)
        assert loaded[0].chat_id == sample_summaries[0].chat_id

    def test_load_priority_strategy(self, temp_dirs, sample_summaries):
        """Test loading with priority strategy."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]

        # Save to second manager only
        managers[1].save_checkpoint("summaries.jsonl", sample_summaries)

        multi_mgr = MultiCheckpointManager(managers, load_strategy="priority")
        loaded = multi_mgr.load_checkpoint("summaries.jsonl", ConversationSummary)

        # Should return None since first manager doesn't have it
        assert loaded is None

    def test_list_checkpoints(self, temp_dirs, sample_summaries, sample_clusters):
        """Test listing checkpoints across all managers."""
        managers = [CheckpointManager(d) for d in temp_dirs[:3]]

        # Save different files to different managers
        managers[0].save_checkpoint("summaries.jsonl", sample_summaries)
        managers[1].save_checkpoint("clusters.jsonl", sample_clusters)
        managers[2].save_checkpoint("summaries.jsonl", sample_summaries)  # Duplicate

        multi_mgr = MultiCheckpointManager(managers)
        checkpoints = multi_mgr.list_checkpoints()

        # Should have unique filenames
        assert len(checkpoints) == 2
        assert "summaries.jsonl" in checkpoints
        assert "clusters.jsonl" in checkpoints

    def test_delete_checkpoint(self, temp_dirs, sample_summaries):
        """Test deleting checkpoints from all managers."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]
        multi_mgr = MultiCheckpointManager(managers, save_strategy="all_enabled")

        # Save to all managers
        multi_mgr.save_checkpoint("summaries.jsonl", sample_summaries)

        # Verify files exist
        for mgr in managers:
            assert os.path.exists(os.path.join(mgr.checkpoint_dir, "summaries.jsonl"))

        # Delete from all
        result = multi_mgr.delete_checkpoint("summaries.jsonl")
        assert result

        # Verify files are gone
        for mgr in managers:
            assert not os.path.exists(
                os.path.join(mgr.checkpoint_dir, "summaries.jsonl")
            )

    def test_get_stats(self, temp_dirs):
        """Test getting statistics about the multi-checkpoint setup."""
        managers = [
            CheckpointManager(temp_dirs[0], enabled=True),
            CheckpointManager(temp_dirs[1], enabled=False),
        ]
        multi_mgr = MultiCheckpointManager(
            managers, save_strategy="primary_only", load_strategy="priority"
        )

        stats = multi_mgr.get_stats()

        assert stats["enabled"] is True
        assert stats["save_strategy"] == "primary_only"
        assert stats["load_strategy"] == "priority"
        assert stats["num_managers"] == 2
        assert stats["enabled_managers"] == 1
        assert len(stats["managers"]) == 2
        assert stats["managers"][0]["enabled"] is True
        assert stats["managers"][1]["enabled"] is False

    def test_mixed_enabled_disabled_save(self, temp_dirs, sample_summaries):
        """Test saving with mix of enabled and disabled managers."""
        managers = [
            CheckpointManager(temp_dirs[0], enabled=True),
            CheckpointManager(temp_dirs[1], enabled=False),
            CheckpointManager(temp_dirs[2], enabled=True),
        ]
        multi_mgr = MultiCheckpointManager(managers, save_strategy="all_enabled")

        multi_mgr.save_checkpoint("summaries.jsonl", sample_summaries)

        # Only enabled managers should have files
        assert os.path.exists(os.path.join(temp_dirs[0], "summaries.jsonl"))
        assert not os.path.exists(os.path.join(temp_dirs[1], "summaries.jsonl"))
        assert os.path.exists(os.path.join(temp_dirs[2], "summaries.jsonl"))

    def test_empty_managers_list_error(self):
        """Test that empty managers list raises error."""
        with pytest.raises(ValueError, match="At least one checkpoint manager"):
            MultiCheckpointManager([])

    def test_invalid_strategies(self, temp_dirs):
        """Test that invalid strategies raise errors."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]

        # Valid initialization
        multi_mgr = MultiCheckpointManager(managers)

        # Test invalid save strategy
        multi_mgr.save_strategy = "invalid_strategy"
        with pytest.raises(ValueError, match="Unknown save strategy"):
            multi_mgr.save_checkpoint("test.jsonl", [])

        # Test invalid load strategy
        multi_mgr.save_strategy = "all_enabled"  # Reset
        multi_mgr.load_strategy = "invalid_strategy"
        with pytest.raises(ValueError, match="Unknown load strategy"):
            multi_mgr.load_checkpoint("test.jsonl", ConversationSummary)

    def test_invalid_save_strategy_during_init(self, temp_dirs):
        """Test that invalid save strategy during initialization raises error."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]

        with pytest.raises(ValueError, match="Invalid save_strategy 'invalid_save'"):
            MultiCheckpointManager(managers, save_strategy="invalid_save")

    def test_invalid_load_strategy_during_init(self, temp_dirs):
        """Test that invalid load strategy during initialization raises error."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]

        with pytest.raises(ValueError, match="Invalid load_strategy 'invalid_load'"):
            MultiCheckpointManager(managers, load_strategy="invalid_load")

    def test_both_invalid_strategies_during_init(self, temp_dirs):
        """Test that invalid strategies during initialization raise appropriate errors."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]

        # Should fail on save_strategy first (since it's validated first)
        with pytest.raises(ValueError, match="Invalid save_strategy 'bad_save'"):
            MultiCheckpointManager(
                managers, save_strategy="bad_save", load_strategy="bad_load"
            )
