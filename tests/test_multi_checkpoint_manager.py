"""Tests for MultiCheckpointManager functionality."""

import os
import tempfile
import shutil
from pathlib import Path
import pytest

from kura.checkpoint import CheckpointManager
from kura.checkpoints import MultiCheckpointManager
from kura.types import ConversationSummary, Cluster
from datetime import datetime


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    dirs = []
    for _ in range(3):
        dirs.append(tempfile.mkdtemp())
    yield dirs
    # Cleanup
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_summaries():
    """Create sample conversation summaries for testing."""
    return [
        ConversationSummary(
            chat_id=f"conv_{i}",
            summary=f"Test summary {i}",
            metadata={"created_at": datetime.now().isoformat()},
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
            description=f"Description for cluster {i}",
            slug=f"cluster_{i}_topic",
            chat_ids=[f"conv_{i}", f"conv_{i + 1}"],
            parent_id=None,
        )
        for i in range(3)
    ]


class TestMultiCheckpointManager:
    """Test suite for MultiCheckpointManager."""

    def test_initialization(self, temp_dirs):
        """Test MultiCheckpointManager initialization."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]
        multi_mgr = MultiCheckpointManager(managers=managers)

        assert len(multi_mgr.managers) == 2
        assert multi_mgr.save_strategy == "all_enabled"
        assert multi_mgr.load_strategy == "first_found"
        assert multi_mgr.enabled

    def test_initialization_with_no_managers(self):
        """Test initialization fails with no managers."""
        with pytest.raises(ValueError, match="At least one checkpoint manager"):
            MultiCheckpointManager(managers=[])

    def test_save_all_enabled_strategy(self, temp_dirs, sample_summaries):
        """Test saving to all enabled managers."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]
        multi_mgr = MultiCheckpointManager(
            managers=managers, save_strategy="all_enabled"
        )

        # Save checkpoint
        multi_mgr.save_checkpoint("summaries.jsonl", sample_summaries)

        # Verify saved to both managers
        for d in temp_dirs[:2]:
            checkpoint_path = Path(d) / "summaries.jsonl"
            assert checkpoint_path.exists()

            # Verify content
            with open(checkpoint_path) as f:
                lines = f.readlines()
                assert len(lines) == len(sample_summaries)

    def test_save_primary_only_strategy(self, temp_dirs, sample_summaries):
        """Test saving only to primary manager."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]
        multi_mgr = MultiCheckpointManager(
            managers=managers, save_strategy="primary_only"
        )

        # Save checkpoint
        multi_mgr.save_checkpoint("summaries.jsonl", sample_summaries)

        # Verify saved only to first manager
        checkpoint_path_1 = Path(temp_dirs[0]) / "summaries.jsonl"
        checkpoint_path_2 = Path(temp_dirs[1]) / "summaries.jsonl"

        assert checkpoint_path_1.exists()
        assert not checkpoint_path_2.exists()

    def test_load_first_found_strategy(self, temp_dirs, sample_summaries):
        """Test loading from first available manager."""
        managers = [CheckpointManager(d) for d in temp_dirs[:3]]

        # Save to second manager only
        managers[1].save_checkpoint("summaries.jsonl", sample_summaries)

        multi_mgr = MultiCheckpointManager(
            managers=managers, load_strategy="first_found"
        )

        # Load should find it in second manager
        loaded = multi_mgr.load_checkpoint("summaries.jsonl", ConversationSummary)

        assert loaded is not None
        assert len(loaded) == len(sample_summaries)
        assert loaded[0].summary == sample_summaries[0].summary

    def test_load_priority_strategy(self, temp_dirs, sample_summaries):
        """Test loading with priority strategy."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]

        # Save different data to each manager
        summaries_1 = sample_summaries[:3]
        summaries_2 = sample_summaries[3:]

        managers[0].save_checkpoint("summaries.jsonl", summaries_1)
        managers[1].save_checkpoint("summaries.jsonl", summaries_2)

        multi_mgr = MultiCheckpointManager(managers=managers, load_strategy="priority")

        # Should load from first manager
        loaded = multi_mgr.load_checkpoint("summaries.jsonl", ConversationSummary)

        assert loaded is not None
        assert len(loaded) == len(summaries_1)
        assert loaded[0].summary == summaries_1[0].summary

    def test_load_with_disabled_managers(self, temp_dirs, sample_summaries):
        """Test loading when some managers are disabled."""
        # First manager disabled, second enabled
        managers = [
            CheckpointManager(temp_dirs[0], enabled=False),
            CheckpointManager(temp_dirs[1], enabled=True),
        ]

        # Save to second manager
        managers[1].save_checkpoint("summaries.jsonl", sample_summaries)

        multi_mgr = MultiCheckpointManager(managers=managers)

        # Should skip disabled manager and load from second
        loaded = multi_mgr.load_checkpoint("summaries.jsonl", ConversationSummary)

        assert loaded is not None
        assert len(loaded) == len(sample_summaries)

    def test_save_with_failures(self, temp_dirs, sample_summaries):
        """Test saving continues when some managers fail."""
        # Create a manager that will fail on save by using a read-only directory
        import tempfile

        read_only_dir = tempfile.mkdtemp()
        os.chmod(read_only_dir, 0o444)  # Make directory read-only

        managers = [
            CheckpointManager(temp_dirs[0]),
            CheckpointManager(
                read_only_dir, enabled=False
            ),  # Disabled to avoid setup error
            CheckpointManager(temp_dirs[2]),
        ]

        # Enable the read-only manager after creation to test save failure
        managers[1].enabled = True

        multi_mgr = MultiCheckpointManager(
            managers=managers, save_strategy="all_enabled"
        )

        # Save should succeed for valid managers
        multi_mgr.save_checkpoint("summaries.jsonl", sample_summaries)

        # Verify saved to valid managers
        assert (Path(temp_dirs[0]) / "summaries.jsonl").exists()
        assert (Path(temp_dirs[2]) / "summaries.jsonl").exists()

        # Cleanup
        os.chmod(read_only_dir, 0o755)
        shutil.rmtree(read_only_dir)

    def test_list_checkpoints(self, temp_dirs, sample_summaries, sample_clusters):
        """Test listing checkpoints across managers."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]

        # Save different checkpoints to each manager
        managers[0].save_checkpoint("summaries.jsonl", sample_summaries)
        managers[1].save_checkpoint("clusters.jsonl", sample_clusters)

        # Both managers have one common checkpoint
        for mgr in managers:
            mgr.save_checkpoint("common.jsonl", sample_summaries)

        multi_mgr = MultiCheckpointManager(managers=managers)
        checkpoints = multi_mgr.list_checkpoints()

        # Should have unique checkpoints
        assert sorted(checkpoints) == [
            "clusters.jsonl",
            "common.jsonl",
            "summaries.jsonl",
        ]

    def test_delete_checkpoint(self, temp_dirs, sample_summaries):
        """Test deleting checkpoints from all managers."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]
        multi_mgr = MultiCheckpointManager(managers=managers)

        # Save to all managers
        multi_mgr.save_checkpoint("summaries.jsonl", sample_summaries)

        # Verify files exist
        for d in temp_dirs[:2]:
            assert (Path(d) / "summaries.jsonl").exists()

        # Delete checkpoint
        deleted = multi_mgr.delete_checkpoint("summaries.jsonl")
        assert deleted

        # Verify files are deleted
        for d in temp_dirs[:2]:
            assert not (Path(d) / "summaries.jsonl").exists()

    def test_disabled_multi_manager(self, temp_dirs, sample_summaries):
        """Test MultiCheckpointManager when all managers are disabled."""
        managers = [
            CheckpointManager(temp_dirs[0], enabled=False),
            CheckpointManager(temp_dirs[1], enabled=False),
        ]

        multi_mgr = MultiCheckpointManager(managers=managers)

        # Should be disabled
        assert not multi_mgr.enabled

        # Operations should be no-ops
        multi_mgr.save_checkpoint("summaries.jsonl", sample_summaries)
        loaded = multi_mgr.load_checkpoint("summaries.jsonl", ConversationSummary)

        assert loaded is None
        assert not (Path(temp_dirs[0]) / "summaries.jsonl").exists()
        assert not (Path(temp_dirs[1]) / "summaries.jsonl").exists()

    def test_mixed_checkpoint_types(self, temp_dirs, sample_summaries, sample_clusters):
        """Test handling different checkpoint types."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]
        multi_mgr = MultiCheckpointManager(managers=managers)

        # Save different types
        multi_mgr.save_checkpoint("summaries.jsonl", sample_summaries)
        multi_mgr.save_checkpoint("clusters.jsonl", sample_clusters)

        # Load different types
        loaded_summaries = multi_mgr.load_checkpoint(
            "summaries.jsonl", ConversationSummary
        )
        loaded_clusters = multi_mgr.load_checkpoint("clusters.jsonl", Cluster)

        assert len(loaded_summaries) == len(sample_summaries)
        assert len(loaded_clusters) == len(sample_clusters)
        assert loaded_clusters[0].name == sample_clusters[0].name

    def test_repr(self, temp_dirs):
        """Test string representation."""
        managers = [CheckpointManager(d) for d in temp_dirs[:2]]
        multi_mgr = MultiCheckpointManager(
            managers=managers, save_strategy="primary_only", load_strategy="priority"
        )

        repr_str = repr(multi_mgr)
        assert "MultiCheckpointManager" in repr_str
        assert "CheckpointManager" in repr_str
        assert "primary_only" in repr_str
        assert "priority" in repr_str
