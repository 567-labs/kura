"""
Unified checkpoint tests using pytest parametrize.

This module uses pytest.mark.parametrize to run the same tests across all
available checkpoint managers, ensuring consistent behavior and reducing
code duplication.
"""

import pytest
import tempfile
import os
from typing import List
from datetime import datetime

from kura.checkpoints.jsonl import JSONLCheckpointManager
from kura.types import Conversation, ConversationSummary, Cluster, Message
from kura.types.dimensionality import ProjectedCluster

# Optional imports
try:
    from kura.checkpoints.parquet import ParquetCheckpointManager

    PARQUET_AVAILABLE = True
except ImportError:
    ParquetCheckpointManager = None
    PARQUET_AVAILABLE = False

try:
    from kura.checkpoints.hf_dataset import HFDatasetCheckpointManager

    HF_AVAILABLE = True
except ImportError:
    HFDatasetCheckpointManager = None
    HF_AVAILABLE = False


# =============================================================================
# Test Data Factories
# =============================================================================


@pytest.fixture
def sample_conversations():
    """Create sample conversations for testing."""
    messages = [
        Message(
            role="user", content="Hello, can you help me?", created_at=datetime.now()
        ),
        Message(
            role="assistant",
            content="Of course! How can I help?",
            created_at=datetime.now(),
        ),
    ]

    return [
        Conversation(
            chat_id="chat_1",
            created_at=datetime.now(),
            messages=messages,
            metadata={"source": "test"},
        ),
        Conversation(
            chat_id="chat_2",
            created_at=datetime.now(),
            messages=messages,
            metadata={"source": "test", "category": "support"},
        ),
    ]


@pytest.fixture
def sample_summaries():
    """Create sample conversation summaries for testing."""
    return [
        ConversationSummary(
            chat_id="chat_1",
            summary="User asks for help",
            request="Help request",
            topic="support",
            languages=["english"],
            task="Provide assistance",
            concerning_score=1,
            user_frustration=1,
            assistant_errors=None,
            metadata={"turns": 2},
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        ),
        ConversationSummary(
            chat_id="chat_2",
            summary="Follow-up question",
            request=None,
            topic="support",
            languages=["english"],
            task=None,
            concerning_score=None,
            user_frustration=None,
            assistant_errors=["Minor issue"],
            metadata={"turns": 1},
            embedding=[0.6, 0.7, 0.8, 0.9, 1.0],
        ),
    ]


@pytest.fixture
def sample_clusters():
    """Create sample clusters for testing."""
    return [
        Cluster(
            id="cluster_1",
            name="Support Questions",
            description="General support questions",
            slug="support_questions",
            chat_ids=["chat_1", "chat_2"],
            parent_id=None,
        ),
        Cluster(
            id="cluster_2",
            name="Technical Issues",
            description="Technical problem reports",
            slug="technical_issues",
            chat_ids=["chat_3"],
            parent_id="cluster_1",
        ),
    ]


@pytest.fixture
def sample_projected_clusters():
    """Create sample projected clusters for testing."""
    return [
        ProjectedCluster(
            id="proj_1",
            name="Support Cluster",
            description="Support-related conversations",
            slug="support_cluster",
            chat_ids=["chat_1", "chat_2"],
            parent_id=None,
            x_coord=1.0,
            y_coord=2.0,
            level=0,
        ),
        ProjectedCluster(
            id="proj_2",
            name="Tech Cluster",
            description="Technical conversations",
            slug="tech_cluster",
            chat_ids=["chat_3"],
            parent_id="proj_1",
            x_coord=-1.0,
            y_coord=1.0,
            level=1,
        ),
    ]


# =============================================================================
# Manager Fixtures and Parameters
# =============================================================================


def create_manager_params():
    """Create pytest parameters for all available checkpoint managers."""
    params = []

    # JSONL - always available
    params.append(pytest.param("jsonl", JSONLCheckpointManager, {}, id="jsonl"))

    # Parquet - if available
    if PARQUET_AVAILABLE:
        params.append(
            pytest.param("parquet", ParquetCheckpointManager, {}, id="parquet")
        )

    # HuggingFace - if available
    if HF_AVAILABLE:
        params.append(
            pytest.param("hf", HFDatasetCheckpointManager, {}, id="hf_dataset")
        )

    return params


@pytest.fixture
def checkpoint_manager(request):
    """Create a checkpoint manager instance for testing."""
    manager_name, manager_class, manager_kwargs = request.param

    with tempfile.TemporaryDirectory() as temp_dir:
        manager = manager_class(temp_dir, enabled=True, **manager_kwargs)
        manager._name = manager_name  # Store name for test logic
        yield manager


# =============================================================================
# Helper Functions
# =============================================================================


def get_checkpoint_params(manager_name: str, filename: str, data: List, model_class):
    """Get appropriate save/load parameters based on manager type."""
    if manager_name == "hf":
        # HuggingFace requires checkpoint_type
        type_map = {
            Conversation: "conversations",
            ConversationSummary: "summaries",
            Cluster: "clusters",
            ProjectedCluster: "projected_clusters",
        }
        checkpoint_type = type_map.get(model_class, "unknown")

        return {
            "save_params": (filename, data),
            "save_kwargs": {"checkpoint_type": checkpoint_type},
            "load_params": (filename, model_class),
            "load_kwargs": {"checkpoint_type": checkpoint_type},
        }
    else:
        # JSONL and Parquet use filename approach
        file_ext = ".jsonl" if manager_name == "jsonl" else ".parquet"
        full_filename = f"{filename}{file_ext}"

        return {
            "save_params": (full_filename, data),
            "load_params": (full_filename, model_class),
            "load_kwargs": {},
        }


def assert_data_equal(original: List, loaded: List, data_type: str):
    """Assert that original and loaded data are equal, handling type-specific comparisons."""
    assert len(loaded) == len(original)

    for orig, load in zip(original, loaded):
        if data_type == "conversations":
            assert orig.chat_id == load.chat_id
            assert orig.metadata == load.metadata
            assert len(orig.messages) == len(load.messages)

        elif data_type == "summaries":
            assert orig.chat_id == load.chat_id
            assert orig.summary == load.summary
            assert orig.request == load.request
            assert orig.topic == load.topic
            assert orig.languages == load.languages
            assert orig.task == load.task
            assert orig.concerning_score == load.concerning_score
            assert orig.user_frustration == load.user_frustration
            assert orig.assistant_errors == load.assistant_errors
            assert orig.metadata == load.metadata

            # Handle float precision for embeddings
            if orig.embedding and load.embedding:
                assert len(orig.embedding) == len(load.embedding)
                for o_val, l_val in zip(orig.embedding, load.embedding):
                    assert abs(o_val - l_val) < 1e-5, (
                        f"Embedding diff: {o_val} vs {l_val}"
                    )

        elif data_type == "clusters":
            assert orig.id == load.id
            assert orig.name == load.name
            assert orig.description == load.description
            assert orig.slug == load.slug
            assert orig.chat_ids == load.chat_ids
            assert orig.parent_id == load.parent_id

        elif data_type == "projected_clusters":
            assert orig.id == load.id
            assert orig.name == load.name
            assert orig.description == load.description
            assert orig.slug == load.slug
            assert orig.chat_ids == load.chat_ids
            assert orig.parent_id == load.parent_id
            assert orig.level == load.level

            # Handle float precision for coordinates
            assert abs(orig.x_coord - load.x_coord) < 1e-5, (
                f"X coord diff: {orig.x_coord} vs {load.x_coord}"
            )
            assert abs(orig.y_coord - load.y_coord) < 1e-5, (
                f"Y coord diff: {orig.y_coord} vs {load.y_coord}"
            )


# =============================================================================
# Unified Tests Using Parametrize
# =============================================================================


@pytest.mark.parametrize(
    "manager_name,manager_class,manager_kwargs", create_manager_params()
)
class TestCheckpointManagerUnified:
    """Unified tests that run across all checkpoint managers."""

    def test_manager_initialization(self, manager_name, manager_class, manager_kwargs):
        """Test that the manager initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = manager_class(temp_dir, enabled=True, **manager_kwargs)
            assert manager.enabled
            assert hasattr(manager, "save_checkpoint")
            assert hasattr(manager, "load_checkpoint")

    def test_disabled_manager(
        self, manager_name, manager_class, manager_kwargs, sample_summaries
    ):
        """Test that disabled managers don't save/load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = manager_class(temp_dir, enabled=False, **manager_kwargs)

            params = get_checkpoint_params(
                manager_name, "test", sample_summaries, ConversationSummary
            )

            # Save should do nothing
            save_kwargs = params.get("save_kwargs", {})
            manager.save_checkpoint(*params["save_params"], **save_kwargs)

            # Load should return None
            loaded = manager.load_checkpoint(
                *params["load_params"], **params["load_kwargs"]
            )
            assert loaded is None

    def test_conversations_roundtrip(
        self, manager_name, manager_class, manager_kwargs, sample_conversations
    ):
        """Test saving and loading conversations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = manager_class(temp_dir, enabled=True, **manager_kwargs)

            params = get_checkpoint_params(
                manager_name, "conversations", sample_conversations, Conversation
            )

            # Save and load
            save_kwargs = params.get("save_kwargs", {})
            manager.save_checkpoint(*params["save_params"], **save_kwargs)
            loaded = manager.load_checkpoint(
                *params["load_params"], **params["load_kwargs"]
            )

            assert loaded is not None
            assert_data_equal(sample_conversations, loaded, "conversations")

    def test_summaries_roundtrip(
        self, manager_name, manager_class, manager_kwargs, sample_summaries
    ):
        """Test saving and loading summaries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = manager_class(temp_dir, enabled=True, **manager_kwargs)

            params = get_checkpoint_params(
                manager_name, "summaries", sample_summaries, ConversationSummary
            )

            # Save and load
            manager.save_checkpoint(*params["save_params"])
            loaded = manager.load_checkpoint(
                *params["load_params"], **params["load_kwargs"]
            )

            assert loaded is not None
            assert_data_equal(sample_summaries, loaded, "summaries")

    def test_clusters_roundtrip(
        self, manager_name, manager_class, manager_kwargs, sample_clusters
    ):
        """Test saving and loading clusters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = manager_class(temp_dir, enabled=True, **manager_kwargs)

            params = get_checkpoint_params(
                manager_name, "clusters", sample_clusters, Cluster
            )

            # Save and load
            manager.save_checkpoint(*params["save_params"])
            loaded = manager.load_checkpoint(
                *params["load_params"], **params["load_kwargs"]
            )

            assert loaded is not None
            assert_data_equal(sample_clusters, loaded, "clusters")

    def test_projected_clusters_roundtrip(
        self, manager_name, manager_class, manager_kwargs, sample_projected_clusters
    ):
        """Test saving and loading projected clusters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = manager_class(temp_dir, enabled=True, **manager_kwargs)

            params = get_checkpoint_params(
                manager_name, "projected", sample_projected_clusters, ProjectedCluster
            )

            # Save and load
            manager.save_checkpoint(*params["save_params"])
            loaded = manager.load_checkpoint(
                *params["load_params"], **params["load_kwargs"]
            )

            assert loaded is not None
            assert_data_equal(sample_projected_clusters, loaded, "projected_clusters")

    def test_empty_data_handling(self, manager_name, manager_class, manager_kwargs):
        """Test handling of empty data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = manager_class(temp_dir, enabled=True, **manager_kwargs)

            params = get_checkpoint_params(
                manager_name, "empty", [], ConversationSummary
            )

            # Save empty data
            manager.save_checkpoint(*params["save_params"])

            # Load should return None
            loaded = manager.load_checkpoint(
                *params["load_params"], **params["load_kwargs"]
            )
            assert loaded is None

    def test_nonexistent_checkpoint(self, manager_name, manager_class, manager_kwargs):
        """Test loading nonexistent checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = manager_class(temp_dir, enabled=True, **manager_kwargs)

            params = get_checkpoint_params(
                manager_name, "nonexistent", [], ConversationSummary
            )

            # Load should return None
            loaded = manager.load_checkpoint(
                *params["load_params"], **params["load_kwargs"]
            )
            assert loaded is None

    def test_list_checkpoints_basic(
        self, manager_name, manager_class, manager_kwargs, sample_summaries
    ):
        """Test basic checkpoint listing functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = manager_class(temp_dir, enabled=True, **manager_kwargs)

            # Initially empty
            checkpoints = manager.list_checkpoints()
            initial_count = len(checkpoints)

            # Save a checkpoint
            params = get_checkpoint_params(
                manager_name, "test", sample_summaries, ConversationSummary
            )
            manager.save_checkpoint(*params["save_params"])

            # Should have more checkpoints now
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) > initial_count


# =============================================================================
# Manager-Specific Feature Tests
# =============================================================================


class TestManagerSpecificFeatures:
    """Test features specific to individual checkpoint managers."""

    @pytest.mark.skipif(not PARQUET_AVAILABLE, reason="PyArrow not available")
    def test_parquet_compression_options(self, sample_summaries):
        """Test Parquet compression options."""
        compression_options = ["snappy", "gzip"]

        for compression in compression_options:
            with tempfile.TemporaryDirectory() as temp_dir:
                manager = ParquetCheckpointManager(temp_dir, compression=compression)

                # Save and load
                manager.save_checkpoint("test.parquet", sample_summaries)
                loaded = manager.load_checkpoint("test.parquet", ConversationSummary)

                assert loaded is not None
                assert_data_equal(sample_summaries, loaded, "summaries")

    @pytest.mark.skipif(not PARQUET_AVAILABLE, reason="PyArrow not available")
    def test_parquet_file_extensions(self, sample_summaries):
        """Test Parquet file extension handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ParquetCheckpointManager(temp_dir)

            # Test .jsonl extension gets converted
            manager.save_checkpoint("test.jsonl", sample_summaries)
            loaded = manager.load_checkpoint("test.jsonl", ConversationSummary)

            assert loaded is not None
            assert_data_equal(sample_summaries, loaded, "summaries")

    @pytest.mark.skipif(not HF_AVAILABLE, reason="HuggingFace datasets not available")
    def test_hf_checkpoint_info(self, sample_summaries):
        """Test HuggingFace checkpoint info."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = HFDatasetCheckpointManager(temp_dir)

            # Save data
            manager.save_checkpoint("info_test", sample_summaries, "summaries")

            # Get info
            info = manager.get_checkpoint_info("info_test")
            assert info is not None
            assert info["num_rows"] == len(sample_summaries)
            assert info["num_columns"] > 0

    @pytest.mark.skipif(not HF_AVAILABLE, reason="HuggingFace datasets not available")
    def test_hf_delete_checkpoint(self, sample_summaries):
        """Test HuggingFace checkpoint deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = HFDatasetCheckpointManager(temp_dir)

            # Save and verify
            manager.save_checkpoint("to_delete", sample_summaries, "summaries")
            checkpoints = manager.list_checkpoints()
            assert "to_delete" in checkpoints

            # Delete and verify
            deleted = manager.delete_checkpoint("to_delete")
            assert deleted is True

            checkpoints = manager.list_checkpoints()
            assert "to_delete" not in checkpoints


# =============================================================================
# Cross-Manager Compatibility Tests
# =============================================================================


class TestCrossManagerCompatibility:
    """Test compatibility between different checkpoint managers."""

    def test_data_consistency(self, sample_summaries):
        """Test that data is consistent across available managers."""
        results = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with JSONL (always available)
            jsonl_manager = JSONLCheckpointManager(os.path.join(temp_dir, "jsonl"))
            jsonl_manager.save_checkpoint("test.jsonl", sample_summaries)
            results["jsonl"] = jsonl_manager.load_checkpoint(
                "test.jsonl", ConversationSummary
            )

            # Test with Parquet if available
            if PARQUET_AVAILABLE:
                parquet_manager = ParquetCheckpointManager(
                    os.path.join(temp_dir, "parquet")
                )
                parquet_manager.save_checkpoint("test.parquet", sample_summaries)
                results["parquet"] = parquet_manager.load_checkpoint(
                    "test.parquet", ConversationSummary
                )

            # Compare results (skip HF for now due to precision issues)
            if "parquet" in results:
                assert_data_equal(results["jsonl"], results["parquet"], "summaries")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
