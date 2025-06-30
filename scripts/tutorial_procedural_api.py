import time
import asyncio
from contextlib import contextmanager

from kura import MultiCheckpointManager
from kura.checkpoints import (
    HFDatasetCheckpointManager,
    JSONLCheckpointManager,
    ParquetCheckpointManager,
)
from kura.types import Conversation
from kura.summarisation import SummaryModel, summarise_conversations
from kura.k_means import MiniBatchKmeansClusteringMethod
from kura.cluster import (
    ClusterDescriptionModel,
    generate_base_clusters_from_conversation_summaries,
)
from kura.meta_cluster import MetaClusterModel, reduce_clusters_from_base_clusters
from kura.dimensionality import HDBUMAP, reduce_dimensionality_from_clusters
from rich.console import Console


class TimerManager:
    """A timer class that collects timing data for review."""

    def __init__(self):
        self.timings = {}

    @contextmanager
    def timer(self, message):
        """Context manager to time operations and store results."""
        start_time = time.time()
        yield
        end_time = time.time()
        duration = end_time - start_time
        self.timings[message] = duration
        print(f"{message} took {duration:.2f} seconds")

    def print_summary(self):
        """Print a summary of all collected timings."""
        print(f"\n{'=' * 60}")
        print(f"{'TIMING SUMMARY':^60}")
        print(f"{'=' * 60}")

        total_time = sum(self.timings.values())

        for operation, duration in self.timings.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            print(f"{operation:<40} {duration:>8.2f}s ({percentage:>5.1f}%)")

        print(f"{'-' * 60}")
        print(f"{'Total Time':<40} {total_time:>8.2f}s")
        print(f"{'=' * 60}\n")


# Create a global timer manager instance
timer_manager = TimerManager()


def show_section_header(title):
    """Display a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}\n")


console = Console()
summary_model = SummaryModel(console=console, max_concurrent_requests=100)
CHECKPOINT_DIR = "./tutorial_checkpoints"

minibatch_kmeans_clustering = MiniBatchKmeansClusteringMethod(
    clusters_per_group=10,  # Target items per cluster
    batch_size=1000,  # Mini-batch size for processing
    max_iter=100,  # Maximum iterations
    random_state=42,  # Random seed for reproducibility
)

cluster_model = ClusterDescriptionModel(
    console=console,
)
meta_cluster_model = MetaClusterModel(console=console, max_concurrent_requests=100)
dimensionality_model = HDBUMAP()

with timer_manager.timer("Loading sample conversations"):
    conversations = Conversation.from_hf_dataset(
        "ivanleomk/synthetic-gemini-conversations", split="train"
    )

print(f"Loaded {len(conversations)} conversations successfully!\n")

with timer_manager.timer("Saving conversations to JSON"):
    import json
    import os

    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Convert conversations to JSON format
    conversations_data = [conv.model_dump() for conv in conversations]

    # Save to conversations.json
    with open(f"{CHECKPOINT_DIR}/conversations.json", "w") as f:
        json.dump(conversations_data, f, indent=2, default=str)

print(
    f"Saved {len(conversations)} conversations to {CHECKPOINT_DIR}/conversations.json\n"
)

# Create individual checkpoint managers
hf_manager = HFDatasetCheckpointManager(f"{CHECKPOINT_DIR}/hf", enabled=True)
parquet_manager = ParquetCheckpointManager(f"{CHECKPOINT_DIR}/parquet", enabled=True)
jsonl_manager = JSONLCheckpointManager(f"{CHECKPOINT_DIR}/jsonl", enabled=True)

# Create a MultiCheckpointManager that saves to all formats
multi_checkpoint_manager = MultiCheckpointManager(
    [hf_manager, parquet_manager, jsonl_manager],
    save_strategy="all_enabled",  # Save to all formats
    load_strategy="first_found",  # Load from the first available
)

print(
    f"Using MultiCheckpointManager with {len(multi_checkpoint_manager.managers)} backends:"
)
for manager in multi_checkpoint_manager.managers:
    print(f"  - {type(manager).__name__}: {manager.checkpoint_dir}")
print()


async def process():
    """Process conversations using the MultiCheckpointManager."""
    print("Step 1: Generating summaries with multi-checkpoint support...")
    with timer_manager.timer("MultiCheckpointManager - Summarization"):
        summaries = await summarise_conversations(
            conversations,
            model=summary_model,
            checkpoint_manager=multi_checkpoint_manager,
        )
    print(f"Generated {len(summaries)} summaries using multi-checkpoint")

    print("\nStep 2: Generating clusters with multi-checkpoint support...")
    with timer_manager.timer("MultiCheckpointManager - Clustering"):
        clusters = await generate_base_clusters_from_conversation_summaries(
            summaries,
            model=cluster_model,
            clustering_method=minibatch_kmeans_clustering,
            checkpoint_manager=multi_checkpoint_manager,
        )
    print(f"Generated {len(clusters)} clusters using multi-checkpoint")

    print("\nStep 3: Meta clustering with multi-checkpoint support...")
    with timer_manager.timer("MultiCheckpointManager - Meta clustering"):
        reduced_clusters = await reduce_clusters_from_base_clusters(
            clusters,
            model=meta_cluster_model,
            checkpoint_manager=multi_checkpoint_manager,
        )
    print(f"Reduced to {len(reduced_clusters)} meta clusters using multi-checkpoint")

    print("\nStep 4: Dimensionality reduction with multi-checkpoint support...")
    with timer_manager.timer("MultiCheckpointManager - Dimensionality reduction"):
        projected_clusters = await reduce_dimensionality_from_clusters(
            reduced_clusters,
            model=dimensionality_model,
            checkpoint_manager=multi_checkpoint_manager,
        )
    print(
        f"Generated {len(projected_clusters)} projected clusters using multi-checkpoint"
    )

    # Show checkpoint statistics
    print("\nCheckpoint Statistics:")
    stats = multi_checkpoint_manager.get_stats()
    for mgr_stat in stats["managers"]:
        print(
            f"  - {mgr_stat['type']}: {mgr_stat.get('checkpoint_count', 'N/A')} files"
        )

    return summaries, clusters, reduced_clusters, projected_clusters


# Run the process once with all checkpoints being saved simultaneously
summaries, clusters, reduced_clusters, projected_clusters = asyncio.run(process())

print("\nDemonstrating load strategies...")

# Create a new MultiCheckpointManager with priority loading (HF first)
priority_manager = MultiCheckpointManager(
    [hf_manager, parquet_manager, jsonl_manager],
    save_strategy="primary_only",  # Only save to HF
    load_strategy="priority",  # Only load from HF
)

print("\nTesting priority load from HF datasets only...")
with timer_manager.timer("Priority load - HF Dataset"):
    loaded_summaries = priority_manager.load_checkpoint(
        "summaries", summaries[0].__class__
    )
    if loaded_summaries:
        print(f"  Successfully loaded {len(loaded_summaries)} summaries from HF")

# Create another manager that tries all formats
fallback_manager = MultiCheckpointManager(
    [parquet_manager, jsonl_manager],  # Exclude HF
    load_strategy="first_found",
)

print("\nTesting fallback loading (Parquet → JSONL)...")
with timer_manager.timer("Fallback load - Parquet/JSONL"):
    loaded_clusters = fallback_manager.load_checkpoint(
        "clusters", clusters[0].__class__
    )
    if loaded_clusters:
        print(f"  Successfully loaded {len(loaded_clusters)} clusters")

timer_manager.print_summary()
