"""
Tutorial: Using MultiCheckpointManager for Redundant and Flexible Checkpointing

This tutorial demonstrates how to use MultiCheckpointManager to coordinate
multiple checkpoint backends for improved reliability and performance.
"""

import asyncio
import os
import shutil
from datetime import datetime
from typing import List

from kura import (
    CheckpointManager,
    MultiCheckpointManager,
    Conversation,
    SummaryModel,
    summarise_conversations,
)
from kura.types import Message


def create_sample_conversations() -> List[Conversation]:
    """Create sample conversations for demonstration."""
    return [
        Conversation(
            id=f"conv_{i}",
            created_at=datetime.now(),
            messages=[
                Message(
                    role="user",
                    content=f"Hello, I need help with task {i}",
                    created_at=str(datetime.now()),
                ),
                Message(
                    role="assistant",
                    content=f"I'd be happy to help you with task {i}. What specifically do you need?",
                    created_at=str(datetime.now()),
                ),
            ],
        )
        for i in range(10)
    ]


async def example_1_basic_redundancy():
    """Example 1: Basic redundancy with multiple JSONL backends."""
    print("\n=== Example 1: Basic Redundancy ===")

    # Create directories
    primary_dir = "./checkpoints/primary"
    backup_dir = "./checkpoints/backup"

    # Clean up from previous runs
    for dir_path in [primary_dir, backup_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    # Create individual managers
    primary_mgr = CheckpointManager(primary_dir)
    backup_mgr = CheckpointManager(backup_dir)

    # Create multi-manager with redundancy
    multi_mgr = MultiCheckpointManager(
        [primary_mgr, backup_mgr],
        save_strategy="all_enabled",  # Save to both
        load_strategy="first_found",  # Load from whichever has it
    )

    # Create sample data
    conversations = create_sample_conversations()

    # Use with the procedural API
    summary_model = SummaryModel(model="gpt-4o-mini")
    summaries = await summarise_conversations(
        conversations, model=summary_model, checkpoint_manager=multi_mgr
    )

    print(f"Generated {len(summaries)} summaries")

    # Verify both managers have the data
    print(f"\nPrimary checkpoints: {os.listdir(primary_dir)}")
    print(f"Backup checkpoints: {os.listdir(backup_dir)}")

    # Simulate primary failure by removing its checkpoint
    os.remove(os.path.join(primary_dir, "summaries.jsonl"))
    print("\nSimulated primary storage failure...")

    # Load should still work from backup
    loaded = multi_mgr.load_checkpoint("summaries.jsonl", type(summaries[0]))
    print(f"Successfully loaded {len(loaded)} summaries from backup!")


async def example_2_performance_optimization():
    """Example 2: Performance optimization with primary-only saves."""
    print("\n=== Example 2: Performance Optimization ===")

    # Create directories
    fast_dir = "./checkpoints/fast_local"
    slow_dir = "./checkpoints/slow_network"

    # Clean up
    for dir_path in [fast_dir, slow_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    # Create managers
    fast_mgr = CheckpointManager(fast_dir)
    slow_mgr = CheckpointManager(slow_dir)

    # Multi-manager optimized for performance
    multi_mgr = MultiCheckpointManager(
        [fast_mgr, slow_mgr],
        save_strategy="primary_only",  # Only save to fast storage
        load_strategy="priority",  # Always try fast storage first
    )

    conversations = create_sample_conversations()
    summary_model = SummaryModel(model="gpt-4o-mini")

    summaries = await summarise_conversations(
        conversations, model=summary_model, checkpoint_manager=multi_mgr
    )

    print(f"\nFast storage checkpoints: {os.listdir(fast_dir)}")
    print(
        f"Slow storage checkpoints: {os.listdir(slow_dir) if os.path.exists(slow_dir) else '(empty)'}"
    )

    # Manually backup to slow storage
    print("\nManually backing up to slow storage...")
    slow_mgr.save_checkpoint("summaries.jsonl", summaries)
    print(f"Slow storage checkpoints: {os.listdir(slow_dir)}")


async def example_3_mixed_backends():
    """Example 3: Mix different checkpoint formats (requires optional dependencies)."""
    print("\n=== Example 3: Mixed Backends ===")

    jsonl_dir = "./checkpoints/jsonl"

    # Clean up
    if os.path.exists(jsonl_dir):
        shutil.rmtree(jsonl_dir)

    managers = [CheckpointManager(jsonl_dir)]

    # Try to add Parquet manager if available
    try:
        from kura.checkpoints.parquet import ParquetCheckpointManager

        parquet_dir = "./checkpoints/parquet"
        if os.path.exists(parquet_dir):
            shutil.rmtree(parquet_dir)
        parquet_mgr = ParquetCheckpointManager(parquet_dir)
        managers.append(parquet_mgr)
        print("Added ParquetCheckpointManager")
    except ImportError:
        print("Parquet support not available (install pyarrow)")

    # Try to add HF Dataset manager if available
    try:
        from kura.checkpoints.hf_dataset import HFDatasetCheckpointManager

        hf_dir = "./checkpoints/hf_dataset"
        if os.path.exists(hf_dir):
            shutil.rmtree(hf_dir)
        hf_mgr = HFDatasetCheckpointManager(hf_dir)
        managers.append(hf_mgr)
        print("Added HFDatasetCheckpointManager")
    except ImportError:
        print("HuggingFace datasets support not available (install datasets)")

    # Create multi-manager with all available backends
    multi_mgr = MultiCheckpointManager(
        managers, save_strategy="all_enabled", load_strategy="first_found"
    )

    print(f"\nUsing {len(managers)} checkpoint backends")
    stats = multi_mgr.get_stats()
    for mgr_stat in stats["managers"]:
        print(f"  - {mgr_stat['type']}: {mgr_stat['checkpoint_dir']}")


async def example_4_graceful_degradation():
    """Example 4: Graceful degradation with mixed enabled/disabled managers."""
    print("\n=== Example 4: Graceful Degradation ===")

    # Create managers with different enabled states
    managers = [
        CheckpointManager("./checkpoints/main", enabled=True),
        CheckpointManager("./checkpoints/backup1", enabled=False),  # Disabled
        CheckpointManager("./checkpoints/backup2", enabled=True),
    ]

    # Clean up enabled directories
    for mgr in managers:
        if mgr.enabled and os.path.exists(mgr.checkpoint_dir):
            shutil.rmtree(mgr.checkpoint_dir)

    multi_mgr = MultiCheckpointManager(managers)

    print("Manager states:")
    stats = multi_mgr.get_stats()
    for mgr_stat in stats["managers"]:
        print(f"  - {mgr_stat['type']}: enabled={mgr_stat['enabled']}")

    print(f"\nTotal managers: {stats['num_managers']}")
    print(f"Enabled managers: {stats['enabled_managers']}")

    # Should still work with only enabled managers
    conversations = create_sample_conversations()
    summary_model = SummaryModel(model="gpt-4o-mini")

    summaries = await summarise_conversations(
        conversations, model=summary_model, checkpoint_manager=multi_mgr
    )

    print(f"\nSuccessfully processed {len(summaries)} summaries")
    print("Checkpoints saved to enabled managers only")


async def example_5_checkpoint_migration():
    """Example 5: Migrate checkpoints between formats."""
    print("\n=== Example 5: Checkpoint Migration ===")

    old_dir = "./checkpoints/old_format"
    new_dir = "./checkpoints/new_format"

    # Clean up
    for dir_path in [old_dir, new_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    # Create and populate old checkpoint
    old_mgr = CheckpointManager(old_dir)
    conversations = create_sample_conversations()
    summary_model = SummaryModel(model="gpt-4o-mini")

    print("Creating checkpoints in old format...")
    summaries = await summarise_conversations(
        conversations, model=summary_model, checkpoint_manager=old_mgr
    )

    # Create new manager
    new_mgr = CheckpointManager(new_dir)

    # Create migration multi-manager
    migration_mgr = MultiCheckpointManager(
        [old_mgr, new_mgr],
        save_strategy="all_enabled",  # Save to both during migration
        load_strategy="priority",  # Prefer old format for reading
    )

    # List all checkpoints to migrate
    checkpoints = old_mgr.list_checkpoints()
    print(f"\nFound {len(checkpoints)} checkpoints to migrate")

    # Migrate each checkpoint
    for checkpoint in checkpoints:
        # Load from old
        data = migration_mgr.load_checkpoint(checkpoint, type(summaries[0]))
        if data:
            # Save to new (will save to both due to strategy)
            new_mgr.save_checkpoint(checkpoint, data)
            print(f"Migrated {checkpoint}: {len(data)} items")

    print("\nMigration complete!")
    print(f"Old format files: {os.listdir(old_dir)}")
    print(f"New format files: {os.listdir(new_dir)}")


async def example_6_advanced_pipeline():
    """Example 6: Advanced pipeline with JSONL + Parquet combination."""
    print("\n=== Example 6: Advanced Pipeline (JSONL + Parquet) ===")

    jsonl_dir = "./checkpoints/jsonl_realtime"

    # Clean up
    if os.path.exists(jsonl_dir):
        shutil.rmtree(jsonl_dir)

    # Always have JSONL manager
    managers = [CheckpointManager(jsonl_dir)]

    # Try to add Parquet for better compression
    parquet_available = False
    try:
        from kura.checkpoints.parquet import ParquetCheckpointManager

        parquet_dir = "./checkpoints/parquet_archive"
        if os.path.exists(parquet_dir):
            shutil.rmtree(parquet_dir)
        parquet_mgr = ParquetCheckpointManager(parquet_dir)
        managers.append(parquet_mgr)
        parquet_available = True
        print("Using JSONL for real-time access + Parquet for efficient storage")
    except ImportError:
        print("Using JSONL only (install pyarrow for Parquet support)")

    # Create optimized multi-manager
    multi_mgr = MultiCheckpointManager(
        managers,
        save_strategy="all_enabled",  # Save to both formats
        load_strategy="first_found",  # Load from whichever is faster
    )

    # Run pipeline
    conversations = create_sample_conversations()
    summary_model = SummaryModel(model="gpt-4o-mini")

    print("\nRunning analysis pipeline...")
    summaries = await summarise_conversations(
        conversations, model=summary_model, checkpoint_manager=multi_mgr
    )

    print(f"\nGenerated {len(summaries)} summaries")

    # Show storage efficiency
    jsonl_size = sum(
        os.path.getsize(os.path.join(jsonl_dir, f)) for f in os.listdir(jsonl_dir)
    )
    print(f"JSONL storage size: {jsonl_size:,} bytes")

    if parquet_available:
        parquet_size = sum(
            os.path.getsize(os.path.join(parquet_dir, f))
            for f in os.listdir(parquet_dir)
        )
        print(f"Parquet storage size: {parquet_size:,} bytes")
        print(f"Compression ratio: {jsonl_size / parquet_size:.2f}x")

    # Demonstrate listing all checkpoints
    all_checkpoints = multi_mgr.list_checkpoints()
    print(f"\nAvailable checkpoints across all formats: {all_checkpoints}")


async def main():
    """Run all examples."""
    print("MultiCheckpointManager Tutorial")
    print("==============================")

    # Run examples
    await example_1_basic_redundancy()
    await example_2_performance_optimization()
    await example_3_mixed_backends()
    await example_4_graceful_degradation()
    await example_5_checkpoint_migration()
    await example_6_advanced_pipeline()

    print("\n\nTutorial complete!")
    print("\nKey takeaways:")
    print("1. MultiCheckpointManager provides redundancy and flexibility")
    print("2. Use 'all_enabled' strategy for redundancy")
    print("3. Use 'primary_only' strategy for performance")
    print("4. Mix different checkpoint formats for optimal storage")
    print("5. Gracefully handles disabled managers")
    print("6. Supports checkpoint migration between formats")

    # Clean up all checkpoint directories
    print("\nCleaning up tutorial checkpoints...")
    checkpoint_root = "./checkpoints"
    if os.path.exists(checkpoint_root):
        shutil.rmtree(checkpoint_root)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
