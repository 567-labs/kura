"""
Tutorial: Using MultiCheckpointManager for redundant and flexible checkpoint storage

This tutorial demonstrates how to use the MultiCheckpointManager to coordinate
multiple checkpoint storage backends for improved reliability, performance,
and flexibility.
"""

import asyncio
import os
from kura.v1 import (
    CheckpointManager,
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
)
from kura.checkpoints import MultiCheckpointManager
from kura import (
    SummaryModel,
    ClusterDescriptionModel,
    MetaClusterModel,
    Conversation,
)
from kura.dimensionality import HDBUMAP


async def example_1_redundancy():
    """Example 1: Using multiple checkpoint managers for redundancy.

    This example shows how to save checkpoints to multiple locations
    for backup and disaster recovery.
    """
    print("\n=== Example 1: Redundancy with Multiple Backends ===\n")

    # Setup multiple checkpoint managers
    local_mgr = CheckpointManager("./checkpoints/local")
    backup_mgr = CheckpointManager("./checkpoints/backup")

    # Create multi-manager with all_enabled strategy
    multi_mgr = MultiCheckpointManager(
        managers=[local_mgr, backup_mgr],
        save_strategy="all_enabled",  # Save to both locations
        load_strategy="first_found",  # Load from first available
    )

    print(f"Created: {multi_mgr}")

    # Load sample conversations
    conversations = [
        Conversation.from_messages(
            [
                {"role": "user", "content": "How do I reset my password?"},
                {
                    "role": "assistant",
                    "content": "You can reset your password from the settings page.",
                },
            ]
        ),
        Conversation.from_messages(
            [
                {"role": "user", "content": "What's the weather like today?"},
                {
                    "role": "assistant",
                    "content": "I don't have access to real-time weather data.",
                },
            ]
        ),
    ]

    # Run pipeline with multi-checkpoint manager
    summary_model = SummaryModel()

    print("\nGenerating summaries...")
    summaries = await summarise_conversations(
        conversations,
        model=summary_model,
        checkpoint_manager=multi_mgr,  # Will save to both locations
    )

    print(f"Generated {len(summaries)} summaries")
    print("Checkpoints saved to both './checkpoints/local' and './checkpoints/backup'")

    # Verify files exist in both locations
    for path in ["./checkpoints/local", "./checkpoints/backup"]:
        if os.path.exists(os.path.join(path, "summaries.jsonl")):
            print(f"✓ Checkpoint verified in {path}")


async def example_2_performance():
    """Example 2: Optimizing performance with primary-only saves.

    This example shows how to save to a fast primary storage while
    maintaining a slower backup for recovery.
    """
    print("\n\n=== Example 2: Performance Optimization ===\n")

    # Setup managers with different characteristics
    fast_local = CheckpointManager("./checkpoints/fast_ssd")
    slow_backup = CheckpointManager("./checkpoints/network_backup")

    # Create multi-manager with primary_only strategy
    multi_mgr = MultiCheckpointManager(
        managers=[fast_local, slow_backup],
        save_strategy="primary_only",  # Save only to fast storage
        load_strategy="priority",  # Try fast storage first
    )

    print(f"Created: {multi_mgr}")
    print(
        "This configuration saves only to fast SSD, with network backup as fallback for loading"
    )


async def example_3_environment_separation():
    """Example 3: Separating development and production checkpoints.

    This example shows how to manage different checkpoint locations
    for different environments.
    """
    print("\n\n=== Example 3: Environment Separation ===\n")

    # Determine environment
    env = os.getenv("KURA_ENV", "development")

    # Setup environment-specific managers
    if env == "production":
        managers = [
            CheckpointManager("./checkpoints/prod"),
            CheckpointManager(
                "s3://prod-backups/checkpoints", enabled=False
            ),  # Placeholder
        ]
    else:
        managers = [
            CheckpointManager("./checkpoints/dev"),
            CheckpointManager("./checkpoints/dev_backup", enabled=True),
        ]

    multi_mgr = MultiCheckpointManager(
        managers=managers, save_strategy="all_enabled", load_strategy="first_found"
    )

    print(f"Running in {env} environment")
    print(f"Created: {multi_mgr}")


async def example_4_conditional_checkpointing():
    """Example 4: Conditionally enabling/disabling checkpoint managers.

    This example shows how to dynamically control which managers are active.
    """
    print("\n\n=== Example 4: Conditional Checkpointing ===\n")

    # Create managers with conditional enabling
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"

    managers = [
        CheckpointManager("./checkpoints/main", enabled=True),
        CheckpointManager("./checkpoints/debug", enabled=debug_mode),
    ]

    multi_mgr = MultiCheckpointManager(
        managers=managers, save_strategy="all_enabled", load_strategy="first_found"
    )

    print(f"Debug mode: {debug_mode}")
    print(f"Created: {multi_mgr}")
    print(f"Active managers: {sum(1 for m in managers if m.enabled)}")


async def example_5_full_pipeline():
    """Example 5: Complete pipeline with MultiCheckpointManager.

    This example demonstrates using MultiCheckpointManager throughout
    the entire Kura pipeline.
    """
    print("\n\n=== Example 5: Full Pipeline with MultiCheckpointManager ===\n")

    # Setup redundant checkpoint managers
    primary = CheckpointManager("./checkpoints/primary")
    secondary = CheckpointManager("./checkpoints/secondary")

    multi_mgr = MultiCheckpointManager(
        managers=[primary, secondary],
        save_strategy="all_enabled",
        load_strategy="first_found",
    )

    # Load conversations
    conversations = [
        Conversation.from_messages(
            [
                {"role": "user", "content": f"Question about topic {i}"},
                {"role": "assistant", "content": f"Answer about topic {i}"},
            ]
        )
        for i in range(10)
    ]

    # Initialize models
    summary_model = SummaryModel()
    cluster_model = ClusterDescriptionModel(
        summary_model_name="gpt-4o-mini",
        embedding_model_name="text-embedding-3-small",
    )
    meta_cluster_model = MetaClusterModel(max_clusters=3)
    dim_reduction_model = HDBUMAP()

    # Run full pipeline with multi-checkpoint manager
    print("Step 1: Summarizing conversations...")
    summaries = await summarise_conversations(
        conversations, model=summary_model, checkpoint_manager=multi_mgr
    )
    print(f"✓ Generated {len(summaries)} summaries")

    print("\nStep 2: Creating base clusters...")
    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries, model=cluster_model, checkpoint_manager=multi_mgr
    )
    print(f"✓ Created {len(clusters)} base clusters")

    print("\nStep 3: Building cluster hierarchy...")
    hierarchical_clusters = await reduce_clusters_from_base_clusters(
        clusters, model=meta_cluster_model, checkpoint_manager=multi_mgr
    )
    root_clusters = [c for c in hierarchical_clusters if c.parent_id is None]
    print(f"✓ Reduced to {len(root_clusters)} root clusters")

    print("\nStep 4: Reducing dimensionality...")
    projected = await reduce_dimensionality_from_clusters(
        hierarchical_clusters, model=dim_reduction_model, checkpoint_manager=multi_mgr
    )
    print(f"✓ Projected {len(projected)} clusters to 2D")

    print(
        "\nPipeline complete! Checkpoints saved to both primary and secondary locations."
    )

    # List all checkpoints
    all_checkpoints = multi_mgr.list_checkpoints()
    print(f"\nAvailable checkpoints: {all_checkpoints}")


async def example_6_mixed_formats():
    """Example 6: Using JSONL and Parquet checkpoint formats together.

    This example shows how to combine JSONL (default) and Parquet checkpoint
    managers for optimal storage and performance.
    """
    print("\n\n=== Example 6: JSONL + Parquet Mixed Format Example ===\n")

    # Import checkpoint managers
    from kura.checkpoints import JSONLCheckpointManager

    # Always use JSONL as the primary format
    jsonl_mgr = JSONLCheckpointManager("./checkpoints/jsonl")
    managers = [jsonl_mgr]
    print("✓ Using JSONLCheckpointManager (primary format)")

    # Try to add ParquetCheckpointManager if available
    try:
        from kura.v1 import ParquetCheckpointManager

        parquet_mgr = ParquetCheckpointManager("./checkpoints/parquet")
        managers.append(parquet_mgr)
        print("✓ Added ParquetCheckpointManager (compressed backup)")
    except ImportError:
        print(
            "✗ ParquetCheckpointManager not available (install with: pip install pyarrow)"
        )

    # Create multi-format manager
    multi_mgr = MultiCheckpointManager(
        managers=managers,
        save_strategy="all_enabled",  # Save in both formats
        load_strategy="priority",  # Load from JSONL first (faster for small files)
    )

    print("\nCreated multi-format checkpoint manager:")
    print("  - Primary: JSONL (human-readable, easy debugging)")
    print("  - Backup: Parquet (50% smaller, columnar format)")
    print(f"  - Strategy: {multi_mgr}")

    # Run a small pipeline to demonstrate
    conversations = [
        Conversation.from_messages(
            [
                {"role": "user", "content": f"Sample question {i}"},
                {"role": "assistant", "content": f"Sample answer {i}"},
            ]
        )
        for i in range(5)
    ]

    summary_model = SummaryModel()

    print("\nRunning pipeline with multi-format checkpoints...")
    summaries = await summarise_conversations(
        conversations, model=summary_model, checkpoint_manager=multi_mgr
    )

    print(f"✓ Generated {len(summaries)} summaries")

    # Show file sizes if both formats were saved
    if len(managers) > 1:
        import os

        jsonl_size = os.path.getsize("./checkpoints/jsonl/summaries.jsonl")
        parquet_size = os.path.getsize("./checkpoints/parquet/summaries.parquet")

        print("\nCheckpoint file sizes:")
        print(f"  - JSONL: {jsonl_size:,} bytes")
        print(
            f"  - Parquet: {parquet_size:,} bytes ({100 * (1 - parquet_size / jsonl_size):.0f}% smaller)"
        )

        print("\nBenefits of this setup:")
        print("  - JSONL: Easy to inspect, debug, and manually edit")
        print("  - Parquet: Better compression, faster for large datasets")
        print("  - MultiCheckpointManager: Automatic redundancy and format flexibility")


def cleanup_examples():
    """Clean up checkpoint directories created by examples."""
    import shutil

    dirs_to_clean = [
        "./checkpoints/local",
        "./checkpoints/backup",
        "./checkpoints/fast_ssd",
        "./checkpoints/network_backup",
        "./checkpoints/dev",
        "./checkpoints/dev_backup",
        "./checkpoints/prod",
        "./checkpoints/main",
        "./checkpoints/debug",
        "./checkpoints/primary",
        "./checkpoints/secondary",
        "./checkpoints/jsonl",
        "./checkpoints/parquet",
        "./checkpoints/hf_datasets",
    ]

    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    # Remove parent checkpoints dir if empty
    if os.path.exists("./checkpoints") and not os.listdir("./checkpoints"):
        os.rmdir("./checkpoints")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("MultiCheckpointManager Tutorial")
    print("=" * 60)

    # Run examples
    await example_1_redundancy()
    await example_2_performance()
    await example_3_environment_separation()
    await example_4_conditional_checkpointing()
    await example_5_full_pipeline()
    await example_6_mixed_formats()

    print("\n" + "=" * 60)
    print("Tutorial complete!")
    print("=" * 60)

    # Cleanup
    print("\nCleaning up example checkpoint directories...")
    cleanup_examples()
    print("✓ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
