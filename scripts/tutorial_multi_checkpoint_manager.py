#!/usr/bin/env python3
"""
Tutorial: Using MultiCheckpointManager with Kura v1 API

This script demonstrates how to use MultiCheckpointManager to coordinate
multiple checkpoint storage backends for redundancy, performance, or
organizational purposes.
"""

import asyncio
import tempfile
from typing import List

from kura import (
    CheckpointManager,
    MultiCheckpointManager,
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
)
from kura.types import Conversation, Message
from kura.summarisation import SummaryModel
from kura.cluster import ClusterModel


def create_sample_conversations() -> List[Conversation]:
    """Create sample conversations for testing."""
    conversations = []
    
    # Tech support conversations
    for i in range(3):
        conv = Conversation(
            id=f"tech_{i}",
            messages=[
                Message(role="user", content=f"I'm having trouble with my software installation {i}"),
                Message(role="assistant", content="Let me help you troubleshoot that issue"),
                Message(role="user", content="Thank you, that fixed it!"),
            ]
        )
        conversations.append(conv)
    
    # Recipe conversations  
    for i in range(3):
        conv = Conversation(
            id=f"recipe_{i}",
            messages=[
                Message(role="user", content=f"Can you help me cook pasta recipe {i}?"),
                Message(role="assistant", content="Here's a great pasta recipe for you"),
                Message(role="user", content="Delicious! Thanks for the recipe."),
            ]
        )
        conversations.append(conv)
        
    return conversations


async def demo_redundancy_setup():
    """Demonstrate redundancy setup: save to both local and backup storage."""
    print("=== Redundancy Setup Demo ===")
    
    with tempfile.TemporaryDirectory() as local_dir, \
         tempfile.TemporaryDirectory() as backup_dir:
        
        # Create individual checkpoint managers
        local_manager = CheckpointManager(local_dir, enabled=True)
        backup_manager = CheckpointManager(backup_dir, enabled=True)
        
        # Create multi-checkpoint manager for redundancy
        multi_manager = MultiCheckpointManager(
            managers=[local_manager, backup_manager],
            save_strategy="all_enabled",    # Save to both locations
            load_strategy="first_found"     # Load from whichever works first
        )
        
        print(f"Local checkpoint dir: {local_dir}")
        print(f"Backup checkpoint dir: {backup_dir}")
        print(f"Multi-manager enabled: {multi_manager.enabled}")
        
        # Sample conversations
        conversations = create_sample_conversations()
        
        # Initialize models (these would normally use real API keys)
        summary_model = SummaryModel()  # Uses mock by default for demo
        
        # Run summarization with multi-checkpoint manager
        print(f"\nSummarizing {len(conversations)} conversations...")
        summaries = await summarise_conversations(
            conversations,
            model=summary_model,
            checkpoint_manager=multi_manager
        )
        
        print(f"Generated {len(summaries)} summaries")
        print("Summaries saved to both local and backup storage!")
        
        # Verify data exists in both locations
        local_data = local_manager.load_checkpoint("summaries.jsonl", type(summaries[0]))
        backup_data = backup_manager.load_checkpoint("summaries.jsonl", type(summaries[0]))
        
        print(f"Local storage has {len(local_data) if local_data else 0} summaries")
        print(f"Backup storage has {len(backup_data) if backup_data else 0} summaries")


async def demo_performance_setup():
    """Demonstrate performance setup: fast primary + slow backup storage."""
    print("\n=== Performance Setup Demo ===")
    
    with tempfile.TemporaryDirectory() as fast_dir, \
         tempfile.TemporaryDirectory() as slow_dir:
        
        # Create managers with different characteristics
        fast_manager = CheckpointManager(fast_dir, enabled=True)  # SSD storage
        slow_manager = CheckpointManager(slow_dir, enabled=True)  # Network storage
        
        # Create multi-checkpoint manager for performance
        multi_manager = MultiCheckpointManager(
            managers=[fast_manager, slow_manager],
            save_strategy="primary_only",   # Save to fast storage only
            load_strategy="priority"        # Load from fast storage first
        )
        
        print(f"Fast storage (primary): {fast_dir}")
        print(f"Slow storage (backup): {slow_dir}")
        
        # Sample conversations
        conversations = create_sample_conversations()
        
        # Initialize models
        cluster_model = ClusterModel()  # Uses mock by default for demo
        
        # Create some dummy summaries for clustering demo
        from kura.types import ConversationSummary
        summaries = [
            ConversationSummary(
                conversation_id=conv.id,
                summary=f"Summary for {conv.id}",
                embedding=[0.1] * 384  # Dummy embedding
            )
            for conv in conversations
        ]
        
        # Run clustering with multi-checkpoint manager
        print(f"\nClustering {len(summaries)} summaries...")
        clusters = await generate_base_clusters_from_conversation_summaries(
            summaries,
            model=cluster_model,
            checkpoint_manager=multi_manager
        )
        
        print(f"Generated {len(clusters)} clusters")
        print("Clusters saved to fast storage only (primary_only strategy)!")
        
        # Verify data location
        fast_data = fast_manager.load_checkpoint("clusters.jsonl", type(clusters[0]))
        slow_data = slow_manager.load_checkpoint("clusters.jsonl", type(clusters[0]))
        
        print(f"Fast storage has {len(fast_data) if fast_data else 0} clusters")
        print(f"Slow storage has {len(slow_data) if slow_data else 0} clusters")


async def demo_environment_separation():
    """Demonstrate environment separation: dev/staging/prod isolation."""
    print("\n=== Environment Separation Demo ===")
    
    with tempfile.TemporaryDirectory() as dev_dir, \
         tempfile.TemporaryDirectory() as staging_dir, \
         tempfile.TemporaryDirectory() as prod_dir:
        
        # Create environment-specific managers
        dev_manager = CheckpointManager(dev_dir, enabled=True)
        staging_manager = CheckpointManager(staging_dir, enabled=True)
        prod_manager = CheckpointManager(prod_dir, enabled=False)  # Disabled for this demo
        
        # Multi-manager for development environment
        dev_multi_manager = MultiCheckpointManager(
            managers=[dev_manager, staging_manager, prod_manager],
            save_strategy="all_enabled",  # Save to all enabled environments
            load_strategy="first_found"
        )
        
        print(f"Dev environment: {dev_dir}")
        print(f"Staging environment: {staging_dir}")
        print(f"Prod environment: {prod_dir} (disabled)")
        print(f"Enabled managers: {sum(1 for m in dev_multi_manager.managers if m.enabled)}")
        
        # Sample conversations for development
        conversations = create_sample_conversations()[:2]  # Smaller dataset for dev
        
        # Initialize models
        summary_model = SummaryModel()
        
        # Run pipeline in development environment
        print(f"\nProcessing {len(conversations)} conversations in dev environment...")
        summaries = await summarise_conversations(
            conversations,
            model=summary_model,
            checkpoint_manager=dev_multi_manager
        )
        
        print(f"Generated {len(summaries)} summaries")
        print("Data saved to dev and staging environments (prod disabled)!")
        
        # Verify data distribution
        dev_data = dev_manager.load_checkpoint("summaries.jsonl", type(summaries[0]))
        staging_data = staging_manager.load_checkpoint("summaries.jsonl", type(summaries[0]))
        prod_data = prod_manager.load_checkpoint("summaries.jsonl", type(summaries[0]))
        
        print(f"Dev has {len(dev_data) if dev_data else 0} summaries")
        print(f"Staging has {len(staging_data) if staging_data else 0} summaries") 
        print(f"Prod has {len(prod_data) if prod_data else 0} summaries (should be 0)")


async def demo_error_handling():
    """Demonstrate error handling and fallback behavior."""
    print("\n=== Error Handling Demo ===")
    
    with tempfile.TemporaryDirectory() as good_dir, \
         tempfile.TemporaryDirectory() as temp_dir:
        
        # Create one good manager and one that will fail
        good_manager = CheckpointManager(good_dir, enabled=True)
        
        # Create a manager with invalid directory to simulate failure
        bad_dir = "/nonexistent/invalid/path"
        bad_manager = CheckpointManager(bad_dir, enabled=True)
        
        multi_manager = MultiCheckpointManager(
            managers=[bad_manager, good_manager],  # Bad manager first
            save_strategy="all_enabled",
            load_strategy="first_found"
        )
        
        print(f"Good storage: {good_dir}")
        print(f"Bad storage: {bad_dir} (will fail)")
        
        # Sample data
        conversations = create_sample_conversations()[:1]
        summary_model = SummaryModel()
        
        # This should handle the error gracefully
        print(f"\nProcessing with error handling...")
        try:
            summaries = await summarise_conversations(
                conversations,
                model=summary_model,
                checkpoint_manager=multi_manager
            )
            print(f"Successfully generated {len(summaries)} summaries despite errors!")
            
            # Verify only good manager has data
            good_data = good_manager.load_checkpoint("summaries.jsonl", type(summaries[0]))
            print(f"Good storage has {len(good_data) if good_data else 0} summaries")
            
        except Exception as e:
            print(f"Error occurred: {e}")


def print_usage_examples():
    """Print usage examples for different scenarios."""
    print("\n=== Usage Examples ===")
    
    print("""
# Redundancy: Save to multiple locations
multi_manager = MultiCheckpointManager(
    managers=[local_manager, cloud_manager],
    save_strategy="all_enabled",    # Save everywhere
    load_strategy="first_found"     # Load from first available
)

# Performance: Fast primary + slow backup
multi_manager = MultiCheckpointManager(
    managers=[ssd_manager, network_manager],
    save_strategy="primary_only",   # Save to fast storage only
    load_strategy="priority"        # Try fast storage first
)

# Environment separation
multi_manager = MultiCheckpointManager(
    managers=[dev_manager, staging_manager, prod_manager],
    save_strategy="all_enabled",
    load_strategy="first_found"
)

# Use in pipeline functions
summaries = await summarise_conversations(
    conversations,
    model=summary_model,
    checkpoint_manager=multi_manager  # Drop-in replacement!
)
""")


async def main():
    """Run all demos."""
    print("MultiCheckpointManager Tutorial")
    print("=" * 50)
    
    await demo_redundancy_setup()
    await demo_performance_setup()
    await demo_environment_separation()
    await demo_error_handling()
    
    print_usage_examples()
    
    print("\n" + "=" * 50)
    print("Tutorial complete! MultiCheckpointManager provides intelligent")
    print("coordination of multiple checkpoint storage backends for:")
    print("• Redundancy and data safety")
    print("• Performance optimization")
    print("• Environment separation")
    print("• Graceful error handling")


if __name__ == "__main__":
    asyncio.run(main())