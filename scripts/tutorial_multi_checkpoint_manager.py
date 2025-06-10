#!/usr/bin/env python3
"""
Tutorial: Using MultiCheckpointManager for Multiple Storage Backends

This script demonstrates how to use the new MultiCheckpointManager to save
checkpoints to multiple locations for redundancy, performance, or organization.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path

from kura import (
    CheckpointManager,
    MultiCheckpointManager,
    SummaryModel,
    Conversation,
)


def create_sample_conversations():
    """Create some sample conversations for testing."""
    from kura.types import Message
    from datetime import datetime
    from uuid import uuid4
    
    conversations = []
    sample_prompts = [
        "Help me write a Python function to calculate fibonacci numbers",
        "Explain the difference between machine learning and deep learning",
        "How do I optimize SQL queries for better performance?",
        "What are the best practices for API design?",
        "Can you help me debug this React component?"
    ]
    
    for i, prompt in enumerate(sample_prompts):
        conversation = Conversation(
            id=str(uuid4()),
            created_at=datetime.now(),
            messages=[
                Message(
                    created_at=str(datetime.now()),
                    role="user",
                    content=prompt
                ),
                Message(
                    created_at=str(datetime.now()),
                    role="assistant", 
                    content=f"This is a sample response to: {prompt}"
                )
            ]
        )
        conversations.append(conversation)
    
    return conversations


async def demo_basic_multi_checkpoint():
    """Demonstrate basic multi-checkpoint manager usage."""
    print("🔧 Setting up Multi-Checkpoint Manager Demo")
    print("=" * 50)
    
    # Create temporary directories to simulate different storage locations
    local_dir = tempfile.mkdtemp(prefix="kura_local_")
    backup_dir = tempfile.mkdtemp(prefix="kura_backup_")
    archive_dir = tempfile.mkdtemp(prefix="kura_archive_")
    
    try:
        print(f"📁 Local storage: {local_dir}")
        print(f"💾 Backup storage: {backup_dir}")
        print(f"🗄️  Archive storage: {archive_dir}")
        print()
        
        # Create individual checkpoint managers
        local_manager = CheckpointManager(local_dir, enabled=True)
        backup_manager = CheckpointManager(backup_dir, enabled=True)
        archive_manager = CheckpointManager(archive_dir, enabled=True)
        
        # Create multi-checkpoint manager with all-enabled save strategy
        multi_manager = MultiCheckpointManager(
            managers=[local_manager, backup_manager, archive_manager],
            save_strategy="all_enabled",     # Save to all locations
            load_strategy="first_found",     # Load from first available
            fail_on_save_error=False         # Continue if some saves fail
        )
        
        print("✅ Multi-checkpoint manager created")
        print(f"   Enabled: {multi_manager.enabled}")
        print(f"   Primary directory: {multi_manager.checkpoint_dir}")
        print()
        
        # Create sample data
        conversations = create_sample_conversations()
        print(f"📝 Created {len(conversations)} sample conversations")
        
        # Note: In a real scenario, you would use actual models
        # For this demo, we'll just simulate the checkpoint operations
        print("💾 Simulating checkpoint save operation...")
        
        # This would normally be done by the pipeline functions
        # We'll simulate by creating some mock summary data
        from kura.types import ConversationSummary
        
        mock_summaries = [
            ConversationSummary(
                conversation_id=conv.id,
                summary=f"Summary: {conv.messages[0].content[:50]}...",
                conversation=conv
            ) for conv in conversations
        ]
        
        # Save checkpoints - should save to all three locations
        multi_manager.save_checkpoint("summaries.jsonl", mock_summaries)
        print("✅ Saved checkpoints to all storage locations")
        
        # Verify files exist in all locations
        for name, directory in [("Local", local_dir), ("Backup", backup_dir), ("Archive", archive_dir)]:
            checkpoint_file = Path(directory) / "summaries.jsonl"
            if checkpoint_file.exists():
                file_size = checkpoint_file.stat().st_size
                print(f"   {name}: ✅ {file_size} bytes")
            else:
                print(f"   {name}: ❌ File not found")
        
        print()
        
        # Demonstrate loading
        print("📖 Testing checkpoint loading...")
        loaded_summaries = multi_manager.load_checkpoint("summaries.jsonl", ConversationSummary)
        
        if loaded_summaries:
            print(f"✅ Loaded {len(loaded_summaries)} summaries from checkpoint")
            print(f"   First summary: {loaded_summaries[0].summary[:60]}...")
        else:
            print("❌ Failed to load checkpoints")
        
        print()
        
    finally:
        # Clean up temporary directories
        print("🧹 Cleaning up temporary directories...")
        for temp_dir in [local_dir, backup_dir, archive_dir]:
            shutil.rmtree(temp_dir, ignore_errors=True)
        print("✅ Cleanup complete")


async def demo_redundancy_scenario():
    """Demonstrate redundancy scenario where one storage fails."""
    print("🛡️  Redundancy Demo: Handling Storage Failures")
    print("=" * 50)
    
    # Create temporary directories
    working_dir = tempfile.mkdtemp(prefix="kura_working_")
    
    try:
        print(f"📁 Working storage: {working_dir}")
        print(f"💥 Simulated failing storage: (will fail)")
        print()
        
        # Create managers - one that works, one that will fail
        working_manager = CheckpointManager(working_dir, enabled=True)
        failing_manager = CheckpointManager("/invalid/nonexistent/path", enabled=True)
        
        # Create multi-manager that continues on partial failures
        multi_manager = MultiCheckpointManager(
            managers=[working_manager, failing_manager],
            save_strategy="all_enabled",
            fail_on_save_error=False  # Don't fail if some saves fail
        )
        
        # Create test data
        conversations = create_sample_conversations()
        from kura.types import ConversationSummary
        
        mock_summaries = [
            ConversationSummary(
                conversation_id=conv.id,
                summary=f"Summary: {conv.messages[0].content[:50]}...",
                conversation=conv
            ) for conv in conversations
        ]
        
        print("💾 Attempting to save to both storage locations...")
        try:
            multi_manager.save_checkpoint("summaries.jsonl", mock_summaries)
            print("✅ Save operation completed (with warnings)")
            print("   Working storage: ✅ Saved successfully")
            print("   Failing storage: ❌ Failed as expected")
        except Exception as e:
            print(f"❌ Save failed: {e}")
        
        print()
        
        # Verify we can still load from working storage
        print("📖 Testing recovery from working storage...")
        loaded = multi_manager.load_checkpoint("summaries.jsonl", ConversationSummary)
        
        if loaded:
            print(f"✅ Successfully recovered {len(loaded)} summaries")
            print("   Data integrity maintained despite partial failure")
        else:
            print("❌ Recovery failed")
        
        print()
        
    finally:
        # Clean up
        print("🧹 Cleaning up...")
        shutil.rmtree(working_dir, ignore_errors=True)
        print("✅ Cleanup complete")


async def demo_performance_scenario():
    """Demonstrate performance-oriented setup."""
    print("⚡ Performance Demo: Primary + Backup Storage")
    print("=" * 50)
    
    # Create temporary directories
    fast_dir = tempfile.mkdtemp(prefix="kura_fast_")
    slow_dir = tempfile.mkdtemp(prefix="kura_slow_")
    
    try:
        print(f"⚡ Fast storage (primary): {fast_dir}")
        print(f"🐌 Slow storage (backup): {slow_dir}")
        print()
        
        # Create managers with different priorities
        fast_manager = CheckpointManager(fast_dir, enabled=True)
        slow_manager = CheckpointManager(slow_dir, enabled=True)
        
        # Create multi-manager that saves to primary only for performance
        # but can load from either location
        multi_manager = MultiCheckpointManager(
            managers=[fast_manager, slow_manager],  # Fast manager first (priority)
            save_strategy="primary_only",           # Only save to primary for speed
            load_strategy="first_found"             # Load from first available
        )
        
        # Create test data
        conversations = create_sample_conversations()
        from kura.types import ConversationSummary
        
        mock_summaries = [
            ConversationSummary(
                conversation_id=conv.id,
                summary=f"Summary: {conv.messages[0].content[:50]}...",
                conversation=conv
            ) for conv in conversations
        ]
        
        print("💾 Saving with primary_only strategy (performance optimized)...")
        multi_manager.save_checkpoint("summaries.jsonl", mock_summaries)
        
        # Verify only primary storage has the file
        fast_file = Path(fast_dir) / "summaries.jsonl"
        slow_file = Path(slow_dir) / "summaries.jsonl"
        
        print(f"   Fast storage: {'✅' if fast_file.exists() else '❌'}")
        print(f"   Slow storage: {'✅' if slow_file.exists() else '❌'} (expected)")
        print()
        
        # Demonstrate fallback loading
        print("📖 Testing fallback loading scenario...")
        
        # First, normal load (should load from fast storage)
        loaded = multi_manager.load_checkpoint("summaries.jsonl", ConversationSummary)
        print(f"✅ Loaded from primary storage: {len(loaded) if loaded else 0} summaries")
        
        # Simulate fast storage failure by moving the file
        backup_file = Path(fast_dir) / "summaries_backup.jsonl"
        fast_file.rename(backup_file)
        
        # Copy to slow storage to simulate manual backup process
        shutil.copy2(backup_file, slow_file)
        
        print("   Simulated primary storage failure...")
        
        # Try loading again - should fallback to slow storage
        loaded = multi_manager.load_checkpoint("summaries.jsonl", ConversationSummary)
        if loaded:
            print(f"✅ Fallback successful: loaded {len(loaded)} summaries from backup")
        else:
            print("❌ Fallback failed")
        
        print()
        
    finally:
        # Clean up
        print("🧹 Cleaning up...")
        for temp_dir in [fast_dir, slow_dir]:
            shutil.rmtree(temp_dir, ignore_errors=True)
        print("✅ Cleanup complete")


async def main():
    """Run all demos."""
    print("🚀 Multi-Checkpoint Manager Tutorial")
    print("=" * 70)
    print()
    
    await demo_basic_multi_checkpoint()
    print("\n" + "=" * 70 + "\n")
    
    await demo_redundancy_scenario()
    print("\n" + "=" * 70 + "\n")
    
    await demo_performance_scenario()
    print("\n" + "=" * 70 + "\n")
    
    print("🎉 Tutorial Complete!")
    print()
    print("Key Takeaways:")
    print("• MultiCheckpointManager provides redundancy and flexibility")
    print("• Configure save_strategy based on your needs (all_enabled vs primary_only)")
    print("• Load strategy determines fallback behavior")
    print("• Error handling keeps pipelines running even with partial failures")
    print("• Compatible with existing pipeline functions")


if __name__ == "__main__":
    asyncio.run(main())