#!/usr/bin/env python3
"""
Tutorial test file to validate procedural API examples and reference check outputs.

This test ensures the tutorial examples work correctly and validates import structure
without requiring API keys for basic functionality testing.
"""
import asyncio
import tempfile
import inspect
from pathlib import Path

from kura.v1 import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    CheckpointManager,
)
from kura.types import Conversation


def test_tutorial_imports():
    """Test that all tutorial imports work correctly."""
    print("📦 Testing tutorial imports...")
    
    from kura.v1 import (
        summarise_conversations,
        generate_base_clusters_from_conversation_summaries,
        reduce_clusters_from_base_clusters,
        reduce_dimensionality_from_clusters,
        CheckpointManager,
    )
    
    from kura.types import Conversation
    
    print("✅ All tutorial imports successful")
    
    assert callable(summarise_conversations), "summarise_conversations should be callable"
    assert callable(generate_base_clusters_from_conversation_summaries), "clustering function should be callable"
    assert callable(reduce_clusters_from_base_clusters), "meta-clustering function should be callable"
    assert callable(reduce_dimensionality_from_clusters), "dimensionality function should be callable"
    
    print("✅ All functions are callable")
    
    from kura import (
        summarise_conversations as main_summarise,
        generate_base_clusters_from_conversation_summaries as main_clusters,
        CheckpointManager as main_checkpoint
    )
    
    assert summarise_conversations is main_summarise, "Functions should be the same between v1 and main"
    assert generate_base_clusters_from_conversation_summaries is main_clusters, "Functions should be the same"
    assert CheckpointManager is main_checkpoint, "CheckpointManager should be the same"
    
    print("✅ Import consistency verified between kura.v1 and main kura module")


def test_tutorial_data_loading():
    """Test conversation loading from tutorial examples."""
    print("📥 Testing conversation data loading...")
    
    conversations = Conversation.from_hf_dataset(
        "ivanleomk/synthetic-gemini-conversations", 
        split="train"
    )[:5]  # Use only 5 conversations for testing
    
    print(f"✅ Loaded {len(conversations)} conversations")
    assert len(conversations) == 5, f"Expected 5 conversations, got {len(conversations)}"
    
    # Validate conversation structure
    sample_conversation = conversations[0]
    assert hasattr(sample_conversation, 'chat_id'), "Conversation should have chat_id"
    assert hasattr(sample_conversation, 'messages'), "Conversation should have messages"
    assert hasattr(sample_conversation, 'created_at'), "Conversation should have created_at"
    assert len(sample_conversation.messages) > 0, "Conversation should have messages"
    
    print(f"   Sample conversation ID: {sample_conversation.chat_id}")
    print(f"   Message count: {len(sample_conversation.messages)}")
    print(f"   Created at: {sample_conversation.created_at}")
    
    # Validate message structure
    sample_message = sample_conversation.messages[0]
    assert hasattr(sample_message, 'role'), "Message should have role"
    assert hasattr(sample_message, 'content'), "Message should have content"
    
    print(f"   Sample message role: {sample_message.role}")
    print(f"   Sample message content: {sample_message.content[:100]}...")
    
    return conversations


def test_checkpoint_manager():
    """Test CheckpointManager functionality from tutorial examples."""
    print("\n💾 Testing CheckpointManager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "test_checkpoints"
        checkpoint_manager = CheckpointManager(str(checkpoint_dir), enabled=True)
        
        assert checkpoint_manager.checkpoint_dir == str(checkpoint_dir), "Checkpoint dir should match"
        assert checkpoint_manager.enabled == True, "Checkpointing should be enabled"
        
        test_filename = "test_summaries.jsonl"
        checkpoint_path = checkpoint_manager.get_checkpoint_path(test_filename)
        expected_path = checkpoint_dir / test_filename
        
        assert checkpoint_path == str(expected_path), "Checkpoint path should be correct"
        
        print(f"✅ CheckpointManager configured correctly")
        print(f"   Checkpoint directory: {checkpoint_manager.checkpoint_dir}")
        print(f"   Enabled: {checkpoint_manager.enabled}")
        print(f"   Sample checkpoint path: {checkpoint_path}")
        
        return checkpoint_manager


async def test_procedural_api_structure():
    """Test the procedural API structure and function signatures from tutorial examples."""
    print("\n🔧 Testing procedural API structure...")
    
    
    sig = inspect.signature(summarise_conversations)
    params = list(sig.parameters.keys())
    assert 'conversations' in params, "summarise_conversations should have conversations parameter"
    assert 'model' in params, "summarise_conversations should have model parameter"
    assert 'checkpoint_manager' in params, "summarise_conversations should have checkpoint_manager parameter"
    
    print(f"✅ summarise_conversations signature: {list(sig.parameters.keys())}")
    
    sig = inspect.signature(generate_base_clusters_from_conversation_summaries)
    params = list(sig.parameters.keys())
    assert 'summaries' in params, "clustering function should have summaries parameter"
    assert 'model' in params, "clustering function should have model parameter"
    assert 'checkpoint_manager' in params, "clustering function should have checkpoint_manager parameter"
    
    print(f"✅ generate_base_clusters_from_conversation_summaries signature: {list(sig.parameters.keys())}")
    
    sig = inspect.signature(reduce_clusters_from_base_clusters)
    params = list(sig.parameters.keys())
    assert 'clusters' in params, "meta-clustering function should have clusters parameter"
    assert 'model' in params, "meta-clustering function should have model parameter"
    
    print(f"✅ reduce_clusters_from_base_clusters signature: {list(sig.parameters.keys())}")
    
    sig = inspect.signature(reduce_dimensionality_from_clusters)
    params = list(sig.parameters.keys())
    assert 'clusters' in params, "dimensionality function should have clusters parameter"
    assert 'model' in params, "dimensionality function should have model parameter"
    
    print(f"✅ reduce_dimensionality_from_clusters signature: {list(sig.parameters.keys())}")
    
    print("✅ All procedural API functions have correct signatures matching tutorial examples")


async def main():
    """Run all tutorial tests."""
    print("🚀 Starting Tutorial Reference Tests")
    print("=" * 60)
    
    test_tutorial_imports()
    
    conversations = test_tutorial_data_loading()
    
    checkpoint_manager = test_checkpoint_manager()
    
    # Test procedural API structure
    await test_procedural_api_structure()
    
    print("\n📊 Tutorial Reference Test Results:")
    print("=" * 40)
    print(f"  ✅ Procedural API imports: PASS")
    print(f"  ✅ Conversation loading: PASS ({len(conversations)} conversations)")
    print(f"  ✅ CheckpointManager: PASS")
    print(f"  ✅ Function signatures: PASS")
    print(f"  ✅ Import consistency: PASS")
    
    print("\n✨ Tutorial reference tests completed successfully!")
    print("=" * 60)
    print("📋 Summary:")
    print("  • All procedural API imports work correctly")
    print("  • Tutorial examples can load conversation data")
    print("  • CheckpointManager functionality is validated")
    print("  • Function signatures match tutorial usage patterns")
    print("  • Import consistency between kura.v1 and main kura module verified")
    print()
    print("🎯 The tutorial examples are ready for reference checking!")
    print("   Users can run this test to verify their environment is set up correctly")
    print("   before running the full tutorial with API keys.")


if __name__ == "__main__":
    asyncio.run(main())
