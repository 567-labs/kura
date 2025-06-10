#!/usr/bin/env python3
"""
Tutorial: Using VLLM models with Kura for scalable processing.

This script demonstrates how to use VLLM-based embedding and summarization
models for cost-effective, high-performance conversation analysis at scale.

Requirements:
- pip install -e ".[vllm]"  # Install VLLM dependencies
- GPU with sufficient memory (see config recommendations)
"""

import asyncio
import logging
from pathlib import Path

# Kura imports
from kura.types import Conversation
from kura.v1 import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    CheckpointManager
)

# VLLM model imports
from kura.vllm_embedding import VLLMEmbeddingModel, create_vllm_embedding_model_for_scale
from kura.vllm_summarisation import VLLMSummaryModel, create_vllm_summary_model_for_scale
from kura.vllm_config import VLLMConfigManager, auto_select_models, should_use_vllm

# Standard Kura models for comparison
from kura.cluster import ClusterModel
from kura.meta_cluster import MetaClusterModel
from kura.dimensionality import HDBUMAP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate VLLM usage with Kura."""
    
    # ==========================================================================
    # Step 1: Load sample conversations
    # ==========================================================================
    
    print("🔄 Loading sample conversations...")
    
    # Option 1: Load from a Hugging Face dataset
    try:
        conversations = Conversation.from_hf_dataset(
            "ivanleomk/synthetic-gemini-conversations",
            split="train",
            max_conversations=1000  # Start with a smaller sample
        )
        print(f"✅ Loaded {len(conversations)} conversations from HF dataset")
    except Exception as e:
        print(f"❌ Failed to load HF dataset: {e}")
        print("💡 Creating mock conversations for demonstration...")
        
        # Option 2: Create mock conversations for testing
        from datetime import datetime
        from kura.types.conversation import Message
        import uuid
        
        conversations = []
        for i in range(100):
            conv = Conversation(
                chat_id=str(uuid.uuid4()),
                created_at=datetime.now(),
                messages=[
                    Message(
                        created_at=datetime.now(),
                        role="user",
                        content=f"Can you help me with Python programming task {i}?"
                    ),
                    Message(
                        created_at=datetime.now(),
                        role="assistant",
                        content=f"I'd be happy to help with your Python task. Here's how to approach problem {i}..."
                    )
                ],
                metadata={"source": "mock", "task_id": i}
            )
            conversations.append(conv)
        
        print(f"✅ Created {len(conversations)} mock conversations")
    
    # ==========================================================================
    # Step 2: Configuration recommendations
    # ==========================================================================
    
    print(f"\n🔧 Getting VLLM configuration recommendations...")
    
    config_manager = VLLMConfigManager()
    
    # Check if VLLM is cost-effective for this scale
    use_vllm = should_use_vllm(len(conversations), api_cost_per_1k=2.0)
    print(f"📊 VLLM recommended for {len(conversations)} conversations: {use_vllm}")
    
    # Get configuration recommendations
    config_manager.print_recommendation(len(conversations), "embedding")
    config_manager.print_recommendation(len(conversations), "summarization")
    
    # ==========================================================================
    # Step 3: Initialize models with optimal configurations
    # ==========================================================================
    
    print(f"\n🤖 Initializing VLLM models...")
    
    try:
        # Auto-select optimal configurations
        embedding_config, summary_config = auto_select_models(
            len(conversations),
            prefer_speed=True  # For demo purposes
        )
        
        # Initialize VLLM models
        print("🔄 Loading VLLM embedding model...")
        vllm_embedding_model = VLLMEmbeddingModel(**embedding_config)
        print("✅ VLLM embedding model loaded successfully")
        
        print("🔄 Loading VLLM summarization model...")
        vllm_summary_model = VLLMSummaryModel(**summary_config)
        print("✅ VLLM summarization model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load VLLM models: {e}")
        print("❌ VLLM models failed to load. This might be due to:")
        print("   - Missing VLLM installation: pip install -e '.[vllm]'")
        print("   - Insufficient GPU memory")
        print("   - Missing model files")
        print("💡 Falling back to standard models for demonstration...")
        
        # Fallback to standard models
        from kura.embedding import SentenceTransformerEmbeddingModel
        from kura.summarisation import SummaryModel
        
        vllm_embedding_model = SentenceTransformerEmbeddingModel(
            model_name="all-MiniLM-L6-v2",
            model_batch_size=32
        )
        vllm_summary_model = SummaryModel(
            model="openai/gpt-4o-mini",
            max_concurrent_requests=10
        )
        print("✅ Using fallback models for demonstration")
    
    # Initialize other models
    cluster_model = ClusterModel()
    meta_cluster_model = MetaClusterModel(max_clusters=5)
    dimensionality_model = HDBUMAP()
    
    # ==========================================================================
    # Step 4: Set up checkpoint management
    # ==========================================================================
    
    checkpoint_dir = Path("./checkpoints_vllm_demo")
    checkpoint_manager = CheckpointManager(str(checkpoint_dir), enabled=True)
    
    print(f"💾 Using checkpoint directory: {checkpoint_dir}")
    
    # ==========================================================================
    # Step 5: Run the complete pipeline
    # ==========================================================================
    
    print(f"\n🚀 Starting Kura pipeline with VLLM models...")
    
    # Step 5.1: Summarization
    print("📝 Step 1/4: Generating conversation summaries...")
    summaries = await summarise_conversations(
        conversations=conversations,
        model=vllm_summary_model,
        checkpoint_manager=checkpoint_manager
    )
    print(f"✅ Generated {len(summaries)} summaries")
    
    # Step 5.2: Embedding and clustering
    print("🔗 Step 2/4: Generating base clusters...")
    
    # Embed summaries using VLLM embedding model
    summary_texts = [summary.summary for summary in summaries]
    embeddings = await vllm_embedding_model.embed(summary_texts)
    
    # Attach embeddings to summaries
    for summary, embedding in zip(summaries, embeddings):
        summary.embedding = embedding
    
    # Generate clusters
    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries=summaries,
        model=cluster_model,
        checkpoint_manager=checkpoint_manager
    )
    print(f"✅ Generated {len(clusters)} base clusters")
    
    # Step 5.3: Meta-clustering
    print("🌳 Step 3/4: Creating cluster hierarchy...")
    meta_clusters = await reduce_clusters_from_base_clusters(
        clusters=clusters,
        model=meta_cluster_model,
        checkpoint_manager=checkpoint_manager
    )
    root_clusters = [c for c in meta_clusters if c.parent_id is None]
    print(f"✅ Reduced to {len(root_clusters)} root clusters from {len(meta_clusters)} total")
    
    # Step 5.4: Dimensionality reduction for visualization
    print("📊 Step 4/4: Reducing dimensions for visualization...")
    projected_clusters = await reduce_dimensionality_from_clusters(
        clusters=meta_clusters,
        model=dimensionality_model,
        checkpoint_manager=checkpoint_manager
    )
    print(f"✅ Projected {len(projected_clusters)} clusters to 2D space")
    
    # ==========================================================================
    # Step 6: Display results
    # ==========================================================================
    
    print(f"\n📋 Pipeline Results Summary:")
    print(f"   • Input conversations: {len(conversations)}")
    print(f"   • Generated summaries: {len(summaries)}")
    print(f"   • Base clusters: {len([c for c in clusters if c.parent_id is None])}")
    print(f"   • Final root clusters: {len(root_clusters)}")
    print(f"   • Projected clusters: {len(projected_clusters)}")
    
    # Display sample clusters
    print(f"\n🎯 Sample Root Clusters:")
    for i, cluster in enumerate(root_clusters[:3]):
        print(f"   {i+1}. {cluster.label} ({len(cluster.conversation_ids)} conversations)")
        if cluster.summary:
            print(f"      Summary: {cluster.summary[:100]}...")
    
    # ==========================================================================
    # Step 7: Performance comparison (optional)
    # ==========================================================================
    
    if len(conversations) >= 100:  # Only for meaningful comparisons
        print(f"\n⚡ Performance Notes:")
        print(f"   • VLLM models provide better cost efficiency at scale")
        print(f"   • Local processing avoids API rate limits")
        print(f"   • GPU utilization optimized through batching")
        
        # Estimate cost savings
        api_cost = (len(conversations) / 1000) * 2.0  # $2 per 1K conversations
        vllm_hours, vllm_cost = config_manager.estimate_cost_and_time(
            len(conversations), summary_config, "summarization"
        )
        
        print(f"   • Estimated API cost: ${api_cost:.2f}")
        print(f"   • Estimated VLLM cost: ${vllm_cost:.2f}")
        if vllm_cost < api_cost:
            savings = api_cost - vllm_cost
            print(f"   • Potential savings: ${savings:.2f} ({savings/api_cost*100:.1f}%)")
    
    print(f"\n✅ VLLM tutorial completed successfully!")
    print(f"💡 Next steps:")
    print(f"   - Scale up to larger datasets (10k+ conversations)")
    print(f"   - Experiment with different model configurations")
    print(f"   - Deploy on cloud GPU instances for production")
    print(f"   - Monitor GPU utilization and optimize batch sizes")


if __name__ == "__main__":
    asyncio.run(main())