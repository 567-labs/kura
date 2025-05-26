import time
import asyncio
from contextlib import contextmanager

@contextmanager
def timer(message):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{message} took {end_time - start_time:.2f} seconds")

def show_section_header(title):
    """Display a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}\n")


with timer("Importing kura modules"):
    # Import the procedural Kura v1 components
    from kura.v1 import (
        summarise_conversations,
        generate_base_clusters_from_conversation_summaries,
        reduce_clusters_from_base_clusters,
        reduce_dimensionality_from_clusters,
        CheckpointManager
    )
    
    # Import v1 visualization functions
    from kura.v1.visualization import (
        visualise_clusters_enhanced,
        visualise_clusters_rich,
        visualise_from_checkpoint_manager,
        visualise_pipeline_results
    )
    
    # Import existing Kura models and types
    from kura.types import Conversation
    from kura.summarisation import SummaryModel
    from kura.cluster import ClusterModel
    from kura.meta_cluster import MetaClusterModel
    from kura.dimensionality import HDBUMAP

    from rich.console import Console


# Set up individual models
console = Console()
summary_model = SummaryModel(console=console)
cluster_model = ClusterModel(console=console)
meta_cluster_model = MetaClusterModel(console=console)
dimensionality_model = HDBUMAP()

# Set up checkpointing
checkpoint_manager = CheckpointManager("./tutorial_checkpoints", enabled=True)

with timer("Loading sample conversations"):
    conversations = Conversation.from_hf_dataset(
        "ivanleomk/synthetic-gemini-conversations",
        split="train"
    )

print(f"Loaded {len(conversations)} conversations successfully!\n")

# Save conversations to JSON for database loading
show_section_header("Saving Conversations")

with timer("Saving conversations to JSON"):
    import json
    import os
    
    # Ensure checkpoint directory exists
    os.makedirs("./tutorial_checkpoints", exist_ok=True)
    
    # Convert conversations to JSON format
    conversations_data = [conv.model_dump() for conv in conversations]
    
    # Save to conversations.json
    with open("./tutorial_checkpoints/conversations.json", "w") as f:
        json.dump(conversations_data, f, indent=2, default=str)
    
print(f"Saved {len(conversations)} conversations to tutorial_checkpoints/conversations.json\n")

# Sample conversation examination
show_section_header("Sample Data Examination")

sample_conversation = conversations[0]

# Print conversation details
print("Sample Conversation Details:")
print(f"Chat ID: {sample_conversation.chat_id}")
print(f"Created At: {sample_conversation.created_at}")
print(f"Number of Messages: {len(sample_conversation.messages)}")
print()

# Sample messages
print("Sample Messages:")
for i, msg in enumerate(sample_conversation.messages[:3]):
    content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
    print(f"  {msg.role}: {content_preview}")

print()

# Processing section
show_section_header("Conversation Processing")

print("Starting conversation clustering...")

async def process_with_progress():
    """Process conversations step by step using the procedural API."""
    print("Step 1: Generating conversation summaries...")
    with timer("Conversation summarization"):
        summaries = await summarise_conversations(
            conversations,
            model=summary_model,
            checkpoint_manager=checkpoint_manager
        )
    print(f"Generated {len(summaries)} summaries")
    
    print("Step 2: Generating base clusters from summaries...")
    with timer("Base clustering"):
        clusters = await generate_base_clusters_from_conversation_summaries(
            summaries,
            model=cluster_model,
            checkpoint_manager=checkpoint_manager
        )
    print(f"Generated {len(clusters)} base clusters")
    
    print("Step 3: Reducing clusters hierarchically...")
    with timer("Meta clustering"):
        reduced_clusters = await reduce_clusters_from_base_clusters(
            clusters,
            model=meta_cluster_model,
            checkpoint_manager=checkpoint_manager
        )
    print(f"Reduced to {len(reduced_clusters)} meta clusters")
    
    print("Step 4: Projecting clusters to 2D for visualization...")
    with timer("Dimensionality reduction"):
        projected_clusters = await reduce_dimensionality_from_clusters(
            reduced_clusters,
            model=dimensionality_model,
            checkpoint_manager=checkpoint_manager
        )
    print(f"Generated {len(projected_clusters)} projected clusters")
    
    return reduced_clusters, projected_clusters

reduced_clusters, projected_clusters = asyncio.run(process_with_progress())

print(f"\nPipeline complete! Generated {len(projected_clusters)} projected clusters!\n")

print("Processing Summary:")
print(f"  • Input conversations: {len(conversations)}")
print(f"  • Final reduced clusters: {len(reduced_clusters)}")
print(f"  • Final projected clusters: {len(projected_clusters)}")
print(f"  • Checkpoints saved to: {checkpoint_manager.checkpoint_dir}")
print()

print("="*80)
print("VISUALIZATION DEMONSTRATION")
print("="*80)

print("\n1. Basic cluster visualization (from checkpoint):")
print("-" * 50)
with timer("Basic visualization"):
    visualise_from_checkpoint_manager(
        checkpoint_manager,
        meta_cluster_model,
        style="basic"
    )

print("\n2. Enhanced cluster visualization (from pipeline results):")
print("-" * 50)
with timer("Enhanced visualization"):
    visualise_pipeline_results(
        reduced_clusters,
        style="enhanced"
    )

print("\n3. Rich cluster visualization (with console integration):")
print("-" * 50)
with timer("Rich visualization"):
    visualise_clusters_rich(
        reduced_clusters,
        console=console
    )

print("\n4. Direct checkpoint path visualization:")
print("-" * 50)
checkpoint_path = checkpoint_manager.get_checkpoint_path(meta_cluster_model.checkpoint_filename)
print(f"Loading from: {checkpoint_path}")
with timer("Direct checkpoint visualization"):
    visualise_clusters_enhanced(checkpoint_path=checkpoint_path)

print("="*80)
print("✨ TUTORIAL COMPLETE!")
print("="*80)

print("Procedural API Benefits Demonstrated:")
print("  ✅ Step-by-step processing with individual control")
print("  ✅ Flexible checkpoint management")
print("  • Clear separation of concerns")
print("  • Easy to customize individual steps")
print("  • Multiple visualization options")
print()

print("Visualization Features Demonstrated:")
print("  • Basic hierarchical tree view")
print("  • Enhanced view with statistics and progress bars")
print("  • Rich-formatted output with colors and tables")
print("  • Direct checkpoint integration")
print("  • Pipeline result visualization")
print()

print("CheckpointManager Integration:")
print("  • Automatic checkpoint loading and saving")
print("  • Seamless integration with visualization functions")
print("  • Resume processing from any checkpoint")
print("  • Visualize results without re-running pipeline")
print()

print(f"Check '{checkpoint_manager.checkpoint_dir}' for saved intermediate results!")
print("Try different visualization styles by modifying the 'style' parameter!")
print("Customize visualization by passing different clusters or checkpoint paths!")
