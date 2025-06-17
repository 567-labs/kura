import asyncio
from rich.console import Console
from kura import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    visualise_pipeline_results,
)
from kura.checkpoints import JSONLCheckpointManager
from kura.types import Conversation
from kura.summarisation import SummaryModel
from kura.cluster import ClusterDescriptionModel
from kura.meta_cluster import MetaClusterModel
from kura.embedding import CohereEmbeddingModel
from kura.dimensionality import HDBUMAP
from kura.hdbscan import HDBSCANClusteringMethod
import logfire
import os
import dotenv

dotenv.load_dotenv()

logfire.configure(
    send_to_logfire=True,
    token="pylf_v1_us_5VFJ2kkzVbQ46PpjDwQWQTRsSsb5P1PX3Hx8tnZGjv20",
    console=False,
)
logfire.instrument_openai()


async def main():
    # Initialize models
    console = Console()

    embeddings = CohereEmbeddingModel()

    # SummaryModel now supports caching to speed up re-runs!
    summary_model = SummaryModel(
        console=console,
        model="openai/gpt-4.1-mini",
        # cache_dir=".summary_cache_6000_gpt_4.1_mini",
    )

    cluster_method = HDBSCANClusteringMethod()

    cluster_model = ClusterDescriptionModel(
        model="openai/gpt-4.1-mini",
        console=console,
    )
    meta_cluster_model = MetaClusterModel(
        console=console,
        model="openai/gpt-4.1-mini",
        clustering_model=cluster_method,
        embedding_model=embeddings,
    )
    dimensionality_model = HDBUMAP()

    # Set up checkpointing - you can choose from multiple backends
    # HuggingFace Datasets (advanced features, cloud sync)
    checkpoint_manager = JSONLCheckpointManager(
        "./benchmarks/data_2000_gpt_4.1_mini", enabled=False
    )

    # Load conversations from Hugging Face dataset
    conversations = Conversation.from_conversation_dump(
        "mt_bench_human_judgments.json"
    )[:300]

    checkpoint_manager.save_checkpoint("conversations", conversations)

    summaries = await summarise_conversations(
        conversations, model=summary_model, checkpoint_manager=checkpoint_manager
    )
    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries,
        embedding_model=embeddings,
        clustering_model=cluster_model,
        checkpoint_manager=checkpoint_manager,
        clustering_method=cluster_method,
    )
    reduced_clusters = await reduce_clusters_from_base_clusters(
        clusters,
        checkpoint_manager=checkpoint_manager,
        model=meta_cluster_model,
    )

    projected_clusters = await reduce_dimensionality_from_clusters(
        reduced_clusters,
        model=dimensionality_model,
        checkpoint_manager=checkpoint_manager,
    )

    # Visualize results
    visualise_pipeline_results(projected_clusters, style="basic")


if __name__ == "__main__":
    with logfire.span("main"):
        asyncio.run(main())
