#!/usr/bin/env python3
"""
End-to-end benchmarking script for the complete Kura pipeline.

This script runs the full pipeline (summarization -> clustering -> meta-clustering -> dimensionality)
with different models (gpt-4o, gpt-4.1-mini) and dataset sizes (500, 1000, 6000).
"""

import asyncio
import csv
import os
import time
import logging
from contextlib import contextmanager
from typing import Dict, List

import logfire
import dotenv
from rich.console import Console

from kura import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
)
from kura.hdbscan import HDBSCANClusteringMethod
from kura.checkpoints import JSONLCheckpointManager
from kura.types import Conversation
from kura.summarisation import SummaryModel
from kura.cluster import ClusterDescriptionModel
from kura.meta_cluster import MetaClusterModel
from kura.dimensionality import HDBUMAP

dotenv.load_dotenv()

# Configure logfire
logfire.configure(
    send_to_logfire=True,
    token=os.getenv("LOGFIRE_WRITE_TOKEN"),
    console=False,
)
logfire.instrument_openai()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TimingManager:
    """Manager for collecting detailed timing information."""

    def __init__(self):
        self.timings: Dict[str, float] = {}

    @contextmanager
    def timer(self, step_name: str):
        """Context manager to time a step."""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.timings[step_name] = duration
            logger.info(f"{step_name} took {duration:.2f} seconds")

    def get_timings(self) -> Dict[str, float]:
        """Get all recorded timings."""
        return self.timings.copy()

    def reset(self):
        """Reset all timings."""
        self.timings.clear()


async def run_full_pipeline_benchmark(
    dataset_size: int, model_name: str, **kwargs
) -> dict:
    """
    Run the complete Kura pipeline with detailed timing.

    Args:
        dataset_size: Number of conversations to process
        model_name: Model to use (gpt-4o, gpt-4.1-mini)
        **kwargs: Additional configuration options
    """
    timing_manager = TimingManager()
    console = Console()

    # Create checkpoint directory with naming convention
    checkpoint_dir = f"./benchmarks/data_{dataset_size}_{model_name.replace('/', '_').replace('-', '_')}"

    logger.info(
        f"Starting full pipeline benchmark: {dataset_size} conversations, {model_name}"
    )
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    with timing_manager.timer("load_conversations"):
        # Load conversations from dataset
        all_conversations = Conversation.from_conversation_dump(
            "mt_bench_human_judgments.json"
        )
        if not all_conversations:
            return {"error": "No conversations found in dataset", "success": False}

        test_conversations = all_conversations[:dataset_size]
        logger.info(f"Loaded {len(test_conversations)} conversations for testing")

    with timing_manager.timer("configure_models"):
        # Configure models with the specified model name
        clustering_method = HDBSCANClusteringMethod()
        summary_model = SummaryModel(
            console=console,
            model=f"openai/{model_name}",
        )

        cluster_model = ClusterDescriptionModel(
            console=console,
            model=f"openai/{model_name}",
            temperature=kwargs.get("temperature", 0.2),
        )

        meta_cluster_model = MetaClusterModel(
            console=console,
            model=f"openai/{model_name}",
            temperature=kwargs.get("temperature", 0.2),
            clustering_model=clustering_method,
        )

        dimensionality_model = HDBUMAP()

        # Set up checkpointing
        checkpoint_manager = JSONLCheckpointManager(checkpoint_dir, enabled=True)
        checkpoint_manager.save_checkpoint("conversations", test_conversations)

    # Run pipeline steps with individual timing and logfire spans
    try:
        span_name = f"full_pipeline_{dataset_size}_{model_name.replace('/', '_').replace('-', '_')}"

        with logfire.span(span_name):
            with timing_manager.timer("summarise_conversations"):
                with logfire.span("summarise_conversations"):
                    summaries = await summarise_conversations(
                        test_conversations,
                        model=summary_model,
                        checkpoint_manager=checkpoint_manager,
                    )

            with timing_manager.timer("generate_base_clusters"):
                with logfire.span("generate_base_clusters"):
                    clusters = await generate_base_clusters_from_conversation_summaries(
                        summaries,
                        clustering_model=cluster_model,
                        checkpoint_manager=checkpoint_manager,
                        clustering_method=clustering_method,
                    )

            with timing_manager.timer("reduce_clusters"):
                with logfire.span("reduce_clusters"):
                    reduced_clusters = await reduce_clusters_from_base_clusters(
                        clusters,
                        checkpoint_manager=checkpoint_manager,
                        model=meta_cluster_model,
                    )

            with timing_manager.timer("reduce_dimensionality"):
                with logfire.span("reduce_dimensionality"):
                    projected_clusters = await reduce_dimensionality_from_clusters(
                        reduced_clusters,
                        model=dimensionality_model,
                        checkpoint_manager=checkpoint_manager,
                    )

        # Calculate results
        timings = timing_manager.get_timings()

        # Count cluster sizes at each level
        base_cluster_sizes = [len(cluster.chat_ids) for cluster in clusters]
        meta_cluster_sizes = [len(cluster.chat_ids) for cluster in reduced_clusters]

        results = {
            "dataset_size": dataset_size,
            "model_name": model_name,
            "checkpoint_dir": checkpoint_dir,
            "conversation_count": len(test_conversations),
            "summary_count": len(summaries),
            "base_cluster_count": len(clusters),
            "meta_cluster_count": len(reduced_clusters),
            "projected_cluster_count": len(projected_clusters),
            "avg_base_cluster_size": sum(base_cluster_sizes) / len(base_cluster_sizes)
            if base_cluster_sizes
            else 0,
            "avg_meta_cluster_size": sum(meta_cluster_sizes) / len(meta_cluster_sizes)
            if meta_cluster_sizes
            else 0,
            "max_base_cluster_size": max(base_cluster_sizes)
            if base_cluster_sizes
            else 0,
            "max_meta_cluster_size": max(meta_cluster_sizes)
            if meta_cluster_sizes
            else 0,
            "cluster_reduction_ratio": len(reduced_clusters) / len(clusters)
            if clusters
            else 0,
            "success": True,
            # Detailed timing information
            "total_time": sum(timings.values()),
            "load_conversations_time": timings.get("load_conversations", 0),
            "configure_models_time": timings.get("configure_models", 0),
            "summarise_conversations_time": timings.get("summarise_conversations", 0),
            "generate_base_clusters_time": timings.get("generate_base_clusters", 0),
            "reduce_clusters_time": timings.get("reduce_clusters", 0),
            "reduce_dimensionality_time": timings.get("reduce_dimensionality", 0),
            # Performance metrics
            "conversations_per_second": len(test_conversations)
            / timings.get("summarise_conversations", 1),
            "seconds_per_conversation": timings.get("summarise_conversations", 0)
            / len(test_conversations),
        }

    except Exception as e:
        timings = timing_manager.get_timings()
        results = {
            "dataset_size": dataset_size,
            "model_name": model_name,
            "checkpoint_dir": checkpoint_dir,
            "conversation_count": len(test_conversations)
            if "test_conversations" in locals()
            else dataset_size,
            "error": str(e),
            "success": False,
            "total_time": sum(timings.values()),
            "load_conversations_time": timings.get("load_conversations", 0),
            "configure_models_time": timings.get("configure_models", 0),
            "summarise_conversations_time": timings.get("summarise_conversations", 0),
            "generate_base_clusters_time": timings.get("generate_base_clusters", 0),
            "reduce_clusters_time": timings.get("reduce_clusters", 0),
            "reduce_dimensionality_time": timings.get("reduce_dimensionality", 0),
        }
        logger.error(f"Pipeline failed: {e}")

    return results


def save_results_to_csv(results: List[dict], filename: str = "results.csv"):
    """Save benchmark results to CSV file."""
    if not results:
        logger.warning("No results to save to CSV")
        return

    # Get all keys from all results for CSV headers (in case some failed)
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())

    fieldnames = sorted(list(all_keys))

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to {filename}")


async def main():
    """Run comprehensive end-to-end benchmarks."""
    # Configuration matrix
    dataset_sizes = [6000]
    models = ["gpt-4.1-mini"]

    # Additional configuration
    benchmark_config = {
        "max_concurrent_requests": 10,
        "temperature": 0.2,
    }

    all_results = []
    total_tests = len(dataset_sizes) * len(models)
    current_test = 0

    logger.info(f"Starting comprehensive benchmark: {total_tests} total configurations")
    logger.info(f"Dataset sizes: {dataset_sizes}")
    logger.info(f"Models: {models}")

    for dataset_size in dataset_sizes:
        for model_name in models:
            current_test += 1
            test_name = (
                f"{dataset_size}_{model_name.replace('/', '_').replace('-', '_')}"
            )

            logger.info(f"Running test {current_test}/{total_tests}: {test_name}")

            with logfire.span(f"benchmark_{test_name}"):
                result = await run_full_pipeline_benchmark(
                    dataset_size=dataset_size, model_name=model_name, **benchmark_config
                )
                all_results.append(result)

    # Save results to CSV

    csv_filename = "results.csv"
    save_results_to_csv(all_results, csv_filename)

    # Print summary
    print("\n" + "=" * 80)
    print("FULL PIPELINE BENCHMARK RESULTS")
    print("=" * 80)

    for result in all_results:
        if result.get("success"):
            print(f"✅ {result['model_name']} - {result['dataset_size']} conversations")
            print(f"   Total time: {result['total_time']:.1f}s")
            print(f"   Summarization: {result['summarise_conversations_time']:.1f}s")
            print(f"   Base clusters: {result['base_cluster_count']}")
            print(f"   Meta clusters: {result['meta_cluster_count']}")
            print(f"   Throughput: {result['conversations_per_second']:.2f} conv/sec")
            print(f"   Checkpoint: {result['checkpoint_dir']}")
        else:
            print(f"❌ {result['model_name']} - {result['dataset_size']} conversations")
            print(f"   Error: {result.get('error', 'Unknown error')}")
        print("-" * 40)

    print(f"\nDetailed results saved to: {csv_filename}")


if __name__ == "__main__":
    asyncio.run(main())
