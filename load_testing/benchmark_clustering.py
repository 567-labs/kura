#!/usr/bin/env python3
"""
Simple load testing script for Kura embedding and clustering.
"""

import asyncio
import json
import time
import logging
import csv
from contextlib import contextmanager
from typing import Dict

from kura.types import ConversationSummary
from kura.embedding import OpenAIEmbeddingModel, embed_summaries
from kura.cluster import (
    KmeansClusteringModel,
    ClusterDescriptionModel,
)
from kura.checkpoint import CheckpointManager

# Optional logfire import for tracing (one-off script dependency)
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None


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


# Configure logfire if available
if LOGFIRE_AVAILABLE:
    logfire.configure(
        send_to_logfire=True,
        token="pylf_v1_us_zqHtQw15s82x8b3dqsRvxP1md9Z1mpVMYC1jrRYqcGVf",
        console=False,
    )
    logfire.instrument_openai()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def load_test_clustering(
    summary_count: int = 100, batch_size: int = 50, **kwargs
) -> dict:
    """
    Load test the clustering pipeline with detailed timing.

    Args:
        summary_count: Number of summaries to test with
        batch_size: Embedding batch size
        **kwargs: Additional configuration options
    """
    timing_manager = TimingManager()

    logger.info(
        f"Starting load test with {summary_count} summaries, batch_size={batch_size}"
    )

    checkpoint_manager = CheckpointManager(
        "/Users/ivanleo/Documents/coding/kura/data/summary_itr"
    )
    summaries = checkpoint_manager.load_checkpoint(
        "summaries.jsonl", ConversationSummary
    )
    if not summaries:
        logger.error("No summaries found in checkpoint")
        return {"error": "No summaries found"}

    # Take subset for testing
    test_summaries = []
    while len(test_summaries) < summary_count:
        test_summaries.extend(
            summaries[: min(summary_count - len(test_summaries), len(summaries))]
        )
    logger.info(f"Using {len(test_summaries)} summaries for testing")

    with timing_manager.timer("configure_models"):
        # Configure models
        embedding_model = OpenAIEmbeddingModel(
            model_name="text-embedding-3-small",
        )

        clustering_method = KmeansClusteringModel(
            clusters_per_group=kwargs.get("clusters_per_group", 10)
        )

        clustering_model = ClusterDescriptionModel(
            model="openai/gpt-4o-mini",
            max_concurrent_requests=kwargs.get("max_concurrent_requests", 10),
            temperature=kwargs.get("temperature", 0.2),
        )

    # Run each step individually with timing
    try:
        with timing_manager.timer("embed_summaries"):
            if LOGFIRE_AVAILABLE:
                with logfire.span("embed_summaries"):
                    embedded_items = await embed_summaries(test_summaries, embedding_model)
            else:
                embedded_items = await embed_summaries(test_summaries, embedding_model)

        with timing_manager.timer("cluster_embeddings"):
            if LOGFIRE_AVAILABLE:
                with logfire.span("cluster_embeddings"):
                    clusters_id_to_summaries = clustering_method.cluster(embedded_items)
            else:
                clusters_id_to_summaries = clustering_method.cluster(embedded_items)

        with timing_manager.timer("generate_cluster_descriptions"):
            if LOGFIRE_AVAILABLE:
                with logfire.span("generate_cluster_descriptions"):
                    clusters = await clustering_model.generate_clusters(
                        cluster_id_to_summaries=clusters_id_to_summaries,
                    )
            else:
                clusters = await clustering_model.generate_clusters(
                    cluster_id_to_summaries=clusters_id_to_summaries,
                )

        # Calculate results
        cluster_sizes = [len(cluster.chat_ids) for cluster in clusters]
        timings = timing_manager.get_timings()

        results = {
            "summary_count": len(test_summaries),
            "batch_size": batch_size,
            "num_clusters": len(clusters),
            "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes)
            if cluster_sizes
            else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "success": True,
            # Add detailed timing information
            "total_time": sum(timings.values()),
            "configure_models_time": timings.get("configure_models", 0),
            "embed_summaries_time": timings.get("embed_summaries", 0),
            "cluster_embeddings_time": timings.get("cluster_embeddings", 0),
            "generate_cluster_descriptions_time": timings.get(
                "generate_cluster_descriptions", 0
            ),
        }

    except Exception as e:
        timings = timing_manager.get_timings()
        results = {
            "summary_count": len(test_summaries),
            "batch_size": batch_size,
            "error": str(e),
            "success": False,
            "total_time": sum(timings.values()),
            "load_summaries_time": timings.get("load_summaries", 0),
            "configure_models_time": timings.get("configure_models", 0),
            "embed_summaries_time": timings.get("embed_summaries", 0),
            "cluster_embeddings_time": timings.get("cluster_embeddings", 0),
            "generate_cluster_descriptions_time": timings.get(
                "generate_cluster_descriptions", 0
            ),
        }

    return results


def save_results_to_csv(results: list[dict], timestamp: str):
    """Save timing results to a CSV file."""
    csv_filename = f"load_test_timing_results_{timestamp}.csv"

    if not results:
        logger.warning("No results to save to CSV")
        return

    # Get all keys from the first result for CSV headers
    fieldnames = list(results[0].keys())

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Detailed timing results saved to {csv_filename}")


async def main():
    """Run load tests with different configurations."""
    test_configs = [
        {
            "summary_count": 100,
        },
        {
            "summary_count": 1000,
        },
        {
            "summary_count": 5000,
        },
        {
            "summary_count": 10000,
        },
    ]

    all_results = []

    for config in test_configs:
        if LOGFIRE_AVAILABLE:
            with logfire.span(f"test_clustering_{config['summary_count']}"):
                logger.info(f"Running test: {config}")
                result = await load_test_clustering(**config)
                all_results.append(result)
        else:
            logger.info(f"Running test: {config}")
            result = await load_test_clustering(**config)
            all_results.append(result)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save JSON results (original format)
    json_output_file = f"load_test_results_{timestamp}.json"
    with open(json_output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {json_output_file}")

    # Save CSV results (detailed timing)
    save_results_to_csv(all_results, timestamp)


if __name__ == "__main__":
    asyncio.run(main())
