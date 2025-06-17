#!/usr/bin/env python3
"""
Simple load testing script for Kura meta clustering (reduce clusters from base clusters).
"""

import asyncio
import json
import os
import time
import logging
import csv
from contextlib import contextmanager
from typing import Dict


from kura.hdbscan import HDBSCANClusteringMethod
from kura.meta_cluster import MetaClusterModel
from kura import reduce_clusters_from_base_clusters
from kura.checkpoints import JSONLCheckpointManager
import logfire

logfire.configure(
    send_to_logfire=True,
    token=os.getenv("LOGFIRE_WRITE_TOKEN"),
    console=False,
)
logfire.instrument_openai()


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


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def load_test_meta_clustering(
    cluster_dataset: str = "100", batch_size: int = 50, **kwargs
) -> dict:
    """
    Load test the meta clustering pipeline with detailed timing.

    This tests the reduce_clusters_from_base_clusters function which takes
    base clusters and groups them into higher-level meta clusters.

    Args:
        cluster_dataset: Dataset size identifier ("100", "1000", "6000")
        batch_size: Embedding batch size
        **kwargs: Additional configuration options
    """
    timing_manager = TimingManager()

    logger.info(
        f"Starting meta clustering load test with dataset {cluster_dataset}, batch_size={batch_size}"
    )

    checkpoint_manager = JSONLCheckpointManager(
        f"/Users/ivanleo/Documents/coding/kura/data/benchmarks_{cluster_dataset}"
    )

    # Load existing clusters instead of generating from scratch
    from kura.types.cluster import Cluster

    clusters = checkpoint_manager.load_checkpoint("clusters", Cluster)
    if not clusters:
        logger.error("No clusters found in checkpoint")
        return {"error": "No clusters found"}

    logger.info(f"Loaded {len(clusters)} existing clusters for meta clustering test")

    with timing_manager.timer("configure_models"):
        # Configure models

        clustering_method = HDBSCANClusteringMethod()

        meta_cluster_model = MetaClusterModel(
            model="openai/gpt-4o-mini",
            max_concurrent_requests=kwargs.get("max_concurrent_requests", 10),
            temperature=kwargs.get("temperature", 0.2),
            clustering_model=clustering_method,
        )

    # Test the meta clustering step using existing clusters
    try:
        with timing_manager.timer("reduce_clusters_from_base_clusters"):
            with logfire.span("reduce_clusters_from_base_clusters"):
                meta_clusters = await reduce_clusters_from_base_clusters(
                    clusters,  # Use loaded clusters directly
                    checkpoint_manager=checkpoint_manager,
                    model=meta_cluster_model,
                )

        # Calculate results
        base_cluster_sizes = [len(cluster.chat_ids) for cluster in clusters]
        meta_cluster_sizes = [len(cluster.chat_ids) for cluster in meta_clusters]
        timings = timing_manager.get_timings()

        results = {
            "base_cluster_count": len(clusters),
            "batch_size": batch_size,
            "num_base_clusters": len(clusters),
            "num_meta_clusters": len(meta_clusters),
            "cluster_reduction_ratio": len(meta_clusters) / len(clusters)
            if clusters
            else 0,
            "avg_base_cluster_size": sum(base_cluster_sizes) / len(base_cluster_sizes)
            if base_cluster_sizes
            else 0,
            "avg_meta_cluster_size": sum(meta_cluster_sizes) / len(meta_cluster_sizes)
            if meta_cluster_sizes
            else 0,
            "min_base_cluster_size": min(base_cluster_sizes)
            if base_cluster_sizes
            else 0,
            "max_base_cluster_size": max(base_cluster_sizes)
            if base_cluster_sizes
            else 0,
            "min_meta_cluster_size": min(meta_cluster_sizes)
            if meta_cluster_sizes
            else 0,
            "max_meta_cluster_size": max(meta_cluster_sizes)
            if meta_cluster_sizes
            else 0,
            "success": True,
            # Add detailed timing information
            "total_time": sum(timings.values()),
            "configure_models_time": timings.get("configure_models", 0),
            "reduce_clusters_from_base_clusters_time": timings.get(
                "reduce_clusters_from_base_clusters", 0
            ),
        }

    except Exception as e:
        timings = timing_manager.get_timings()
        results = {
            "base_cluster_count": len(clusters),
            "batch_size": batch_size,
            "error": str(e),
            "success": False,
            "total_time": sum(timings.values()),
            "configure_models_time": timings.get("configure_models", 0),
            "reduce_clusters_from_base_clusters_time": timings.get(
                "reduce_clusters_from_base_clusters", 0
            ),
        }

    return results


def save_results_to_csv(results: list[dict], timestamp: str):
    """Save timing results to a CSV file."""
    csv_filename = f"meta_clustering_load_test_results_{timestamp}.csv"

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
    """Run meta clustering load tests with different configurations."""
    test_configs = [
        {
            "cluster_dataset": "100",
        },
        {
            "cluster_dataset": "1000",
        },
        {
            "cluster_dataset": "6000",
        },
    ]

    all_results = []

    for config in test_configs:
        with logfire.span(f"test_meta_clustering_{config['cluster_dataset']}"):
            logger.info(f"Running meta clustering test: {config}")
            result = await load_test_meta_clustering(**config)
            all_results.append(result)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save JSON results (original format)
    json_output_file = f"meta_clustering_load_test_results_{timestamp}.json"
    with open(json_output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {json_output_file}")

    # Save CSV results (detailed timing)
    save_results_to_csv(all_results, timestamp)

    # Print summary
    print("\n=== META CLUSTERING BENCHMARK RESULTS ===")
    for result in all_results:
        if result.get("success"):
            print(f"Base clusters: {result['num_base_clusters']}")
            print(f"Meta clusters: {result['num_meta_clusters']}")
            print(f"Reduction ratio: {result['cluster_reduction_ratio']:.2f}")
            print(f"Total time: {result['total_time']:.2f}s")
            print(
                f"Meta clustering time: {result['reduce_clusters_from_base_clusters_time']:.2f}s"
            )
            print(f"Avg base cluster size: {result['avg_base_cluster_size']:.1f}")
            print(f"Avg meta cluster size: {result['avg_meta_cluster_size']:.1f}")
            print("---")
        else:
            print(
                f"FAILED - Base clusters: {result['cluster_dataset']}, Error: {result.get('error')}"
            )
            print("---")


if __name__ == "__main__":
    asyncio.run(main())
