from kura.base_classes import BaseClusteringMethod
import hdbscan
from typing import TypeVar
import numpy as np
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class HDBSCANClusteringMethod(BaseClusteringMethod):
    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
        alpha: float = 1.0,
        cluster_selection_method: str = "eom",
        metric: str = "euclidean",
        **kwargs,
    ):
        """
        Initialize HDBSCAN clustering method.

        Args:
            min_cluster_size: The minimum size of clusters; single linkage splits that contain
                            fewer points than this will be considered points "falling out" of a cluster
            min_samples: The number of samples in a neighbourhood for a point to be considered
                        a core point. If None, defaults to min_cluster_size
            cluster_selection_epsilon: A distance threshold. Clusters below this value will be merged
            alpha: A distance scaling parameter as used in robust single linkage
            cluster_selection_method: The method used to select clusters from the tree
                                    ('eom' for Excess of Mass, 'leaf' for leaf clustering)
            metric: The metric to use when calculating distance between instances in a feature array
            **kwargs: Additional parameters to pass to HDBSCAN
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.alpha = alpha
        self.cluster_selection_method = cluster_selection_method
        self.metric = metric
        self.kwargs = kwargs

        logger.info(
            f"Initialized HDBSCANClusteringMethod with min_cluster_size={min_cluster_size}, "
            f"min_samples={self.min_samples}, cluster_selection_epsilon={cluster_selection_epsilon}, "
            f"alpha={alpha}, cluster_selection_method={cluster_selection_method}, metric={metric}"
        )

    def cluster(self, items: list[T]) -> dict[int, list[T]]:
        """
        Perform HDBSCAN clustering on the provided items.

        We assume that the item is passed in as a dictionary with:
        - its relevant embedding stored in the "embedding" key.
        - the item itself stored in the "item" key.

        {
            "embedding": list[float],
            "item": any,
        }

        Returns:
            A dictionary mapping cluster IDs to lists of items in that cluster.
            Noise points (outliers) are assigned to cluster ID -1.
        """
        if not items:
            logger.warning("Empty items list provided to cluster method")
            return {}

        logger.info(f"Starting HDBSCAN clustering of {len(items)} items")

        try:
            embeddings = [item["embedding"] for item in items]  # pyright: ignore
            data: list[T] = [item["item"] for item in items]  # pyright: ignore

            logger.debug(f"Extracted embeddings for {len(data)} items")

            X = np.array(embeddings)
            logger.debug(f"Created embedding matrix of shape {X.shape}")

            # Initialize HDBSCAN clusterer
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                alpha=self.alpha,
                cluster_selection_method=self.cluster_selection_method,
                metric=self.metric,
                **self.kwargs,
            )

            # Perform clustering
            cluster_labels = clusterer.fit_predict(X)

            logger.debug(
                f"HDBSCAN clustering completed, assigned {len(set(cluster_labels))} unique cluster labels"
            )

            # Group items by cluster label
            result = {}
            for i, label in enumerate(cluster_labels):
                if label not in result:
                    result[label] = []
                result[label].append(data[i])

            # Convert noise cluster (-1) to a positive cluster ID if it exists
            # This ensures compatibility with the rest of the system
            if -1 in result:
                noise_items = result.pop(-1)
                # Find the next available cluster ID
                max_cluster_id = max(result.keys()) if result else -1
                noise_cluster_id = max_cluster_id + 1
                result[noise_cluster_id] = noise_items
                logger.info(
                    f"Reassigned {len(noise_items)} noise points to cluster {noise_cluster_id}"
                )

            # Log cluster size distribution
            cluster_sizes = [len(cluster_items) for cluster_items in result.values()]
            noise_count = len([label for label in cluster_labels if label == -1])

            logger.info(
                f"HDBSCAN clustering completed: {len(result)} clusters created with sizes {cluster_sizes}"
            )
            if noise_count > 0:
                logger.info(f"Found {noise_count} noise points (outliers)")

            if cluster_sizes:
                logger.debug(
                    f"Cluster size stats - min: {min(cluster_sizes)}, max: {max(cluster_sizes)}, "
                    f"avg: {sum(cluster_sizes) / len(cluster_sizes):.1f}"
                )

            return result

        except Exception as e:
            logger.error(
                f"Failed to perform HDBSCAN clustering on {len(items)} items: {e}"
            )
            raise
