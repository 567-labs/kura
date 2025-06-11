from kura.base_classes import BaseClusteringMethod
from sklearn.cluster import KMeans, MiniBatchKMeans
import math
from typing import TypeVar
import numpy as np
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class KmeansClusteringMethod(BaseClusteringMethod):
    def __init__(self, clusters_per_group: int = 10):
        self.clusters_per_group = clusters_per_group
        logger.info(
            f"Initialized KmeansClusteringMethod with clusters_per_group={clusters_per_group}"
        )

    def cluster(self, items: list[T]) -> dict[int, list[T]]:
        """
        We perform a clustering here using an embedding defined on each individual item.

        We assume that the item is passed in as a dictionary with

        - its relevant embedding stored in the "embedding" key.
        - the item itself stored in the "item" key.

        {
            "embedding": list[float],
            "item": any,
        }
        """
        if not items:
            logger.warning("Empty items list provided to cluster method")
            return {}

        logger.info(f"Starting K-means clustering of {len(items)} items")

        try:
            embeddings = [item["embedding"] for item in items]  # pyright: ignore
            data: list[T] = [item["item"] for item in items]  # pyright: ignore
            n_clusters = math.ceil(len(data) / self.clusters_per_group)

            logger.debug(
                f"Calculated {n_clusters} clusters for {len(data)} items (target: {self.clusters_per_group} items per cluster)"
            )

            X = np.array(embeddings)
            logger.debug(f"Created embedding matrix of shape {X.shape}")

            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(X)

            logger.debug(
                f"K-means clustering completed, assigned {len(set(cluster_labels))} unique cluster labels"
            )

            result = {
                i: [data[j] for j in range(len(data)) if cluster_labels[j] == i]
                for i in range(n_clusters)
            }

            # Log cluster size distribution
            cluster_sizes = [len(cluster_items) for cluster_items in result.values()]
            logger.info(
                f"K-means clustering completed: {len(result)} clusters created with sizes {cluster_sizes}"
            )
            logger.debug(
                f"Cluster size stats - min: {min(cluster_sizes)}, max: {max(cluster_sizes)}, avg: {sum(cluster_sizes) / len(cluster_sizes):.1f}"
            )

            return result

        except Exception as e:
            logger.error(
                f"Failed to perform K-means clustering on {len(items)} items: {e}"
            )
            raise


class MiniBatchKmeansClusteringMethod(BaseClusteringMethod):
    """
    MiniBatch K-means clustering implementation optimized for large datasets.
    
    This implementation uses MiniBatchKMeans from scikit-learn, which processes
    data in small batches rather than loading the entire dataset into memory at once.
    This makes it more memory-efficient for large datasets (100k+ items) but may
    produce slightly different results compared to standard K-means due to the
    stochastic nature of mini-batch processing.
    
    Key differences from standard K-means:
    - Lower memory usage: Processes data in configurable batch sizes
    - Faster convergence: Updates centroids incrementally 
    - Slightly less accurate: Results may vary between runs due to randomness
    - Better scalability: Handles large datasets without memory issues
    
    Args:
        clusters_per_group (int): Target number of items per cluster. Defaults to 10.
        batch_size (int): Size of mini-batches for processing. Defaults to 1000.
                         Larger batch sizes use more memory but may be more stable.
        max_iter (int): Maximum number of iterations. Defaults to 100.
        random_state (int): Random seed for reproducibility. Defaults to 42.
    """
    
    def __init__(
        self, 
        clusters_per_group: int = 10,
        batch_size: int = 1000,
        max_iter: int = 100,
        random_state: int = 42
    ):
        self.clusters_per_group = clusters_per_group
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        logger.info(
            f"Initialized MiniBatchKmeansClusteringMethod with clusters_per_group={clusters_per_group}, "
            f"batch_size={batch_size}, max_iter={max_iter}, random_state={random_state}"
        )

    def cluster(self, items: list[T]) -> dict[int, list[T]]:
        """
        Perform MiniBatch K-means clustering on the provided items.

        This method processes items in small batches to reduce memory usage,
        making it suitable for large datasets. Each item should be a dictionary
        containing:
        - 'embedding': list[float] - The vector representation
        - 'item': any - The actual data item to cluster

        Args:
            items: List of dictionaries with 'embedding' and 'item' keys

        Returns:
            Dictionary mapping cluster IDs to lists of clustered items

        Raises:
            Exception: If clustering fails due to invalid input or processing errors
        """
        if not items:
            logger.warning("Empty items list provided to MiniBatch K-means cluster method")
            return {}

        logger.info(f"Starting MiniBatch K-means clustering of {len(items)} items")

        try:
            embeddings = [item["embedding"] for item in items]  # pyright: ignore
            data: list[T] = [item["item"] for item in items]  # pyright: ignore
            n_clusters = math.ceil(len(data) / self.clusters_per_group)

            logger.debug(
                f"Calculated {n_clusters} clusters for {len(data)} items "
                f"(target: {self.clusters_per_group} items per cluster)"
            )

            X = np.array(embeddings)
            logger.debug(f"Created embedding matrix of shape {X.shape}")

            # Use MiniBatchKMeans instead of regular KMeans
            minibatch_kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(self.batch_size, len(data)),  # Don't exceed data size
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_init="auto"  # Let sklearn choose optimal number of initializations
            )
            
            logger.debug(
                f"Using MiniBatch K-means with batch_size={min(self.batch_size, len(data))}, "
                f"max_iter={self.max_iter}"
            )
            
            cluster_labels = minibatch_kmeans.fit_predict(X)

            logger.debug(
                f"MiniBatch K-means clustering completed, assigned {len(set(cluster_labels))} unique cluster labels"
            )

            result = {
                i: [data[j] for j in range(len(data)) if cluster_labels[j] == i]
                for i in range(n_clusters)
            }

            # Log cluster size distribution
            cluster_sizes = [len(cluster_items) for cluster_items in result.values()]
            logger.info(
                f"MiniBatch K-means clustering completed: {len(result)} clusters created with sizes {cluster_sizes}"
            )
            logger.debug(
                f"Cluster size stats - min: {min(cluster_sizes)}, max: {max(cluster_sizes)}, "
                f"avg: {sum(cluster_sizes) / len(cluster_sizes):.1f}"
            )

            return result

        except Exception as e:
            logger.error(
                f"Failed to perform MiniBatch K-means clustering on {len(items)} items: {e}"
            )
            raise
