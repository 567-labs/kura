from kura.base_classes import BaseClusteringMethod
from kura.types.summarisation import ConversationSummary
from sklearn.cluster import KMeans
import math
from typing import Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class KmeansClusteringModel(BaseClusteringMethod):
    def __init__(self, clusters_per_group: int = 10):
        self.clusters_per_group = clusters_per_group
        logger.info(
            f"Initialized KmeansClusteringModel with clusters_per_group={clusters_per_group}"
        )

    def cluster(
        self, items: list[dict[str, Union[ConversationSummary, list[float]]]]
    ) -> dict[int, list[ConversationSummary]]:
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
