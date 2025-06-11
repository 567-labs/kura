from kura.types.dimensionality import ProjectedCluster


def batch_texts(texts: list[str], batch_size: int) -> list[list[str]]:
    """Helper function to divide a list of texts into batches.

    Args:
        texts: List of texts to batch
        batch_size: Maximum size of each batch

    Returns:
        List of batches, where each batch is a list of texts
    """
    if not texts:
        return []

    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batches.append(batch)
    return batches


def calculate_cluster_levels(
    clusters: list[ProjectedCluster], parent_id: str | None = None, level: int = 0
) -> list[ProjectedCluster]:
    """Calculate the hierarchical level of each cluster using a top-down approach.

    Root clusters (parent_id=None) are at level 0, their children at level 1, etc.
    This modifies the original cluster objects in-place.

    Args:
        clusters: List of ProjectedCluster objects with id and parent_id attributes
        parent_id: Parent ID to filter clusters by (internal use)
        level: Current level in the hierarchy (internal use)
    """
    if not clusters:
        return clusters

    res = []

    level_clusters = [cluster for cluster in clusters if cluster.parent_id == parent_id]
    for cluster in level_clusters:
        cluster.level = level
        res.append(cluster)
        res.extend(calculate_cluster_levels(clusters, cluster.id, level + 1))

    return res
