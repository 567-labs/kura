from kura.utils import calculate_cluster_levels
from kura.types.dimensionality import ProjectedCluster


def test_calculate_cluster_levels_empty():
    """Test with empty cluster list."""
    clusters = []
    calculate_cluster_levels(clusters)
    assert clusters == []


def test_calculate_cluster_levels_single_item():
    """Test with a single cluster."""
    cluster = ProjectedCluster(
        name="Root Cluster",
        description="A root cluster",
        slug="root_cluster",
        chat_ids=["1", "2"],
        parent_id=None,
        x_coord=1.0,
        y_coord=2.0,
        level=999  # Will be overwritten
    )
    clusters = [cluster]
    calculate_cluster_levels(clusters)
    
    assert cluster.level == 0


def test_calculate_cluster_levels_two_levels():
    """Test parent-child relationship (2 levels)."""
    parent = ProjectedCluster(
        id="parent",
        name="Parent Cluster",
        description="Parent cluster",
        slug="parent_cluster",
        chat_ids=["1", "2"],
        parent_id=None,
        x_coord=1.0,
        y_coord=2.0,
        level=999  # Will be overwritten
    )
    child = ProjectedCluster(
        id="child",
        name="Child Cluster",
        description="Child cluster",
        slug="child_cluster",
        chat_ids=["3", "4"],
        parent_id="parent",
        x_coord=3.0,
        y_coord=4.0,
        level=999  # Will be overwritten
    )

    clusters = [parent, child]
    calculate_cluster_levels(clusters)

    assert parent.level == 0
    assert child.level == 1


def test_calculate_cluster_levels_three_levels():
    """Test grandparent-parent-child relationship (3 levels)."""
    grandparent = ProjectedCluster(
        id="grandparent",
        name="Grandparent Cluster",
        description="Grandparent cluster",
        slug="grandparent_cluster",
        chat_ids=["1", "2"],
        parent_id=None,
        x_coord=1.0,
        y_coord=2.0,
        level=999  # Will be overwritten
    )
    parent = ProjectedCluster(
        id="parent",
        name="Parent Cluster",
        description="Parent cluster",
        slug="parent_cluster",
        chat_ids=["3", "4"],
        parent_id="grandparent",
        x_coord=3.0,
        y_coord=4.0,
        level=999  # Will be overwritten
    )
    child = ProjectedCluster(
        id="child",
        name="Child Cluster",
        description="Child cluster",
        slug="child_cluster",
        chat_ids=["5", "6"],
        parent_id="parent",
        x_coord=5.0,
        y_coord=6.0,
        level=999  # Will be overwritten
    )

    clusters = [grandparent, parent, child]
    calculate_cluster_levels(clusters)

    assert grandparent.level == 0
    assert parent.level == 1
    assert child.level == 2
