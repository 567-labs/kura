"""Procedural cluster visualization utilities for Kura v1.

This module provides various methods for visualizing hierarchical cluster structures
in the terminal, including basic tree views, enhanced visualizations with statistics,
and rich-formatted output using the Rich library when available.

Compatible with the procedural Kura v1 pipeline approach.
"""

from typing import List, Optional, Union, TYPE_CHECKING, Any, Literal
from pathlib import Path
import logging
from kura.types import Cluster, ClusterTreeNode

if TYPE_CHECKING:
    from rich.console import Console as ConsoleType
else:
    ConsoleType = Any

# Try to import Rich, fall back gracefully if not available
try:
    from rich.console import Console
    from rich.tree import Tree
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.align import Align
    from rich.box import ROUNDED

    RICH_AVAILABLE = True
except ImportError:
    Console = None
    Tree = None
    Table = None
    Panel = None
    Text = None
    Align = None
    ROUNDED = None
    RICH_AVAILABLE = False

# Set up logger
logger = logging.getLogger(__name__)


def _build_tree_structure(
    node: ClusterTreeNode,
    node_id_to_cluster: dict[str, ClusterTreeNode],
    level: int = 0,
    is_last: bool = True,
    prefix: str = "",
) -> str:
    """Build a text representation of the hierarchical cluster tree.

    This is a recursive helper function used by visualise_clusters().

    Args:
        node: Current tree node
        node_id_to_cluster: Dictionary mapping node IDs to nodes
        level: Current depth in the tree (for indentation)
        is_last: Whether this is the last child of its parent
        prefix: Current line prefix for tree structure

    Returns:
        String representation of the tree structure
    """
    # Current line prefix (used for tree visualization symbols)
    current_prefix = prefix

    # Add the appropriate connector based on whether this is the last child
    if level > 0:
        if is_last:
            current_prefix += "╚══ "
        else:
            current_prefix += "╠══ "

    # Print the current node
    result = current_prefix + node.name + " (" + str(node.count) + " conversations)\n"

    # Calculate the prefix for children (continue vertical lines for non-last children)
    child_prefix = prefix
    if level > 0:
        if is_last:
            child_prefix += "    "  # No vertical line needed for last child's children
        else:
            child_prefix += (
                "║   "  # Continue vertical line for non-last child's children
            )

    # Process children
    children = node.children
    for i, child_id in enumerate(children):
        child = node_id_to_cluster[child_id]
        is_last_child = i == len(children) - 1
        result += _build_tree_structure(
            child, node_id_to_cluster, level + 1, is_last_child, child_prefix
        )

    return result


def _build_enhanced_tree_structure(
    node: ClusterTreeNode,
    node_id_to_cluster: dict[str, ClusterTreeNode],
    level: int = 0,
    is_last: bool = True,
    prefix: str = "",
    total_conversations: int = 0,
) -> str:
    """Build an enhanced text representation with colors and better formatting.

    Args:
        node: Current tree node
        node_id_to_cluster: Dictionary mapping node IDs to nodes
        level: Current depth in the tree (for indentation)
        is_last: Whether this is the last child of its parent
        prefix: Current line prefix for tree structure
        total_conversations: Total conversations for percentage calculation

    Returns:
        String representation of the enhanced tree structure
    """
    # Color scheme based on level
    colors = [
        "bright_cyan",
        "bright_green",
        "bright_yellow",
        "bright_magenta",
        "bright_blue",
    ]
    colors[level % len(colors)]

    # Current line prefix (used for tree visualization symbols)
    current_prefix = prefix

    # Add the appropriate connector based on whether this is the last child
    if level > 0:
        if is_last:
            current_prefix += "╚══ "
        else:
            current_prefix += "╠══ "

    # Calculate percentage of total conversations
    percentage = (
        (node.count / total_conversations * 100) if total_conversations > 0 else 0
    )

    # Create progress bar for visual representation
    bar_width = 20
    filled_width = (
        int((node.count / total_conversations) * bar_width)
        if total_conversations > 0
        else 0
    )
    progress_bar = "█" * filled_width + "░" * (bar_width - filled_width)

    # Build the line with enhanced formatting
    result = f"{current_prefix}🔸 {node.name}\n"
    result += f"{prefix}{'║   ' if not is_last and level > 0 else '    '}📊 {node.count:,} conversations ({percentage:.1f}%) [{progress_bar}]\n"

    # Add description if available and not too long
    if (
        hasattr(node, "description")
        and node.description
        and len(node.description) < 100
    ):
        result += f"{prefix}{'║   ' if not is_last and level > 0 else '    '}💭 {node.description}\n"

    result += "\n"

    # Calculate the prefix for children
    child_prefix = prefix
    if level > 0:
        if is_last:
            child_prefix += "    "
        else:
            child_prefix += "║   "

    # Process children
    children = node.children
    for i, child_id in enumerate(children):
        child = node_id_to_cluster[child_id]
        is_last_child = i == len(children) - 1
        result += _build_enhanced_tree_structure(
            child,
            node_id_to_cluster,
            level + 1,
            is_last_child,
            child_prefix,
            total_conversations,
        )

    return result


def _load_clusters_from_checkpoint(checkpoint_path: Union[str, Path]) -> List[Cluster]:
    """Load clusters from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        List of clusters loaded from the checkpoint

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint file is malformed
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        with open(checkpoint_path) as f:
            clusters = [Cluster.model_validate_json(line) for line in f]
        logger.info(f"Loaded {len(clusters)} clusters from {checkpoint_path}")
        return clusters
    except Exception as e:
        raise ValueError(f"Failed to load clusters from {checkpoint_path}: {e}")


def _build_cluster_tree(clusters: List[Cluster]) -> dict[str, ClusterTreeNode]:
    """Build a tree structure from a list of clusters.

    Args:
        clusters: List of clusters to build tree from

    Returns:
        Dictionary mapping cluster IDs to tree nodes
    """
    node_id_to_cluster = {}

    # Create tree nodes
    for cluster in clusters:
        node_id_to_cluster[cluster.id] = ClusterTreeNode(
            id=cluster.id,
            name=cluster.name,
            description=cluster.description,
            slug=cluster.slug,
            count=len(cluster.chat_ids),
            children=[],
        )

    # Link parent-child relationships
    for cluster in clusters:
        if cluster.parent_id:
            node_id_to_cluster[cluster.parent_id].children.append(cluster.id)

    return node_id_to_cluster


def visualise_clusters(
    clusters: Optional[List[Cluster]] = None,
    *,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> None:
    """Print a hierarchical visualization of clusters to the terminal.

    This function loads clusters either from the provided list or from a checkpoint file,
    builds a tree representation, and prints it to the console.
    The visualization shows the hierarchical relationship between clusters
    with indentation and tree structure symbols.

    Args:
        clusters: List of clusters to visualize. If None, loads from checkpoint_path
        checkpoint_path: Path to checkpoint file to load clusters from

    Raises:
        ValueError: If neither clusters nor checkpoint_path is provided
        FileNotFoundError: If checkpoint file doesn't exist

    Example output:
        ╠══ Compare and improve Flutter and React state management (45 conversations)
        ║   ╚══ Improve and compare Flutter and React state management (32 conversations)
        ║       ╠══ Improve React TypeScript application (15 conversations)
        ║       ╚══ Compare and select Flutter state management solutions (17 conversations)
        ╠══ Optimize blog posts for SEO and improved user engagement (28 conversations)
    """
    # Load clusters
    if clusters is None:
        if checkpoint_path is None:
            raise ValueError("Either clusters or checkpoint_path must be provided")
        clusters = _load_clusters_from_checkpoint(checkpoint_path)

    logger.info(f"Visualizing {len(clusters)} clusters")

    # Build tree structure
    node_id_to_cluster = _build_cluster_tree(clusters)

    # Find root nodes and build the tree
    root_nodes = [
        node_id_to_cluster[cluster.id] for cluster in clusters if not cluster.parent_id
    ]

    total_conversations = sum(node.count for node in root_nodes)
    fake_root = ClusterTreeNode(
        id="root",
        name="Clusters",
        description="All clusters",
        slug="all_clusters",
        count=total_conversations,
        children=[node.id for node in root_nodes],
    )

    tree_output = _build_tree_structure(fake_root, node_id_to_cluster, 0, False)
    print(tree_output)


def visualise_clusters_enhanced(
    clusters: Optional[List[Cluster]] = None,
    *,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> None:
    """Print an enhanced hierarchical visualization of clusters with colors and statistics.

    This function provides a more detailed visualization than visualise_clusters(),
    including conversation counts, percentages, progress bars, and descriptions.

    Args:
        clusters: List of clusters to visualize. If None, loads from checkpoint_path
        checkpoint_path: Path to checkpoint file to load clusters from

    Raises:
        ValueError: If neither clusters nor checkpoint_path is provided
        FileNotFoundError: If checkpoint file doesn't exist
    """
    # Load clusters
    if clusters is None:
        if checkpoint_path is None:
            raise ValueError("Either clusters or checkpoint_path must be provided")
        clusters = _load_clusters_from_checkpoint(checkpoint_path)

    logger.info(f"Enhanced visualization of {len(clusters)} clusters")

    print("\n" + "=" * 80)
    print("🎯 ENHANCED CLUSTER VISUALIZATION")
    print("=" * 80)

    # Build tree structure
    node_id_to_cluster = _build_cluster_tree(clusters)

    # Calculate total conversations from root clusters only
    root_clusters = [cluster for cluster in clusters if not cluster.parent_id]
    total_conversations = sum(len(cluster.chat_ids) for cluster in root_clusters)

    # Find root nodes
    root_nodes = [node_id_to_cluster[cluster.id] for cluster in root_clusters]

    fake_root = ClusterTreeNode(
        id="root",
        name=f"📚 All Clusters ({total_conversations:,} total conversations)",
        description="Hierarchical conversation clustering results",
        slug="all_clusters",
        count=total_conversations,
        children=[node.id for node in root_nodes],
    )

    tree_output = _build_enhanced_tree_structure(
        fake_root, node_id_to_cluster, 0, False, "", total_conversations
    )

    print(tree_output)

    # Add summary statistics
    print("=" * 80)
    print("📈 CLUSTER STATISTICS")
    print("=" * 80)
    print(f"📊 Total Clusters: {len(clusters)}")
    print(f"🌳 Root Clusters: {len(root_nodes)}")
    print(f"💬 Total Conversations: {total_conversations:,}")
    print(
        f"📏 Average Conversations per Root Cluster: {total_conversations / len(root_nodes):.1f}"
    )
    print("=" * 80 + "\n")


def visualise_clusters_rich(
    clusters: Optional[List[Cluster]] = None,
    *,
    checkpoint_path: Optional[Union[str, Path]] = None,
    console: Optional[ConsoleType] = None,
) -> None:
    """Print a rich-formatted hierarchical visualization using Rich library.

    This function provides the most visually appealing output with colors,
    interactive-style formatting, and comprehensive statistics when Rich is available.
    Falls back to enhanced visualization if Rich is not available.

    Args:
        clusters: List of clusters to visualize. If None, loads from checkpoint_path
        checkpoint_path: Path to checkpoint file to load clusters from
        console: Rich Console instance. If None, creates a new one or falls back

    Raises:
        ValueError: If neither clusters nor checkpoint_path is provided
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not RICH_AVAILABLE:
        logger.warning("Rich library not available. Using enhanced visualization...")
        visualise_clusters_enhanced(clusters, checkpoint_path=checkpoint_path)
        return

    # Create console if not provided
    if console is None and Console is not None:
        console = Console()

    if console is None:
        logger.warning("Console not available. Using enhanced visualization...")
        visualise_clusters_enhanced(clusters, checkpoint_path=checkpoint_path)
        return

    # Load clusters
    if clusters is None:
        if checkpoint_path is None:
            raise ValueError("Either clusters or checkpoint_path must be provided")
        clusters = _load_clusters_from_checkpoint(checkpoint_path)

    logger.info(f"Rich visualization of {len(clusters)} clusters")

    # Build cluster tree structure
    node_id_to_cluster = _build_cluster_tree(clusters)

    # Calculate total conversations from root clusters only
    root_clusters = [cluster for cluster in clusters if not cluster.parent_id]
    total_conversations = sum(len(cluster.chat_ids) for cluster in root_clusters)

    # Create Rich Tree
    if Tree is None:
        logger.warning(
            "Rich Tree component not available. Using enhanced visualization..."
        )
        visualise_clusters_enhanced(clusters, checkpoint_path=checkpoint_path)
        return

    tree = Tree(
        f"[bold bright_cyan]📚 All Clusters ({total_conversations:,} conversations)[/]",
        style="bold bright_cyan",
    )

    # Add root clusters to tree
    root_nodes = [node_id_to_cluster[cluster.id] for cluster in root_clusters]

    def add_node_to_tree(rich_tree, cluster_node, level=0):
        """Recursively add nodes to Rich tree with formatting."""
        # Color scheme based on level
        colors = [
            "bright_green",
            "bright_yellow",
            "bright_magenta",
            "bright_blue",
            "bright_red",
        ]
        color = colors[level % len(colors)]

        # Calculate percentage
        percentage = (
            (cluster_node.count / total_conversations * 100)
            if total_conversations > 0
            else 0
        )

        # Create progress bar representation
        bar_width = 15
        filled_width = (
            int((cluster_node.count / total_conversations) * bar_width)
            if total_conversations > 0
            else 0
        )
        progress_bar = "█" * filled_width + "░" * (bar_width - filled_width)

        # Create node label with rich formatting
        label = f"[bold {color}]{cluster_node.name}[/] [dim]({cluster_node.count:,} conversations, {percentage:.1f}%)[/]"
        if hasattr(cluster_node, "description") and cluster_node.description:
            short_desc = (
                cluster_node.description[:80] + "..."
                if len(cluster_node.description) > 80
                else cluster_node.description
            )
            label += f"\n[italic dim]{short_desc}[/]"
        label += f"\n[dim]Progress: [{progress_bar}][/]"

        node = rich_tree.add(label)

        # Add children
        for child_id in cluster_node.children:
            child = node_id_to_cluster[child_id]
            add_node_to_tree(node, child, level + 1)

    # Add all root nodes to the tree
    for root_node in sorted(root_nodes, key=lambda x: x.count, reverse=True):
        add_node_to_tree(tree, root_node)

    # Only create tables if Rich components are available
    if Table is None or ROUNDED is None:
        console.print(tree)
        return

    # Create statistics table
    stats_table = Table(
        title="📈 Cluster Statistics", box=ROUNDED, title_style="bold bright_cyan"
    )
    stats_table.add_column("Metric", style="bold bright_yellow")
    stats_table.add_column("Value", style="bright_green")

    stats_table.add_row("📊 Total Clusters", f"{len(clusters):,}")
    stats_table.add_row("🌳 Root Clusters", f"{len(root_nodes):,}")
    stats_table.add_row("💬 Total Conversations", f"{total_conversations:,}")
    stats_table.add_row(
        "📏 Avg per Root Cluster", f"{total_conversations / len(root_nodes):.1f}"
    )

    # Create cluster size distribution table
    size_table = Table(
        title="📊 Cluster Size Distribution",
        box=ROUNDED,
        title_style="bold bright_magenta",
    )
    size_table.add_column("Size Range", style="bold bright_yellow")
    size_table.add_column("Count", style="bright_green")
    size_table.add_column("Percentage", style="bright_blue")

    # Calculate size distribution for root clusters
    root_sizes = [node.count for node in root_nodes]
    size_ranges = [
        ("🔥 Large (>100)", lambda x: x > 100),
        ("📈 Medium (21-100)", lambda x: 21 <= x <= 100),
        ("📊 Small (6-20)", lambda x: 6 <= x <= 20),
        ("🔍 Tiny (1-5)", lambda x: 1 <= x <= 5),
    ]

    for range_name, condition in size_ranges:
        count = sum(1 for size in root_sizes if condition(size))
        percentage = (count / len(root_sizes) * 100) if root_sizes else 0
        size_table.add_row(range_name, f"{count}", f"{percentage:.1f}%")

    # Display everything
    console.print("\n")

    # Only use Panel and Align if they're available
    if Panel is not None and Align is not None and Text is not None:
        console.print(
            Panel(
                Align.center(
                    Text("🎯 RICH CLUSTER VISUALIZATION", style="bold bright_cyan")
                ),
                box=ROUNDED,
                style="bright_cyan",
            )
        )
    else:
        console.print("[bold bright_cyan]🎯 RICH CLUSTER VISUALIZATION[/]")

    console.print("\n")
    console.print(tree)
    console.print("\n")

    # Display tables side by side if Table.grid is available
    if hasattr(Table, "grid"):
        layout = Table.grid(padding=2)
        layout.add_column()
        layout.add_column()
        layout.add_row(stats_table, size_table)
        console.print(layout)
    else:
        # Fallback to printing tables separately
        console.print(stats_table)
        console.print(size_table)

    console.print("\n")


# =============================================================================
# Convenience Functions for Integration with v1 Pipeline
# =============================================================================


def visualise_from_checkpoint_manager(
    checkpoint_manager,
    meta_cluster_model,
    *,
    style: str = "basic",
    console: Optional[ConsoleType] = None,
) -> None:
    """Visualize clusters using a CheckpointManager and meta cluster model.

    This function integrates with the v1 pipeline's CheckpointManager to automatically
    load and visualize clusters.

    Args:
        checkpoint_manager: CheckpointManager instance from v1 pipeline
        meta_cluster_model: Meta cluster model with checkpoint_filename
        style: Visualization style ("basic", "enhanced", or "rich")
        console: Rich Console instance (for rich style)

    Raises:
        ValueError: If invalid style is provided
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not hasattr(meta_cluster_model, "checkpoint_filename"):
        raise ValueError("Meta cluster model must have checkpoint_filename attribute")

    checkpoint_path = checkpoint_manager.get_checkpoint_path(
        meta_cluster_model.checkpoint_filename
    )

    if style == "basic":
        visualise_clusters(checkpoint_path=checkpoint_path)
    elif style == "enhanced":
        visualise_clusters_enhanced(checkpoint_path=checkpoint_path)
    elif style == "rich":
        visualise_clusters_rich(checkpoint_path=checkpoint_path, console=console)
    else:
        raise ValueError(
            f"Invalid style '{style}'. Must be one of: basic, enhanced, rich"
        )


def visualise_pipeline_results(
    clusters: List[Cluster],
    *,
    style: Literal["basic", "enhanced", "rich"] = "enhanced",
    console: Optional[ConsoleType] = None,
) -> None:
    """Visualize clusters that are the result of a pipeline execution.

    Convenience function for visualizing clusters directly from pipeline results.

    Args:
        clusters: List of clusters from pipeline execution
        style: Visualization style ("basic", "enhanced", or "rich")
        console: Rich Console instance (for rich style)

    Raises:
        ValueError: If invalid style is provided
    """
    if style == "basic":
        visualise_clusters(clusters)
    elif style == "enhanced":
        visualise_clusters_enhanced(clusters)
    elif style == "rich":
        visualise_clusters_rich(clusters, console=console)
    else:
        raise ValueError(
            f"Invalid style '{style}'. Must be one of: basic, enhanced, rich"
        )
