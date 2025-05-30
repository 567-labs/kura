"""Cluster visualization utilities for Kura.

This module provides various methods for visualizing hierarchical cluster structures
in the terminal, including basic tree views, enhanced visualizations with statistics,
and rich-formatted output using the Rich library when available.
"""

from typing import TYPE_CHECKING, List, Optional, Union, Any, Dict
from pathlib import Path
from kura.types import Cluster, ClusterTreeNode
from kura.base_classes.visualization import BaseVisualizationModel

if TYPE_CHECKING:
    from rich.console import Console
    ConsoleType = Optional["Console"]
else:
    ConsoleType = Any

# Try to import Rich, fall back gracefully if not available
try:
    import rich
    import rich.console
    import rich.tree
    import rich.table
    import rich.panel
    import rich.text
    import rich.align
    import rich.box
    RICH_AVAILABLE = True
except ImportError:
    class RichStub:
        """Stub class for rich module when not available."""
        
        class console:
            """Stub for rich.console module."""
            class Console:
                """Stub for rich.console.Console class."""
                def __init__(self, *args, **kwargs):
                    pass
                
                def print(self, *args, **kwargs):
                    print(*args)
        
        class tree:
            """Stub for rich.tree module."""
            class Tree:
                """Stub for rich.tree.Tree class."""
                def __init__(self, *args, **kwargs):
                    pass
                
                def add(self, *args, **kwargs):
                    return self
    
    rich = RichStub()
    RICH_AVAILABLE = False


class ClusterVisualizer(BaseVisualizationModel):
    """Handles visualization of hierarchical cluster structures."""

    def __init__(self, console: Optional[ConsoleType] = None):
        """Initialize the visualizer.

        Args:
            console: Optional Rich Console instance for enhanced output
        """
        if console is not None:
            self.console = console
        elif RICH_AVAILABLE:
            self.console = rich.console.Console()
        else:
            self.console = None
    
    @property
    def checkpoint_filename(self) -> str:
        """The filename used for cluster checkpoints."""
        return "meta_clusters.jsonl"
        
    def _load_clusters_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> List[Cluster]:
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
            return clusters
        except Exception as e:
            raise ValueError(f"Failed to load clusters from {checkpoint_path}: {e}")

    def _build_cluster_tree(self, clusters: List[Cluster]) -> Dict[str, ClusterTreeNode]:
        """Build a tree structure from a list of clusters.

        Args:
            clusters: List of clusters to build tree from

        Returns:
            Dictionary mapping cluster IDs to tree nodes
        """
        node_id_to_cluster = {}

        for cluster in clusters:
            node_id_to_cluster[cluster.id] = ClusterTreeNode(
                id=cluster.id,
                name=cluster.name,
                description=cluster.description,
                slug=cluster.slug,
                count=len(cluster.chat_ids),
                children=[],
            )

        for cluster in clusters:
            if cluster.parent_id:
                node_id_to_cluster[cluster.parent_id].children.append(cluster.id)

        return node_id_to_cluster
        
    def visualize_clusters(
        self,
        clusters: Optional[List[Cluster]] = None,
        *,
        checkpoint_path: Optional[Union[str, Path]] = None,
        style: str = "basic",
        console: Any = None,
    ) -> None:
        """Visualize a list of clusters.
        
        Args:
            clusters: List of clusters to visualize. If None, loads from checkpoint_path
            checkpoint_path: Path to checkpoint file to load clusters from
            style: Visualization style ("basic", "enhanced", or "rich")
            console: Console instance for rich output (overrides instance console)
            
        Raises:
            ValueError: If neither clusters nor checkpoint_path is provided
            FileNotFoundError: If checkpoint file doesn't exist
        """
        output_console = console or self.console
        
        if style == "basic":
            self.visualise_clusters(clusters=clusters, checkpoint_path=checkpoint_path)
        elif style == "enhanced":
            self.visualise_clusters_enhanced(clusters=clusters, checkpoint_path=checkpoint_path)
        elif style == "rich":
            self.visualise_clusters_rich(clusters=clusters, checkpoint_path=checkpoint_path, console=output_console)
        else:
            raise ValueError(f"Invalid style '{style}'. Must be one of: basic, enhanced, rich")
            
    def visualise_clusters(
        self,
        clusters: Optional[List[Cluster]] = None,
        *,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Print a hierarchical visualization of clusters to the terminal.
        
        Args:
            clusters: List of clusters to visualize. If None, loads from checkpoint_path
            checkpoint_path: Path to checkpoint file to load clusters from
            
        Raises:
            ValueError: If neither clusters nor checkpoint_path is provided
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if clusters is None:
            if checkpoint_path is None:
                raise ValueError("Either clusters or checkpoint_path must be provided")
            clusters = self._load_clusters_from_checkpoint(checkpoint_path)
            
        node_id_to_cluster = self._build_cluster_tree(clusters)
        
        root_clusters = [cluster for cluster in clusters if not cluster.parent_id]
        
        for cluster in root_clusters:
            node = node_id_to_cluster[cluster.id]
            self._print_cluster_tree(node, node_id_to_cluster)
            
    def _print_cluster_tree(
        self,
        node: ClusterTreeNode,
        node_id_to_cluster: Dict[str, ClusterTreeNode],
        level: int = 0,
        is_last: bool = True,
        prefix: str = "",
    ) -> None:
        """Print a text representation of the cluster tree.
        
        Args:
            node: Current tree node
            node_id_to_cluster: Dictionary mapping node IDs to tree nodes
            level: Current depth in the tree
            is_last: Whether this is the last child at this level
            prefix: Prefix string for indentation
        """
        branch = "└── " if is_last else "├── "
        
        print(f"{prefix}{branch}{node.name} ({node.count} conversations)")
        
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        children = [node_id_to_cluster[child_id] for child_id in node.children]
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            self._print_cluster_tree(
                child, 
                node_id_to_cluster, 
                level + 1, 
                is_last_child, 
                child_prefix
            )
            
    def visualise_clusters_enhanced(
        self,
        clusters: Optional[List[Cluster]] = None,
        *,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Print an enhanced hierarchical visualization of clusters.
        
        Args:
            clusters: List of clusters to visualize. If None, loads from checkpoint_path
            checkpoint_path: Path to checkpoint file to load clusters from
            
        Raises:
            ValueError: If neither clusters nor checkpoint_path is provided
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if clusters is None:
            if checkpoint_path is None:
                raise ValueError("Either clusters or checkpoint_path must be provided")
            clusters = self._load_clusters_from_checkpoint(checkpoint_path)
            
        node_id_to_cluster = self._build_cluster_tree(clusters)
        
        root_clusters = [cluster for cluster in clusters if not cluster.parent_id]
        
        total_conversations = sum(len(cluster.chat_ids) for cluster in root_clusters)
        
        print(f"\n📚 All Clusters ({total_conversations} conversations)\n")
        
        for cluster in sorted(root_clusters, key=lambda x: len(x.chat_ids), reverse=True):
            node = node_id_to_cluster[cluster.id]
            percentage = (node.count / total_conversations * 100) if total_conversations > 0 else 0
            print(f"📊 {node.name} ({node.count} conversations, {percentage:.1f}%)")
            
            if node.description:
                print(f"   {node.description[:80]}..." if len(node.description) > 80 else f"   {node.description}")
                
            for child_id in node.children:
                child = node_id_to_cluster[child_id]
                child_percentage = (child.count / total_conversations * 100) if total_conversations > 0 else 0
                print(f"  ├── 📌 {child.name} ({child.count} conversations, {child_percentage:.1f}%)")
                
                for grandchild_id in child.children:
                    grandchild = node_id_to_cluster[grandchild_id]
                    grandchild_percentage = (grandchild.count / total_conversations * 100) if total_conversations > 0 else 0
                    print(f"  │   └── 🔹 {grandchild.name} ({grandchild.count} conversations, {grandchild_percentage:.1f}%)")
            
            print()  # Add spacing between root clusters
            
    def visualise_clusters_rich(
        self,
        clusters: Optional[List[Cluster]] = None,
        *,
        checkpoint_path: Optional[Union[str, Path]] = None,
        console: Any = None,
    ) -> None:
        """Print a rich-formatted hierarchical visualization using Rich library.
        
        Args:
            clusters: List of clusters to visualize. If None, loads from checkpoint_path
            checkpoint_path: Path to checkpoint file to load clusters from
            console: Console instance for rich output. If None, uses instance console
            
        Raises:
            ValueError: If neither clusters nor checkpoint_path is provided
            FileNotFoundError: If checkpoint file doesn't exist
        """
        output_console = console or self.console
        
        if not RICH_AVAILABLE or not output_console:
            print("⚠️  Rich library not available or console disabled. Using enhanced visualization...")
            self.visualise_clusters_enhanced(clusters, checkpoint_path=checkpoint_path)
            return
            
        if clusters is None:
            if checkpoint_path is None:
                raise ValueError("Either clusters or checkpoint_path must be provided")
            clusters = self._load_clusters_from_checkpoint(checkpoint_path)
            
        node_id_to_cluster = self._build_cluster_tree(clusters)
        
        root_clusters = [cluster for cluster in clusters if not cluster.parent_id]
        total_conversations = sum(len(cluster.chat_ids) for cluster in root_clusters)
        
        if not RICH_AVAILABLE or not output_console:
            return
            
        tree = rich.tree.Tree(
            f"[bold bright_cyan]📚 All Clusters ({total_conversations:,} conversations)[/]",
            style="bold bright_cyan",
        )
        
        root_nodes = [node_id_to_cluster[cluster.id] for cluster in root_clusters]
        
        def add_node_to_tree(rich_tree, cluster_node, level=0):
            """Recursively add nodes to Rich tree with formatting."""
            colors = [
                "bright_green",
                "bright_yellow",
                "bright_magenta",
                "bright_blue",
                "bright_red",
            ]
            color = colors[level % len(colors)]
            
            percentage = (
                (cluster_node.count / total_conversations * 100)
                if total_conversations > 0
                else 0
            )
            
            label = f"[bold {color}]{cluster_node.name}[/] [dim]({cluster_node.count:,} conversations, {percentage:.1f}%)[/]"
            if hasattr(cluster_node, "description") and cluster_node.description:
                short_desc = (
                    cluster_node.description[:80] + "..."
                    if len(cluster_node.description) > 80
                    else cluster_node.description
                )
                label += f"\n[italic dim]{short_desc}[/]"
                
            node = rich_tree.add(label)
            
            for child_id in cluster_node.children:
                child = node_id_to_cluster[child_id]
                add_node_to_tree(node, child, level + 1)
                
        for root_node in sorted(root_nodes, key=lambda x: x.count, reverse=True):
            add_node_to_tree(tree, root_node)
            
        output_console.print("\n")
        output_console.print("[bold bright_cyan]🎯 RICH CLUSTER VISUALIZATION[/]")
        output_console.print("\n")
        output_console.print(tree)
        output_console.print("\n")
