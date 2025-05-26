"""Cluster visualization utilities for Kura.

This module provides various methods for visualizing hierarchical cluster structures
in the terminal, including basic tree views, enhanced visualizations with statistics,
and rich-formatted output using the Rich library when available.
"""

from typing import TYPE_CHECKING
from kura.types import Cluster, ClusterTreeNode

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

if TYPE_CHECKING:
    from kura.kura import Kura


class ClusterVisualizer:
    """Handles visualization of hierarchical cluster structures."""
    
    def __init__(self, kura_instance: 'Kura'):
        """Initialize the visualizer with a Kura instance.
        
        Args:
            kura_instance: The Kura instance containing cluster data and configuration
        """
        self.kura = kura_instance
        self.console = kura_instance.console
        self.meta_cluster_model = kura_instance.meta_cluster_model
    
    def _build_tree_structure(
        self,
        node: ClusterTreeNode,
        node_id_to_cluster: dict[str, ClusterTreeNode],
        level: int = 0,
        is_last: bool = True,
        prefix: str = "",
    ) -> str:
        """Build a text representation of the hierarchical cluster tree.
        
        This is a recursive helper method used by visualise_clusters().
        
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
        result = (
            current_prefix + node.name + " (" + str(node.count) + " conversations)\n"
        )

        # Calculate the prefix for children (continue vertical lines for non-last children)
        child_prefix = prefix
        if level > 0:
            if is_last:
                child_prefix += "    "  # No vertical line needed for last child's children
            else:
                child_prefix += "║   "  # Continue vertical line for non-last child's children

        # Process children
        children = node.children
        for i, child_id in enumerate(children):
            child = node_id_to_cluster[child_id]
            is_last_child = i == len(children) - 1
            result += self._build_tree_structure(
                child, node_id_to_cluster, level + 1, is_last_child, child_prefix
            )

        return result

    def visualise_clusters(self):
        """Print a hierarchical visualization of clusters to the terminal.
        
        This method loads clusters from the meta_cluster_checkpoint file,
        builds a tree representation, and prints it to the console.
        The visualization shows the hierarchical relationship between clusters
        with indentation and tree structure symbols.
        
        Example output:
        ╠══ Compare and improve Flutter and React state management (45 conversations)
        ║   ╚══ Improve and compare Flutter and React state management (32 conversations)
        ║       ╠══ Improve React TypeScript application (15 conversations)
        ║       ╚══ Compare and select Flutter state management solutions (17 conversations)
        ╠══ Optimize blog posts for SEO and improved user engagement (28 conversations)
        """
        with open(self.kura.meta_cluster_checkpoint_path) as f:
            clusters = [Cluster.model_validate_json(line) for line in f]

        node_id_to_cluster = {}

        for node in clusters:
            node_id_to_cluster[node.id] = ClusterTreeNode(
                id=node.id,
                name=node.name,
                description=node.description,
                count=len(node.chat_ids),  # Access the actual count value
                children=[],
            )

        for node in clusters:
            if node.parent_id:
                node_id_to_cluster[node.parent_id].children.append(node.id)

        # Find root nodes and build the tree
        tree_output = ""
        root_nodes = [
            node_id_to_cluster[node.id] for node in clusters if not node.parent_id
        ]

        fake_root = ClusterTreeNode(
            id="root",
            name="Clusters",
            description="All clusters",
            count=sum(node.count for node in root_nodes),
            children=[node.id for node in root_nodes],
        )

        tree_output += self._build_tree_structure(
            fake_root, node_id_to_cluster, 0, False
        )

        print(tree_output)

    def _build_enhanced_tree_structure(
        self,
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
        colors = ["bright_cyan", "bright_green", "bright_yellow", "bright_magenta", "bright_blue"]
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
        percentage = (node.count / total_conversations * 100) if total_conversations > 0 else 0
        
        # Create progress bar for visual representation
        bar_width = 20
        filled_width = int((node.count / total_conversations) * bar_width) if total_conversations > 0 else 0
        progress_bar = "█" * filled_width + "░" * (bar_width - filled_width)
        
        # Build the line with enhanced formatting
        result = f"{current_prefix}🔸 {node.name}\n"
        result += f"{prefix}{'║   ' if not is_last and level > 0 else '    '}📊 {node.count:,} conversations ({percentage:.1f}%) [{progress_bar}]\n"
        
        # Add description if available and not too long
        if hasattr(node, 'description') and node.description and len(node.description) < 100:
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
            result += self._build_enhanced_tree_structure(
                child, node_id_to_cluster, level + 1, is_last_child, child_prefix, total_conversations
            )

        return result

    def visualise_clusters_enhanced(self):
        """Print an enhanced hierarchical visualization of clusters with colors and statistics.
        
        This method provides a more detailed visualization than visualise_clusters(),
        including conversation counts, percentages, progress bars, and descriptions.
        """
        print("\n" + "="*80)
        print("🎯 ENHANCED CLUSTER VISUALIZATION")
        print("="*80)
        
        with open(self.kura.meta_cluster_checkpoint_path) as f:
            clusters = [Cluster.model_validate_json(line) for line in f]

        node_id_to_cluster = {}
        total_conversations = sum(len(node.chat_ids) for node in clusters if not node.parent_id)

        for node in clusters:
            node_id_to_cluster[node.id] = ClusterTreeNode(
                id=node.id,
                name=node.name,
                description=node.description,
                count=len(node.chat_ids),  # Access the actual count value
                children=[],
            )

        for node in clusters:
            if node.parent_id:
                node_id_to_cluster[node.parent_id].children.append(node.id)

        # Find root nodes and build the tree
        root_nodes = [
            node_id_to_cluster[node.id] for node in clusters if not node.parent_id
        ]

        fake_root = ClusterTreeNode(
            id="root",
            name=f"📚 All Clusters ({total_conversations:,} total conversations)",
            description="Hierarchical conversation clustering results",
            count=total_conversations,
            children=[node.id for node in root_nodes],
        )

        tree_output = self._build_enhanced_tree_structure(
            fake_root, node_id_to_cluster, 0, False, "", total_conversations
        )

        print(tree_output)
        
        # Add summary statistics
        print("="*80)
        print("📈 CLUSTER STATISTICS")
        print("="*80)
        print(f"📊 Total Clusters: {len(clusters)}")
        print(f"🌳 Root Clusters: {len(root_nodes)}")
        print(f"💬 Total Conversations: {total_conversations:,}")
        print(f"📏 Average Conversations per Root Cluster: {total_conversations/len(root_nodes):.1f}")
        print("="*80 + "\n")

    def visualise_clusters_rich(self):
        """Print a rich-formatted hierarchical visualization using Rich library.
        
        This method provides the most visually appealing output with colors, 
        interactive-style formatting, and comprehensive statistics when Rich is available.
        Falls back to enhanced visualization if Rich is not available.
        """
        if not RICH_AVAILABLE or not self.console:
            print("⚠️  Rich library not available or console disabled. Using enhanced visualization...")
            self.visualise_clusters_enhanced()
            return

        with open(self.kura.meta_cluster_checkpoint_path) as f:
            clusters = [Cluster.model_validate_json(line) for line in f]

        # Build cluster tree structure
        node_id_to_cluster = {}
        total_conversations = sum(len(node.chat_ids) for node in clusters if not node.parent_id)

        for node in clusters:
            node_id_to_cluster[node.id] = ClusterTreeNode(
                id=node.id,
                name=node.name,
                description=node.description,
                count=len(node.chat_ids),  # Access the actual count value
                children=[],
            )

        for node in clusters:
            if node.parent_id:
                node_id_to_cluster[node.parent_id].children.append(node.id)

        # Create Rich Tree
        if Tree is None:
            print("⚠️  Rich Tree component not available. Using enhanced visualization...")
            self.visualise_clusters_enhanced()
            return
            
        tree = Tree(
            f"[bold bright_cyan]📚 All Clusters ({total_conversations:,} conversations)[/]",
            style="bold bright_cyan"
        )

        # Add root clusters to tree
        root_nodes = [
            node_id_to_cluster[node.id] for node in clusters if not node.parent_id
        ]

        def add_node_to_tree(rich_tree, cluster_node, level=0):
            """Recursively add nodes to Rich tree with formatting."""
            # Color scheme based on level
            colors = ["bright_green", "bright_yellow", "bright_magenta", "bright_blue", "bright_red"]
            color = colors[level % len(colors)]
            
            # Calculate percentage
            percentage = (cluster_node.count / total_conversations * 100) if total_conversations > 0 else 0
            
            # Create progress bar representation
            bar_width = 15
            filled_width = int((cluster_node.count / total_conversations) * bar_width) if total_conversations > 0 else 0
            progress_bar = "█" * filled_width + "░" * (bar_width - filled_width)
            
            # Create node label with rich formatting
            label = f"[bold {color}]{cluster_node.name}[/] [dim]({cluster_node.count:,} conversations, {percentage:.1f}%)[/]"
            if hasattr(cluster_node, 'description') and cluster_node.description:
                short_desc = cluster_node.description[:80] + "..." if len(cluster_node.description) > 80 else cluster_node.description
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
            if self.console:
                self.console.print(tree)
            return

        # Create statistics table
        stats_table = Table(title="📈 Cluster Statistics", box=ROUNDED, title_style="bold bright_cyan")
        stats_table.add_column("Metric", style="bold bright_yellow")
        stats_table.add_column("Value", style="bright_green")
        
        stats_table.add_row("📊 Total Clusters", f"{len(clusters):,}")
        stats_table.add_row("🌳 Root Clusters", f"{len(root_nodes):,}")
        stats_table.add_row("💬 Total Conversations", f"{total_conversations:,}")
        stats_table.add_row("📏 Avg per Root Cluster", f"{total_conversations/len(root_nodes):.1f}")
        
        # Create cluster size distribution table
        size_table = Table(title="📊 Cluster Size Distribution", box=ROUNDED, title_style="bold bright_magenta")
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
        if self.console:
            self.console.print("\n")
            
            # Only use Panel and Align if they're available
            if Panel is not None and Align is not None and Text is not None:
                self.console.print(Panel(
                    Align.center(Text("🎯 RICH CLUSTER VISUALIZATION", style="bold bright_cyan")),
                    box=ROUNDED,
                    style="bright_cyan"
                ))
            else:
                self.console.print("[bold bright_cyan]🎯 RICH CLUSTER VISUALIZATION[/]")
                
            self.console.print("\n")
            self.console.print(tree)
            self.console.print("\n")
            
            # Display tables side by side if Table.grid is available
            if hasattr(Table, 'grid'):
                layout = Table.grid(padding=2)
                layout.add_column()
                layout.add_column()
                layout.add_row(stats_table, size_table)
                self.console.print(layout)
            else:
                # Fallback to printing tables separately
                self.console.print(stats_table)
                self.console.print(size_table)
                
            self.console.print("\n")