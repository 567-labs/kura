#!/usr/bin/env python3
"""
Script to compare file sizes and performance between JSONL and Parquet checkpoint formats.

This script runs the same pipeline with both checkpoint managers and compares:
- File sizes
- Load/save times
- Memory usage (basic measurement)

Usage:
    python scripts/compare_checkpoint_formats.py
"""

import asyncio
import time
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any

# Import Kura components
from kura import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    CheckpointManager,
)

try:
    from kura import ParquetCheckpointManager

    PARQUET_AVAILABLE = True
except ImportError:
    ParquetCheckpointManager = None
    PARQUET_AVAILABLE = False

from kura.types import Conversation
from kura.summarisation import SummaryModel
from kura.cluster import ClusterModel
from kura.meta_cluster import MetaClusterModel
from kura.dimensionality import HDBUMAP
from kura.k_means import MiniBatchKmeansClusteringMethod

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


@contextmanager
def timer():
    """Context manager to time operations."""
    start_time = time.time()
    yield
    end_time = time.time()
    return end_time - start_time


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


def get_directory_size(directory: Path) -> int:
    """Get total size of all files in a directory."""
    total_size = 0
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def get_file_sizes(checkpoint_dir: Path) -> Dict[str, int]:
    """Get sizes of individual checkpoint files."""
    sizes = {}

    # Common checkpoint file patterns
    patterns = [
        "conversations.*",
        "summaries.*",
        "clusters.*",
        "meta_clusters.*",
        "dimensionality.*",
    ]

    for pattern in patterns:
        files = list(checkpoint_dir.glob(pattern))
        if files:
            # Take the first match (should only be one per format)
            file_path = files[0]
            sizes[pattern.replace(".*", "")] = file_path.stat().st_size

    return sizes


async def run_pipeline_with_manager(
    conversations: list, checkpoint_manager, manager_type: str, console: Console
) -> Dict[str, Any]:
    """Run the full pipeline with a given checkpoint manager and measure performance."""

    # Set up models
    summary_model = SummaryModel(console=None)  # Disable progress for cleaner output

    minibatch_kmeans = MiniBatchKmeansClusteringMethod(
        clusters_per_group=10,
        batch_size=100,
        max_iter=50,
        random_state=42,
    )

    cluster_model = ClusterModel(clustering_method=minibatch_kmeans, console=None)
    meta_cluster_model = MetaClusterModel(console=None)
    dimensionality_model = HDBUMAP()

    results = {
        "manager_type": manager_type,
        "times": {},
        "file_sizes": {},
        "total_size": 0,
    }

    console.print(f"\n[bold blue]Running pipeline with {manager_type}...[/bold blue]")

    # Step 1: Summarization
    start_time = time.time()
    summaries = await summarise_conversations(
        conversations, model=summary_model, checkpoint_manager=checkpoint_manager
    )
    results["times"]["summarization"] = time.time() - start_time
    console.print(f"✓ Summarization: {len(summaries)} summaries")

    # Step 2: Base clustering
    start_time = time.time()
    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries, model=cluster_model, checkpoint_manager=checkpoint_manager
    )
    results["times"]["clustering"] = time.time() - start_time
    console.print(f"✓ Clustering: {len(clusters)} clusters")

    # Step 3: Meta clustering
    start_time = time.time()
    reduced_clusters = await reduce_clusters_from_base_clusters(
        clusters, model=meta_cluster_model, checkpoint_manager=checkpoint_manager
    )
    results["times"]["meta_clustering"] = time.time() - start_time
    console.print(f"✓ Meta clustering: {len(reduced_clusters)} clusters")

    # Step 4: Dimensionality reduction
    start_time = time.time()
    projected_clusters = await reduce_dimensionality_from_clusters(
        reduced_clusters,
        model=dimensionality_model,
        checkpoint_manager=checkpoint_manager,
    )
    results["times"]["dimensionality"] = time.time() - start_time
    console.print(
        f"✓ Dimensionality reduction: {len(projected_clusters)} projected clusters"
    )

    # Measure file sizes
    checkpoint_dir = Path(checkpoint_manager.checkpoint_dir)
    results["file_sizes"] = get_file_sizes(checkpoint_dir)
    results["total_size"] = get_directory_size(checkpoint_dir)

    return results


def create_comparison_table(jsonl_results: Dict, parquet_results: Dict) -> Table:
    """Create a Rich table comparing the results."""
    table = Table(title="Checkpoint Format Comparison")

    table.add_column("Metric", style="bold")
    table.add_column("JSONL", justify="right")
    table.add_column("Parquet", justify="right")
    table.add_column("Improvement", justify="right")

    # File size comparisons
    table.add_section()
    table.add_row("[bold]File Sizes[/bold]", "", "", "")

    for file_type in [
        "conversations",
        "summaries",
        "clusters",
        "meta_clusters",
        "dimensionality",
    ]:
        jsonl_size = jsonl_results["file_sizes"].get(file_type, 0)
        parquet_size = parquet_results["file_sizes"].get(file_type, 0)

        if jsonl_size > 0 and parquet_size > 0:
            improvement = (
                f"{(jsonl_size - parquet_size) / jsonl_size * 100:.1f}% smaller"
            )
            f"{jsonl_size / parquet_size:.1f}x"
        elif jsonl_size > 0:
            improvement = "N/A"
        else:
            continue

        table.add_row(
            f"  {file_type}",
            format_size(jsonl_size),
            format_size(parquet_size),
            improvement,
        )

    # Total size
    table.add_row(
        "[bold]Total Size[/bold]",
        format_size(jsonl_results["total_size"]),
        format_size(parquet_results["total_size"]),
        f"{(jsonl_results['total_size'] - parquet_results['total_size']) / jsonl_results['total_size'] * 100:.1f}% smaller",
    )

    # Performance comparisons
    table.add_section()
    table.add_row("[bold]Processing Times[/bold]", "", "", "")

    for step in ["summarization", "clustering", "meta_clustering", "dimensionality"]:
        jsonl_time = jsonl_results["times"].get(step, 0)
        parquet_time = parquet_results["times"].get(step, 0)

        if jsonl_time > 0 and parquet_time > 0:
            if parquet_time < jsonl_time:
                improvement = (
                    f"{(jsonl_time - parquet_time) / jsonl_time * 100:.1f}% faster"
                )
            else:
                improvement = (
                    f"{(parquet_time - jsonl_time) / jsonl_time * 100:.1f}% slower"
                )
        else:
            improvement = "N/A"

        table.add_row(
            f"  {step}", f"{jsonl_time:.2f}s", f"{parquet_time:.2f}s", improvement
        )

    return table


async def main():
    """Main comparison function."""
    console = Console()

    console.print(
        Panel.fit(
            "[bold blue]Kura Checkpoint Format Comparison[/bold blue]\n"
            "This script compares JSONL vs Parquet checkpoint formats",
            title="Benchmark",
        )
    )

    if not PARQUET_AVAILABLE:
        console.print(
            "[red]❌ PyArrow not available. Please install with: pip install pyarrow[/red]"
        )
        return

    # Load a sample dataset
    console.print("\n[bold]Loading sample dataset...[/bold]")
    try:
        conversations = Conversation.from_hf_dataset(
            "ivanleomk/synthetic-gemini-conversations", split="train"
        )
        # Use a subset for faster comparison
        conversations = conversations[:50]  # Smaller subset for demo
        console.print(f"✓ Loaded {len(conversations)} conversations")
    except Exception as e:
        console.print(f"[red]❌ Failed to load dataset: {e}[/red]")
        return

    # Create temporary directories for comparison
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        jsonl_dir = temp_path / "jsonl_checkpoints"
        parquet_dir = temp_path / "parquet_checkpoints"

        jsonl_dir.mkdir()
        parquet_dir.mkdir()

        # Test JSONL format
        console.print("\n[bold cyan]Testing JSONL format...[/bold cyan]")
        jsonl_manager = CheckpointManager(str(jsonl_dir), enabled=True)
        jsonl_results = await run_pipeline_with_manager(
            conversations, jsonl_manager, "JSONL", console
        )

        # Test Parquet format
        console.print("\n[bold green]Testing Parquet format...[/bold green]")
        parquet_manager = ParquetCheckpointManager(str(parquet_dir), enabled=True)
        parquet_results = await run_pipeline_with_manager(
            conversations, parquet_manager, "Parquet", console
        )

        # Display comparison
        console.print("\n")
        comparison_table = create_comparison_table(jsonl_results, parquet_results)
        console.print(comparison_table)

        # Summary
        total_size_reduction = (
            (jsonl_results["total_size"] - parquet_results["total_size"])
            / jsonl_results["total_size"]
            * 100
        )

        console.print("\n[bold green]Summary:[/bold green]")
        console.print(
            f"• Parquet format is {total_size_reduction:.1f}% smaller than JSONL"
        )
        console.print(
            f"• Total space saved: {format_size(jsonl_results['total_size'] - parquet_results['total_size'])}"
        )

        if parquet_results["total_size"] > 0:
            compression_ratio = (
                jsonl_results["total_size"] / parquet_results["total_size"]
            )
            console.print(f"• Compression ratio: {compression_ratio:.1f}x")


if __name__ == "__main__":
    asyncio.run(main())
