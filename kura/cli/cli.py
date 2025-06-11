import typer
import uvicorn
from kura.cli.server import api
from rich import print
import os
from typing import Optional
from pathlib import Path

app = typer.Typer()


@app.command()
def start_app(
    dir: str = typer.Option(
        "./checkpoints",
        help="Directory to use for checkpoints, relative to the current directory",
    ),
    checkpoint_format: str = typer.Option(
        "jsonl",
        help="Checkpoint format to use: 'jsonl' (default, legacy) or 'hf-dataset' (new, recommended for large datasets)",
    ),
):
    """Start the FastAPI server"""
    os.environ["KURA_CHECKPOINT_DIR"] = dir
    os.environ["KURA_CHECKPOINT_FORMAT"] = checkpoint_format
    print(
        f"\n[bold green]🚀 Starting Kura with {checkpoint_format} checkpoints at {dir}[/bold green]"
    )
    print(
        "[bold blue]Access website at[/bold blue] [bold cyan][http://localhost:8000](http://localhost:8000)[/bold cyan]\n"
    )
    uvicorn.run(api, host="0.0.0.0", port=8000)


@app.command()
def migrate_checkpoints(
    source_dir: str = typer.Argument(
        help="Directory containing JSONL checkpoints to migrate"
    ),
    target_dir: str = typer.Argument(
        help="Directory for new HuggingFace dataset checkpoints"
    ),
    hub_repo: Optional[str] = typer.Option(
        None, help="Optional HuggingFace Hub repository name for uploading checkpoints"
    ),
    hub_token: Optional[str] = typer.Option(
        None, help="Optional HuggingFace Hub token for authentication"
    ),
    compression: str = typer.Option(
        "gzip", help="Compression algorithm to use (gzip, lz4, zstd, or none)"
    ),
    delete_source: bool = typer.Option(
        False, help="Delete source JSONL files after successful migration"
    ),
    verify: bool = typer.Option(True, help="Verify migration by comparing data"),
):
    """Migrate JSONL checkpoints to HuggingFace datasets format"""
    from kura.checkpoints.migration import migrate_jsonl_to_hf_dataset, verify_migration

    print(
        f"\n[bold yellow]🔄 Migrating checkpoints from {source_dir} to {target_dir}[/bold yellow]"
    )

    # Check if source directory exists
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"[bold red]❌ Source directory {source_dir} does not exist[/bold red]")
        raise typer.Exit(1)

    # Set compression to None if user specified 'none'
    compression_arg = None if compression.lower() == "none" else compression

    try:
        results = migrate_jsonl_to_hf_dataset(
            source_dir=source_dir,
            target_dir=target_dir,
            hub_repo=hub_repo,
            hub_token=hub_token,
            compression=compression_arg,
            delete_source=delete_source,
        )

        # Display results
        successful = sum(results.values())
        total = len(results)

        if successful == total:
            print(
                f"[bold green]✅ Migration complete: {successful}/{total} checkpoints migrated successfully[/bold green]"
            )
        else:
            print(
                f"[bold yellow]⚠️  Migration partial: {successful}/{total} checkpoints migrated successfully[/bold yellow]"
            )
            for checkpoint, success in results.items():
                if not success:
                    print(f"  [red]❌ Failed: {checkpoint}[/red]")

        # Verify migration if requested
        if verify and successful > 0:
            print("\n[bold blue]🔍 Verifying migration...[/bold blue]")
            verification = verify_migration(source_dir, target_dir, detailed=True)

            verified = verification["verified_checkpoints"]
            total_checkpoints = verification["total_checkpoints"]

            if verified == total_checkpoints:
                print(
                    f"[bold green]✅ Verification complete: {verified}/{total_checkpoints} checkpoints verified[/bold green]"
                )
            else:
                print(
                    f"[bold yellow]⚠️  Verification partial: {verified}/{total_checkpoints} checkpoints verified[/bold yellow]"
                )
                for failure in verification["failed_checkpoints"]:
                    print(f"  [red]❌ {failure}[/red]")

    except Exception as e:
        print(f"[bold red]❌ Migration failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def analyze_checkpoints(
    checkpoint_dir: str = typer.Argument(
        help="Directory containing JSONL checkpoints to analyze"
    ),
):
    """Analyze current checkpoints and estimate migration benefits"""
    from kura.checkpoints.migration import estimate_migration_benefits

    print(f"\n[bold blue]📊 Analyzing checkpoints in {checkpoint_dir}[/bold blue]")

    try:
        stats = estimate_migration_benefits(checkpoint_dir)

        print("\n[bold cyan]Current Setup:[/bold cyan]")
        print(f"  Format: {stats['current_format']}")
        print(f"  Files: {stats['total_files']}")
        print(f"  Size: {stats['total_size_mb']} MB")

        if stats.get("error"):
            print(f"[red]Error: {stats['error']}[/red]")
            return

        print(
            f"\n[bold cyan]Migration Priority: {stats['migration_priority']}[/bold cyan]"
        )

        print("\n[bold green]Estimated Benefits after migration:[/bold green]")
        for benefit, description in stats["estimated_benefits"].items():
            print(f"  • {benefit.replace('_', ' ').title()}: {description}")

        if stats["total_size_mb"] > 1:
            print("\n[bold cyan]Storage Estimates:[/bold cyan]")
            print(f"  Compressed size: ~{stats['estimated_compressed_size_mb']} MB")
            print(
                f"  Space savings: ~{stats['estimated_space_savings_mb']} MB ({int(stats['estimated_space_savings_mb'] / stats['total_size_mb'] * 100)}%)"
            )

    except Exception as e:
        print(f"[bold red]❌ Analysis failed: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
