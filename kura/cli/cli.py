import typer
import uvicorn
from kura.cli.server import api
from rich import print
import os

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


if __name__ == "__main__":
    app()
