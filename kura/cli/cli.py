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
):
    """Start the FastAPI server"""
    os.environ["KURA_CHECKPOINT_DIR"] = dir
    print(
        "\n[bold green]🚀 Access website at[/bold green] [bold blue][http://localhost:8000](http://localhost:8000)[/bold blue]\n"
    )
    uvicorn.run(api, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    app()
