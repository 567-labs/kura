import typer
import subprocess
import uvicorn
from kura.cli.server import api
from rich import print
import yaml
import os
from instructor_classify.schema import LabelDefinition, ClassificationDefinition

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


@app.command()
def generate(
    output_dir: str = typer.Option(
        "./output",
        help="Directory where to generate the output",
    ),
    checkpoint_dir: str = typer.Option(
        "./checkpoints",
        help="Directory to read checkpoint files from",
    ),
):
    """Generate output from checkpoint files"""
    print(f"[bold green]📁 Reading from:[/bold green] {checkpoint_dir}")
    print(f"[bold green]📝 Generating to:[/bold green] {output_dir}")

    labels = [
        LabelDefinition(
            label="topic",
            description="The main high-level topic of the conversation (e.g., 'software development', 'creative writing', 'scientific research').",
        ),
        LabelDefinition(
            label="request",
            description="The user's request or question.",
        ),
    ]

    classification_definition = ClassificationDefinition(
        label_definitions=labels,
        system_message="You are a helpful assistant that classifies conversations into topics and requests.",
    )

    # Run instruct-classify init command
    try:
        subprocess.run(
            ["instruct-classify", "init", output_dir],
            check=True,
        )
        with open(os.path.join(output_dir, "prompt.yaml"), "w") as f:
            yaml.dump(
                {
                    "system_message": classification_definition.system_message,
                    "label_definitions": [
                        {k: v for k, v in label.model_dump().items() if v is not None}
                        for label in classification_definition.label_definitions
                    ],
                },
                f,
            )
        print("[bold green]✅ Successfully initialized output directory[/bold green]")
    except subprocess.CalledProcessError as e:
        print(f"[bold red]❌ Failed to initialize output directory: {e}[/bold red]")
        raise typer.Exit(1)
    except FileNotFoundError:
        print(
            "[bold red]❌ instruct-classify command not found. Is it installed?[/bold red]"
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
