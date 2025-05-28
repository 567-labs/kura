import typer
import subprocess
import uvicorn
import instructor
from kura.cli.server import api
from rich import print
import yaml
import os
from instructor_classify.schema import (
    ClassificationDefinition,
    LabelDefinition,
)
from pydantic import BaseModel, Field, model_validator
from kura.types.cluster import Cluster
from kura.v1.kura import CheckpointManager

app = typer.Typer()


class GeneratedLabel(BaseModel):
    label: str = Field(..., description="The name of the label")
    description: str = Field(..., description="The description of the label")


class GeneratedClassifiers(BaseModel):
    chain_of_thought: str = Field(
        ..., description="The chain of thought used to generate the classifiers"
    )
    system_prompt: str = Field(
        ...,
        description="The system prompt used for a classifier which will be used to classify conversations according to the labels you generate",
    )
    labels: list[GeneratedLabel] = Field(
        ..., description="Labels that will be used to classify conversations"
    )

    @model_validator(mode="after")
    def validate_labels(self) -> "GeneratedClassifiers":
        if len(self.labels) == 0:
            raise ValueError("No labels generated")
        return self


def generate_labels_from_clusters(
    meta_clusters: list[Cluster], description: str
) -> GeneratedClassifiers:
    """Generate labels from clusters and meta clusters based on user description."""

    client = instructor.from_provider("openai/gpt-4.1")

    resp = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating classification labels for conversations. Your goal is to analyze conversation clusters and create the minimum necessary set of clear, specific labels that effectively categorize the conversations.",
            },
            {
                "role": "user",
                "content": """
Your primary task is described below. Read it carefully as it contains the specific requirements for the classification system you need to create.

<task>
{{ description }}
</task>

Based on this task description, you will analyze the following conversation clusters and create the minimum number of labels needed to satisfy the requirements:

<clusters>
{% for cluster in clusters %}
<cluster>
<name>{{cluster.name}}</name>
<description>{{cluster.description}}</description>
<slug>{{cluster.slug}}</slug>
</cluster>
{% endfor %}
</clusters>

Important:
1. Only create labels that are explicitly required by the task description
2. Ensure labels are mutually exclusive to avoid classification ambiguity

""",
            },
        ],
        context={
            "clusters": [
                {
                    "name": cluster.name,
                    "description": cluster.description,
                    "slug": cluster.slug,
                }
                for cluster in meta_clusters
            ],
            "description": description,
        },
        response_model=GeneratedClassifiers,
    )
    return resp


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
    classifier_description: str = typer.Option(
        ...,
        help="Description of the classifiers you'd like to generate",
    ),
):
    """Generate output from checkpoint files"""
    print(f"[bold green]📁 Reading from:[/bold green] {checkpoint_dir}")
    print(f"[bold green]📝 Generating to:[/bold green] {output_dir}")

    # Load clusters from checkpoint directory using CheckpointManager
    try:
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        meta_clusters = checkpoint_manager.load_checkpoint(
            "meta_clusters.jsonl", Cluster
        )

        if meta_clusters is None:
            print(
                f"[bold yellow]⚠️  No meta_clusters.jsonl found in {checkpoint_dir}[/bold yellow]"
            )
            raise typer.Exit(1)
    except Exception as e:
        print(f"[bold yellow]⚠️  Could not load clusters: {e}[/bold yellow]")
        raise typer.Exit(1)

    generated_classifier = generate_labels_from_clusters(
        meta_clusters, classifier_description
    )

    classification_definition = ClassificationDefinition(
        label_definitions=[
            LabelDefinition(
                label=label.label,
                description=label.description,
            )
            for label in generated_classifier.labels
        ],
        system_message=generated_classifier.system_prompt,
    )

    # Run instruct-classify init command
    try:
        subprocess.run(
            ["instruct-classify", "init", output_dir],
            check=True,
        )
        with open(os.path.join(output_dir, "prompt.yaml"), "w") as f:
            # Create nicer YAML structure
            yaml_content = {
                "system_message": classification_definition.system_message,
                "label_definitions": [
                    {
                        "label": label.label,
                        "description": label.description,
                    }
                    for label in classification_definition.label_definitions
                ],
            }
            yaml.dump(
                yaml_content,
                f,
                indent=4,
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
