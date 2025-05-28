import typer
import subprocess
import uvicorn
import instructor
import asyncio
from kura.cli.server import api
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import yaml
import os
from instructor_classify.schema import (
    ClassificationDefinition,
    LabelDefinition,
    Examples,
)
from pydantic import BaseModel, Field, model_validator, field_validator, ValidationInfo
from kura.types.cluster import Cluster
from kura.types.summarisation import ConversationSummary
from kura.v1.kura import CheckpointManager

app = typer.Typer()


class GeneratedLabel(BaseModel):
    label: str = Field(..., description="The name of the label")
    description: str = Field(..., description="The description of the label")
    cluster_slugs: list[str] = Field(
        ...,
        description="Slugs of 2-3 clusters that best represent this label",
    )

    @field_validator("cluster_slugs")
    def validate_cluster_slugs(
        cls, cluster_slugs: list[str], info: ValidationInfo
    ) -> list[str]:
        valid_slugs = info.context["valid_slugs"]
        invalid_slugs = set(cluster_slugs) - set(valid_slugs)
        if invalid_slugs:
            raise ValueError(
                f"Only use cluster slugs from the provided list: {valid_slugs}"
            )
        return cluster_slugs


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


class GeneratedExamples(BaseModel):
    positive_examples: list[str] = Field(
        ...,
        description="3 positive examples that clearly demonstrate this label",
        min_items=2,
        max_items=3,
    )
    negative_examples: list[str] = Field(
        ...,
        description="3 negative examples that clearly do NOT fit this label",
        min_items=2,
        max_items=3,
    )


def generate_labels_from_clusters(
    meta_clusters: list[Cluster], description: str, is_single_label: bool
) -> GeneratedClassifiers:
    """Generate labels from clusters and meta clusters based on user description."""

    client = instructor.from_provider("openai/gpt-4.1", async_client=False)

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

{% if is_single_label %}
- Only create labels that are explicitly required by the task description
- Create labels that are mutually exclusive - each conversation should be assigned exactly one label
{% else %}
- Only create labels that are explicitly required by the task description 
- Create independent labels - each conversation can be assigned one or more relevant labels. Make sure to reflect this in the generated system prompt too.
{% endif %}

For each label you create, you must cite 2-3 cluster slugs that best represent that label. These will be used down the line to generate few shot examples for the classifier.

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
            "valid_slugs": [cluster.slug for cluster in meta_clusters],
            "description": description,
            "is_single_label": is_single_label,
        },
        response_model=GeneratedClassifiers,
    )
    return resp


async def generate_examples_for_label(
    client: instructor.AsyncInstructor,
    label: GeneratedLabel,
    clusters: list[Cluster],
    summaries: dict[str, ConversationSummary],
) -> LabelDefinition:
    """Generate 3 new examples for a label and return a LabelDefinition with few shot examples."""

    relevant_clusters = [c for c in clusters if c.slug in label.cluster_slugs]
    for cluster in relevant_clusters:
        cluster.chat_ids = cluster.chat_ids[:3]

    relevant_summaries = [
        summaries[chat_id]
        for cluster in relevant_clusters
        for chat_id in cluster.chat_ids
    ]

    response = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert at generating training examples for classification tasks. Create clear, specific examples that demonstrate the label concept.",
            },
            {
                "role": "user",
                "content": """
Generate training examples for the following classification label by analyzing the provided conversation summaries.

<label>
<name>{{ label.label }}</name>
<description>{{ label.description }}</description>
</label>

<conversation_summaries>
{% for summary in conversation_summaries %}
<conversation id="{{ summary.chat_id }}">
<summary>{{ summary.summary }}</summary>
<key_topics>{{ summary.key_topics | join(", ") }}</key_topics>
</conversation>
{% endfor %}
</conversation_summaries>

Based on these conversation summaries and the label definition:

1. Create 3 positive examples - conversations that clearly demonstrate this label
2. Create 3 negative examples - conversations that clearly do NOT fit this label

The examples should be realistic conversation summaries that could plausibly appear in a customer support or conversation dataset. Make them specific and clear so they can effectively train a classifier.
""",
            },
        ],
        context={
            "label": label,
            "conversation_summaries": relevant_summaries,
        },
        response_model=GeneratedExamples,
    )

    result = LabelDefinition(
        label=label.label,
        description=label.description,
        examples=Examples(
            examples_positive=response.positive_examples,
            examples_negative=response.negative_examples,
        ),
    )
    return result


async def generate_all_examples(
    labels: list[GeneratedLabel],
    clusters: list[Cluster],
    summaries: dict[str, ConversationSummary],
) -> list[LabelDefinition]:
    """Generate examples for all labels concurrently."""
    client = instructor.from_provider("openai/gpt-4.1", async_client=True)
    tasks = [
        generate_examples_for_label(client, label, clusters, summaries)
        for label in labels
    ]
    return await asyncio.gather(*tasks)


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
    single: bool = typer.Option(
        True,
        help="Whether to create a single-label (True) or multi-label (False) classifier",
    ),
):
    """Generate output from checkpoint files"""
    print(f"[bold green]📁 Reading from:[/bold green] {checkpoint_dir}")
    print(f"[bold green]📝 Generating to:[/bold green] {output_dir}")

    # Load clusters and conversations from checkpoint directory using CheckpointManager
    try:
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        meta_clusters = checkpoint_manager.load_checkpoint(
            "meta_clusters.jsonl", Cluster
        )
        conversation_summaries = checkpoint_manager.load_checkpoint(
            "summaries.jsonl", ConversationSummary
        )

        if meta_clusters is None or conversation_summaries is None:
            print(
                f"[bold yellow]⚠️  No meta_clusters.jsonl found in {checkpoint_dir}[/bold yellow]"
            )
            raise typer.Exit(1)

    except Exception as e:
        print(f"[bold yellow]⚠️  Could not load data: {e}[/bold yellow]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=Console()
    ) as progress:
        # Generate labels
        task = progress.add_task("🏷️  Generating labels...", total=None)
        generated_classifier = generate_labels_from_clusters(
            meta_clusters, classifier_description, single
        )
        progress.remove_task(task)
        
        print(f"[bold green]✅ Generated {len(generated_classifier.labels)} labels[/bold green]")
        
        # Generate few shot examples
        task = progress.add_task(f"🎯 Generating few shot examples for {len(generated_classifier.labels)} labels...", total=None)
        summary_mapping = {conv.chat_id: conv for conv in conversation_summaries}
        labels = asyncio.run(
            generate_all_examples(
                generated_classifier.labels, meta_clusters, summary_mapping
            )
        )
        progress.remove_task(task)

    classification_definition = ClassificationDefinition(
        label_definitions=labels,
        system_message=generated_classifier.system_prompt,
        classification_type="single" if single else "multi",
    )

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
                        "examples": {
                            "positive_examples": label.examples.examples_positive,
                            "negative_examples": label.examples.examples_negative,
                        },
                    }
                    for label in classification_definition.label_definitions
                ],
                "classification_type": classification_definition.classification_type,
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
    except Exception as e:
        print(f"[bold red]❌ Failed to generate output: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
