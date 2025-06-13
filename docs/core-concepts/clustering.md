# Clustering

Kura groups similar conversation summaries into meaningful clusters using semantic similarity. This bottom-up pattern discovery enables identification of dominant themes, understanding diverse user intents, and surfacing "unknown unknown" patterns from large conversational datasets. Clustering transforms individual summaries into organized, interpretable groups.

The clustering process takes `ConversationSummary` objects (from [Summarization](summarization.md)) and produces `Cluster` objects with descriptive names and summaries. Each cluster represents a coherent group of related conversations, making it easier to navigate and understand large-scale conversational data.

---

## Quick Start

Here's the simplest way to cluster conversation summaries:

```python
from kura import generate_base_clusters_from_conversation_summaries
from kura.cluster import ClusterModel
from kura.types import ConversationSummary
import asyncio

# Assume you have summaries from the summarization step
summaries = [...]  # List of ConversationSummary objects


async def main():
    # Initialize the clustering model
    cluster_model = ClusterModel()

    # Cluster the summaries
    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries,
        clustering_model=cluster_model
    )

    # Each cluster contains: name, description, chat_ids
    for cluster in clusters:
        print(f"Cluster: {cluster.name}")
        print(f"Description: {cluster.description}")
        print(f"Conversations: {len(cluster.chat_ids)}")
        print("---")


asyncio.run(main())
```

This automatically

1. Embeds Summaries
2. Groups them using K-means clustering
3. Generates descriptive names and summaries for each cluster

---

## How It Works

Kura clusters conversations in three steps: generate embeddings, group similar summaries, and create descriptive names. This produces human-readable clusters from conversation summaries.

The pipeline produces clusters like:

- **"Help troubleshoot React TypeScript Redux issues"** (3 conversations)
- **"Optimize real-time data pipelines with Spark and Kafka"** (11 conversations)
- **"Assist with data analysis and visualization in Python"** (19 conversations)

---

## Customizing Clustering

Kura's clustering follows a procedural, configurable design where you control behavior through function parameters. You can customize three key aspects:

### 1. Modify the Clustering Model

Different models offer varying performance, cost, and capability trade-offs for generating cluster names and descriptions:

```python
from kura.cluster import ClusterModel

# Use a different model with custom settings
cluster_model = ClusterModel(
    model="anthropic/claude-3-5-sonnet-20241022",
    max_concurrent_requests=20,  # Control API rate limits
    temperature=0.1,  # Lower temperature for more consistent naming
)

clusters = await generate_base_clusters_from_conversation_summaries(
    summaries,
    clustering_model=cluster_model
)
```

### 2. Customize Cluster Prompts

You can modify how clusters are named and described by providing custom prompts:

```python
from kura.cluster import DEFAULT_CLUSTER_PROMPT

# Use default prompt with modifications
custom_prompt = DEFAULT_CLUSTER_PROMPT + """
Focus on technical aspects and programming languages mentioned.
Prioritize framework-specific details in cluster names.
"""

clusters = await generate_base_clusters_from_conversation_summaries(
    summaries,
    clustering_model=cluster_model,
    prompt=custom_prompt,
    max_contrastive_examples=15,  # More contrastive examples for specificity
)
```

You can also completely replace the default prompt:

```python
technical_prompt = """
Analyze the provided conversation summaries and create a cluster description.

Focus specifically on:
- Technical frameworks and libraries mentioned
- Programming languages used
- Problem complexity level
- Solution approaches

Generate a concise name (max 8 words) that captures the technical essence.
Create a two-sentence description focusing on the technical aspects.

Summaries to analyze:
{% for summary in positive_examples %}
- {{ summary.summary }}
{% endfor %}

Contrast with these examples from other clusters:
{% for example in contrastive_examples[:5] %}
- {{ example.summary }}
{% endfor %}
"""

clusters = await generate_base_clusters_from_conversation_summaries(
    summaries,
    prompt=technical_prompt
)
```

### 3. Configure Performance and Visualization

Control concurrency, visualization, and checkpointing:

```python
from rich.console import Console
from kura.checkpoint import CheckpointManager

console = Console()
checkpoint_mgr = CheckpointManager("./cluster_cache", enabled=True)

cluster_model = ClusterModel(
    max_concurrent_requests=50,  # Higher concurrency for faster processing
    console=console,  # Enable rich visualization
)

clusters = await generate_base_clusters_from_conversation_summaries(
    summaries,
    clustering_model=cluster_model,
    checkpoint_manager=checkpoint_mgr,  # Cache results
    max_contrastive_examples=20,  # More examples for better distinction
)
```

The console visualization provides:

- Real-time progress bars during clustering
- Live preview of the latest 3 generated clusters
- Cluster names, descriptions, and conversation counts

---

## Integration with Kura Pipeline

Clustering is the third major step in Kura's analysis pipeline:

1. **Loading:** Conversations are loaded from various sources
2. **Summarization:** Each conversation is summarized using the CLIO framework
3. **Clustering:** Summaries are grouped into meaningful clusters (this step)
4. **Meta-Clustering:** Clusters can be further grouped hierarchically (optional)
5. **Visualization:** Results are explored through interactive interfaces

The procedural API design allows each step to be customized independently while maintaining compatibility with the overall pipeline.
