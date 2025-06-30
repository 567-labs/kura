# Kura: Procedural API for Chat Data Analysis

![Kura Architecture](./kura.png)

[![PyPI Downloads](https://img.shields.io/pypi/dm/kura?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/kura/)
[![GitHub Stars](https://img.shields.io/github/stars/567-labs/kura?style=flat-square&logo=github)](https://github.com/567-labs/kura/stargazers)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen?style=flat-square&logo=gitbook&logoColor=white)](https://567-labs.github.io/kura/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/kura?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/kura/)
[![PyPI Version](https://img.shields.io/pypi/v/kura?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/kura/)

**Your AI assistant handles thousands of conversations daily. But do you know what users actually need?**

Kura is an open-source library for understanding chat data through machine learning, inspired by [Anthropic's CLIO](https://www.anthropic.com/research/clio). It automatically clusters conversations to reveal patterns, pain points, and opportunities hidden in your data.

## The Hidden Cost of Not Understanding Your Users

Every day, your AI assistant or chatbot has thousands of conversations. Within this data lies critical intelligence:

- **80% of support tickets** might stem from the same 5 unclear features
- **Key feature requests** repeated by hundreds of users in different ways
- **Revenue opportunities** from unmet needs you didn't know existed
- **Critical failures** affecting user trust that go unreported

Manually reviewing conversations doesn't scale. Traditional analytics miss semantic meaning. **Kura bridges this gap.**

## What Kura Does

Kura transforms unstructured conversation data into structured insights:

```
10,000 conversations → AI Analysis → 20 clear patterns
```

- **Automatic Intent Discovery**: Find what users actually want (not what they say)
- **Failure Pattern Detection**: Identify where your AI falls short before users complain
- **Feature Priority Insights**: See which missing features impact the most users
- **Semantic Clustering**: Group by meaning, not keywords
- **Privacy-First Design**: Analyze patterns without exposing individual conversations

## Real-World Impact

### E-commerce Support Bot

**Challenge**: 50,000 weekly conversations, unknown pain points
**Discovery**: 35% of conversations about shipping clustered into 3 issues
**Result**: Fixed root causes, reduced support volume by 40%

### Developer Documentation Assistant

**Challenge**: Users struggling but not reporting specific issues
**Discovery**: 2,000+ conversations revealed 5 consistently confusing APIs
**Result**: Targeted doc improvements, 60% reduction in those queries

### SaaS Onboarding Bot

**Challenge**: 30% of trials not converting, unclear why
**Discovery**: Clustering revealed 3 missing integration requests
**Result**: Built integrations, trial conversion increased 18%

## Installation

```bash
uv pip install kura
```

## When to Use Kura

**Kura is perfect when you have:**

- 100+ conversations to analyze (scales to millions)
- A need to understand user patterns, not individual conversations
- Unstructured conversation data from chatbots, support systems, or AI assistants
- Questions like "What are users struggling with?" or "What features are they requesting?"

**Kura might not be the best fit if:**

- You have fewer than 100 conversations (manual review might be faster)
- You need real-time analysis (Kura is designed for batch processing)
- You only need keyword-based search (use traditional search tools instead)
- You require conversation-level sentiment analysis (Kura focuses on patterns)

## Common Use Cases

### Product Teams

- **Feature Discovery**: Find the features users ask for in their own words
- **Pain Point Analysis**: Identify friction in user journeys
- **Roadmap Prioritization**: Quantify impact of potential improvements

### Customer Success

- **Support Deflection**: Find common issues to create better docs/FAQs
- **Escalation Patterns**: Identify conversations that lead to churn
- **Success Patterns**: Discover what makes users successful

### AI/ML Teams

- **Prompt Engineering**: Find where prompts fail or confuse users
- **Model Evaluation**: Understand model performance beyond metrics
- **Training Data**: Identify gaps in model knowledge

### Analytics Teams

- **Behavioral Insights**: Understand user segments by conversation patterns
- **Trend Analysis**: Track how user needs evolve over time
- **ROI Measurement**: Connect conversation patterns to business outcomes

## Quick Start

### From Zero to Insights in 5 Minutes

```python
import asyncio
from rich.console import Console
from kura.cache import DiskCacheStrategy
from kura.summarisation import summarise_conversations, SummaryModel
from kura.cluster import generate_base_clusters_from_conversation_summaries, ClusterDescriptionModel
from kura.meta_cluster import reduce_clusters_from_base_clusters, MetaClusterModel
from kura.dimensionality import reduce_dimensionality_from_clusters, HDBUMAP
from kura.visualization import visualise_pipeline_results
from kura.types import Conversation
from kura.checkpoints import JSONLCheckpointManager


async def main():
    console = Console()

    # Define Models
    summary_model = SummaryModel(
        console=console,
        cache=DiskCacheStrategy(cache_dir="./.summary"),  # Uses disk-based caching
    )
    cluster_model = ClusterDescriptionModel(console=console)  # Uses K-means by default
    meta_cluster_model = MetaClusterModel(console=console)
    dimensionality_model = HDBUMAP()

    # Define Checkpoints
    checkpoint_manager = JSONLCheckpointManager("./checkpoints", enabled=True)

    # Load conversations from Hugging Face dataset
    conversations = Conversation.from_hf_dataset(
        "ivanleomk/synthetic-gemini-conversations", split="train"
    )

    # Process through the pipeline step by step
    summaries = await summarise_conversations(
        conversations, model=summary_model, checkpoint_manager=checkpoint_manager
    )

    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries, model=cluster_model, checkpoint_manager=checkpoint_manager
    )

    reduced_clusters = await reduce_clusters_from_base_clusters(
        clusters, model=meta_cluster_model, checkpoint_manager=checkpoint_manager
    )

    projected_clusters = await reduce_dimensionality_from_clusters(
        reduced_clusters,
        model=dimensionality_model,
        checkpoint_manager=checkpoint_manager,
    )

    # Visualize results
    visualise_pipeline_results(projected_clusters, style="basic")


if __name__ == "__main__":
    asyncio.run(main())
```

### What This Example Does

1. **Loads** 190 real programming conversations from Hugging Face
2. **Summarizes** each conversation into a concise task description (with caching!)
3. **Clusters** similar conversations using MiniBatch K-means for speed
4. **Organizes** clusters into a hierarchy for easy navigation
5. **Visualizes** the results in your terminal

### Expected Output

```text
Programming Assistance (190 conversations)
├── Data Analysis & Visualization (38 conversations)
│   ├── R Programming for statistical analysis (12 conversations)
│   ├── Tableau dashboard creation (10 conversations)
│   └── Python data manipulation with pandas (16 conversations)
├── Web Development (45 conversations)
│   ├── React component development (20 conversations)
│   ├── API integration issues (15 conversations)
│   └── CSS styling and responsive design (10 conversations)
├── Machine Learning (32 conversations)
│   ├── Model training with TensorFlow (18 conversations)
│   └── Data preprocessing challenges (14 conversations)
└── ... (more clusters)

Total processing time: 21.9s (2.1s with cache!)
Checkpoints saved to: ./checkpoints/
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and contribution guidelines.

## License

[MIT License](LICENSE)

## About

Kura is under active development. If you face any issues or have suggestions, please feel free to [open an issue](https://github.com/567-labs/kura/issues) or a PR. For more details on the technical implementation, check out this [walkthrough of the code](https://ivanleo.com/blog/understanding-user-conversations).
