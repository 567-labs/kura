# Kura: Procedural API for Chat Data Analysis

![Kura Architecture](assets/images/kura-architecture.png)

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

- **80% of support tickets** stem from 5 unclear features
- **Feature requests** repeated by hundreds of users differently
- **Revenue opportunities** from unmet needs
- **Critical failures** affecting user trust

Make sense of that data with Kura today

```bash
# Install from PyPI
pip install kura

# Or use uv for faster installation
uv pip install kura
```

### What Kura Does

!!! note "Kura transforms chaos into clarity"

    Imagine having 10,000 scattered conversations and ending up with 20 crystal-clear patterns that tell you exactly what your users need. That's what Kura does.

Kura is built for scale and flexibility, processing your conversation data through a sophisticated four-stage pipeline.

1. **Smart caching** makes re-runs 85x faster
2. **Checkpointing system** never loses progress
3. **Parallel processing** handles thousands of conversations,
4. **Extensible design** works with any model (OpenAI, Anthropic, local)

We also provide a web-ui that ships with the package to visualise the different clusters that we've extracted.

#### **Summarization**

Extract core intent from each conversation. Works with any conversation length - from quick questions to complex multi-turn dialogues. Uses AI to distill the essence while preserving critical context and user intent.

Transforms: _"I've been trying to cancel my subscription for 30 minutes and the button doesn't work and I'm really frustrated..."_ **→** _"Cancel subscription"_

#### **Semantic Clustering**

Group by meaning, not keywords. The AI understands synonyms, context, and user intent across different phrasings and languages.

Transforms: _"cancel subscription"_, _"stop my account"_, _"end my service"_, _"how do I delete my profile?"_, _"terminate my membership"_ **→** _Single cluster: "Account Cancellation"_

#### **Meta-Clustering**

Build hierarchy of insights. Creates multiple levels of organization: individual issues → feature categories → business themes.

Transforms: _"Account Cancellation"_, _"Login Problems"_, _"Password Resets"_ **→** _"Account Management Issues" (40% of support load)_

#### **Dimensionality Reduction**

Create interactive exploration maps. See clusters as bubbles on a 2D map where proximity indicates similarity. Discover edge cases, identify emerging trends, and spot conversations that don't fit existing patterns.

Transforms: _High-dimensional cluster embeddings_ **→** _Interactive 2D visualization map_

**The result?** Instead of drowning in individual conversations, you get a clear picture of what's actually happening across your entire user base.

## Quick Start

```python
from kura import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
    CheckpointManager
)
from kura.types import Conversation
from kura.summarisation import SummaryModel
from kura.cluster import ClusterModel
from kura.meta_cluster import MetaClusterModel
from kura.dimensionality import HDBUMAP
import asyncio

# Load conversations
conversations = Conversation.from_hf_dataset(
    "ivanleomk/synthetic-gemini-conversations",
    split="train"
)

# Set up models with new caching support!
summary_model = SummaryModel(
    enable_caching=True,  # NEW: 85x faster on re-runs!
    cache_dir="./.summary_cache"
)
cluster_model = ClusterModel()
meta_cluster_model = MetaClusterModel(max_clusters=10)
dimensionality_model = HDBUMAP()

# Set up checkpoint manager
checkpoint_mgr = CheckpointManager("./checkpoints", enabled=True)

# Run pipeline with explicit steps
async def process_conversations():
    # Step 1: Generate summaries
    summaries = await summarise_conversations(
        conversations,
        model=summary_model,
        checkpoint_manager=checkpoint_mgr
    )

    # Step 2: Create base clusters
    clusters = await generate_base_clusters_from_conversation_summaries(
        summaries,
        model=cluster_model,
        checkpoint_manager=checkpoint_mgr
    )

    # Step 3: Build hierarchy
    meta_clusters = await reduce_clusters_from_base_clusters(
        clusters,
        model=meta_cluster_model,
        checkpoint_manager=checkpoint_mgr
    )

    # Step 4: Project to 2D
    projected = await reduce_dimensionality_from_clusters(
        meta_clusters,
        model=dimensionality_model,
        checkpoint_manager=checkpoint_mgr
    )

    return projected

# Execute the pipeline
results = asyncio.run(process_conversations())
visualise_pipeline_results(results, style="enhanced")

# Expected output:
Programming Assistance Clusters (190 conversations)
├── Data Analysis & Visualization (38 conversations)
│   ├── "Help me create R plots for statistical analysis"
│   ├── "Debug my Tableau dashboard performance issues"
│   └── "Convert Excel formulas to pandas operations"
├── Web Development (45 conversations)
│   ├── "Fix React component re-rendering issues"
│   ├── "Integrate Stripe API with Next.js"
│   └── "Make my CSS grid responsive on mobile"
└── ... (more clusters)

Performance: 21.9s first run → 2.1s with cache (10x faster!)
```

## Documentation

**Installation**: [Get started](getting-started/installation.md) with Kura today. Install via pip or uv and configure your first analysis pipeline.

**Quickstart**: [Jump right in](getting-started/quickstart.md) with a complete example that processes conversations from raw data to insights in minutes.

**Complete Workflow**: [See how a full analysis looks](./getting-started/tutorial.md) from loading data to interpreting clusters and extracting actionable insights.

**Core Concepts**: Learn how Kura works under the hood - from [conversation loading](core-concepts/conversations.md) and [embedding](core-concepts/embedding.md) to [clustering](core-concepts/clustering.md), [summarization](core-concepts/summarization.md), [meta-clustering](core-concepts/meta-clustering.md), and [dimensionality reduction](core-concepts/dimensionality-reduction.md).

**API Reference**: [Complete documentation](api/index.md) of Kura's procedural API with examples and best practices.

## Frequently Asked Questions

1. **Can Kura work with my data and models?** Yes! Kura supports any conversation format (JSON, CSV, databases) and works with OpenAI, Anthropic, local models, or custom implementations.

2. **How much data do I need?** Start with 100+ conversations for basic patterns, 1,000+ for robust clustering, or 10,000+ for detailed insights.

3. **Is my data secure?** Absolutely. Run Kura entirely on your infrastructure, use local models for complete isolation, and analyze patterns without exposing individual conversations.

4. **What languages does Kura support?** Any language supported by your chosen model - from English to 90+ languages with models like GPT-4.

5. **Can I integrate Kura into my application?** Yes, Kura is designed as a library for seamless integration into your existing async applications.

## About

Kura is under active development. If you face any issues or have suggestions, please feel free to [open an issue](https://github.com/567-labs/kura/issues) or a PR. For more details on the technical implementation, check out this [walkthrough of the code](https://ivanleo.com/blog/understanding-user-conversations).
