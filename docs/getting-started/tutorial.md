# Tutorial: Analyzing Chat Data with Kura

Learn how to analyze RAG system chat data through a three-part tutorial series. Work with 560 real user queries to discover patterns and build production-ready classifiers.

## Prerequisites

- Install `Kura` in a virtual environment with `uv pip install kura`
- Set `OPENAI_API_KEY` to use OpenAI's GPT-4o-mini model
- Download the tutorial dataset

[**Download Dataset**](../assets/conversations.json){ .md-button .md-button--primary }

## Tutorial Series

### Step 1. Cluster Conversations

Discover user query patterns through topic modeling and clustering. Learn to identify that three major topics account for 67% of queries, with artifact management appearing in 61% of conversations.

[**Start Clustering Tutorial**](../notebooks/how-to-look-at-data/01_clustering_task.ipynb){ .md-button }

### Step 2. Better Summaries

Transform generic summaries into domain-specific insights. Build custom summarization models that turn seven vague clusters into three actionable categories: Access Controls, Deployment, and Experiment Management.

[**Start Summaries Tutorial**](../notebooks/how-to-look-at-data/02_summaries_task.ipynb){ .md-button }

### Step 3. Building Classifiers

Convert clustering insights into production classifiers. Build real-time systems that automatically categorize new queries and scale your insights.

[**Start Classifiers Tutorial**](../notebooks/how-to-look-at-data/03_classifiers_task.ipynb){ .md-button }

## What You'll Learn

- Systematically analyze large volumes of user queries
- Build custom models for your specific domain
- Create production systems for automatic query classification
- Make data-driven decisions about system improvements

Ready to start? Begin with [Step 1: Cluster Conversations](../notebooks/how-to-look-at-data/01_clustering_task.ipynb) below.
