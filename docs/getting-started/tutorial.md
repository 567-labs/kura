# Tutorial: Analyzing Chat Data with Kura

!!! info "What You'll Build"

    By the end of this tutorial, you'll have built a complete analysis pipeline that processes 560 real user queries, identifies the top 3 query categories that represent 67% of all requests, and creates production-ready classifiers for automatic query categorization.

This tutorial showcases an entire end to end flow of how to use Kura for your specific application.

## Prerequisites

- Install `Kura` in a virtual environment with `uv pip install kura`
- Set your `OPENAI_API_KEY` environment variable to use OpenAI's GPT-4o-mini model
- Download the tutorial dataset

[**Download Dataset**](../assets/conversations.json){ .md-button .md-button--primary }

## Tutorial Series

### Step 1. Cluster Conversations

Discover user query patterns through topic modeling and clustering. You'll learn to identify that three major topics account for 67% of queries, with artifact management appearing in 61% of conversations.

[**Start Clustering Tutorial**](../notebooks/how-to-look-at-data/01_clustering_task.ipynb){ .md-button}

### Step 2. Better Summaries

Transform generic summaries into domain-specific insights. Build custom summarization models that turn seven vague clusters into three actionable categories: Access Controls, Deployment, and Experiment Management.

[**Start Summaries Tutorial**](../notebooks/how-to-look-at-data/02_summaries_task.ipynb){ .md-button }

### Step 3. Building Classifiers

Convert clustering insights into production classifiers. Build real-time systems that automatically categorize new queries and scale your insights.

[**Start Classifiers Tutorial**](../notebooks/how-to-look-at-data/03_classifiers_task.ipynb){ .md-button }
