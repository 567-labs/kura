# Tutorials

Welcome to the Kura tutorials! These interactive Jupyter notebooks will guide you through practical, hands-on examples of analyzing and understanding conversational data using machine learning techniques.

## What You'll Learn

These tutorials are designed to take you from basic conversation clustering to advanced analysis techniques. Each notebook builds upon the previous one, creating a comprehensive learning path for understanding how to extract insights from conversational data.

### Tutorial Overview

#### 1. [Cluster Conversations](1-cluster-conversations.ipynb)
**Goal**: Learn the fundamentals of conversation clustering

In this introductory notebook, you'll discover how to:
- Load and preprocess conversational data from various sources
- Generate meaningful embeddings from conversations
- Apply clustering algorithms to group similar conversations
- Visualize cluster results to identify patterns
- Understand the basic building blocks of conversation analysis

**Practical Application**: This technique helps you automatically organize thousands of customer support tickets, user feedback, or chat logs into meaningful categories without manual labeling.

#### 2. [Better Summaries](2-better-summaries.ipynb)
**Goal**: Improve cluster interpretation through intelligent summarization

Building on the clustering foundation, this notebook teaches you to:
- Generate concise, informative summaries of conversation clusters
- Use LLMs to extract key themes and topics from grouped conversations
- Create hierarchical summaries for multi-level cluster analysis
- Optimize prompts for better summary quality
- Handle edge cases and outliers in your data

**Practical Application**: Transform raw clusters into actionable insights by automatically generating reports that highlight what users are talking about, common pain points, and emerging trends.

#### 3. [Classifiers](3-classifiers.ipynb)
**Goal**: Build custom classifiers for specific conversation attributes

The advanced notebook shows you how to:
- Create custom metadata extractors using LLMs
- Build classifiers for sentiment, intent, and other conversation properties
- Combine multiple classifiers for rich conversation analysis
- Evaluate classifier performance and iterate on improvements
- Integrate classification results with clustering for deeper insights

**Practical Application**: Automatically tag conversations with custom labels like priority level, department routing, sentiment scores, or any domain-specific categories your organization needs.

## Why These Skills Matter

Understanding conversational data at scale is crucial for:

- **Product Teams**: Discover what features users request most frequently
- **Customer Success**: Identify common issues and pain points automatically
- **Research Teams**: Analyze interview transcripts and user studies efficiently
- **Community Managers**: Track trending topics and community sentiment
- **Data Scientists**: Build foundational skills for conversational AI applications

## Prerequisites

To get the most out of these tutorials, you should have:

- Basic Python programming knowledge
- Familiarity with Jupyter notebooks
- Understanding of basic machine learning concepts (clustering, embeddings)
- Access to an OpenAI API key or similar LLM service

## Getting Started

1. Ensure you have Kura installed:
   ```bash
   pip install kura
   ```

2. Set up your API keys for the LLM provider you'll be using

3. Open the first notebook and follow along with the examples

4. Experiment with your own data - the best way to learn is by applying these techniques to real conversations!

## Tips for Success

- **Start Small**: Begin with a modest dataset (100-1000 conversations) to understand the concepts before scaling up
- **Iterate Often**: Don't expect perfect results on the first try - refine your approach based on what you learn
- **Visualize Everything**: Use the built-in visualization tools to understand your data better
- **Document Your Process**: Keep notes on what works for your specific use case

## Next Steps

After completing these tutorials, you'll be ready to:

- Build production-ready conversation analysis pipelines
- Create custom analysis tools for your specific domain
- Contribute to the Kura project with your own improvements
- Apply these techniques to new types of conversational data

Ready to begin? Start with [Cluster Conversations](1-cluster-conversations.ipynb) and embark on your journey to mastering conversational data analysis!
