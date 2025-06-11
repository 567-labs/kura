---
title: Kura v0.5.0 Released - Procedural API, Better Docs & More
date: 2025-05-29
categories:
  - Kura
  - Release
---

# Kura v0.5.0 Released

We're excited to announce the release of Kura v0.5.0! This release brings significant improvements to documentation, introduces a new procedural API for maximum flexibility, and includes numerous enhancements to make Kura even better for analyzing conversation data.

## What's New in v0.5.0

### New Procedural API (v1)

The headline feature of this release is the introduction of a functional, procedural API that gives you fine-grained control over the analysis pipeline:

```python
from kura import (
    summarise_conversations,
    generate_base_clusters_from_conversation_summaries,
    reduce_clusters_from_base_clusters,
    reduce_dimensionality_from_clusters,
)

# Run each step independently
summaries = await summarise_conversations(conversations, model=summary_model)
clusters = await generate_base_clusters_from_conversation_summaries(summaries, model=cluster_model)
meta_clusters = await reduce_clusters_from_base_clusters(clusters, model=meta_cluster_model)
projected = await reduce_dimensionality_from_clusters(meta_clusters, model=dim_reduction_model)
```

This new API offers:
- Complete control over each pipeline step
- Easy integration with heterogeneous models (OpenAI, vLLM, Hugging Face)
- Functional programming style with no hidden state
- Keyword-only arguments for clarity

<!-- more -->


### Enhanced Documentation

We've made major improvements to our documentation:

- **API Reference**: Now generated with mkdocstrings for always up-to-date documentation
- **CLAUDE.md**: Repository guidance for AI assistants working with the codebase
- **CONTRIBUTING.md**: Clear guidelines for contributors with testing and UV setup
- **Better Examples**: Added context about real datasets like the ivanleomk dataset

### Technical Improvements

#### Refactored Architecture
- Extracted visualization logic into separate modules for better maintainability
- Moved `max_clusters` parameter from Kura to MetaClusterModel where it belongs
- Implemented lazy imports for UMap to improve startup time
- Simplified embedding extensibility by replacing `embed_text()` with `__repr__()`

#### Enhanced Cluster Visualization
- Added slug field to cluster models for better identification
- Improved cluster visualization with more meaningful labels
- Better support for cluster hierarchies in the UI

#### Developer Experience
- Added Ruff workflows and pre-commit hooks for consistent code quality
- Fixed numerous type checking bugs
- Improved Summary class implementation
- Better error messages and debugging support

## Breaking Changes

While we've tried to maintain backward compatibility, please note:
- The `max_clusters` parameter has moved from the main Kura class to MetaClusterModel
- Some internal APIs have been refactored for the new procedural approach

## What's Next

We're already working on the next release with plans for:

- More embedding model integrations
- Enhanced meta-clustering algorithms
- Performance optimizations for large datasets
- Additional visualization options

## Feedback Welcome!

We'd love to hear your thoughts on this release. Please:

- Report issues on [GitHub](https://github.com/567-labs/kura/issues)
- Join the discussion in [GitHub Discussions](https://github.com/567-labs/kura/discussions)
- Share your use cases and success stories
