# API Reference

This section provides detailed API reference documentation for the Kura package, automatically generated from the source code using mkdocstrings.

## How to Use This Reference

The API reference is organized by module, with each module containing related classes and functions. For each class, you'll find:

- Constructor parameters and their descriptions
- Instance methods with parameter details and return types
- Properties and attributes

To use these classes in your code, import them from their respective modules:

```python
from kura import Kura
from kura.embedding import OpenAIEmbeddingModel
from kura.summarisation import SummaryModel
# And so on...
```

## Core Classes

## Procedural API

The procedural API provides a functional approach to conversation analysis with composable pipeline functions.

### Pipeline Functions

::: kura.summarise_conversations

::: kura.generate_base_clusters_from_conversation_summaries

::: kura.reduce_clusters_from_base_clusters

::: kura.reduce_dimensionality_from_clusters

### Checkpoint Management

::: kura.CheckpointManager

## Implementation Classes

### Embedding Models

::: kura.embedding

### Summarization

::: kura.summarisation

### Clustering

::: kura.cluster

### Meta-Clustering

::: kura.meta_cluster

### Dimensionality Reduction

::: kura.dimensionality
