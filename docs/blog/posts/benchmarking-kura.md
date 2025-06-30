---
draft: false
date: 2025-06-16
categories:
  - Benchmarks
---

# Benchmarking Kura

Kura is an open-source topic modeling library that automatically discovers and summarizes high-level themes from large collections of conversational data. By combining AI-powered summarization with advanced clustering techniques, Kura transforms thousands of raw conversations into actionable insights about what your users are actually discussing.

We benchmarked Kura across three critical dimensions: processing performance, storage efficiency, and clustering quality. Our results show that Kura delivers production-ready performance with:

1. **Fast, predictable processing**: 6,000 conversations analyzed with GPT-4o-mini in under 7 minutes and just around $2 in token costs (using 20 concurrent tasks)
2. **Storage is not an issue**: 440x compression ratios mean even 100,000 conversations require only 20MB of storage - storage overhead is negligible for production workloads
3. **Accurate topic discovery**: Over 85% cluster alignment when validated against similar conversation topics

In this article, we'll walk through our benchmark methodology, detailed findings and how you can apply these results to your own use cases.

## Dataset Used

For this benchmark, we used the [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) dataset from Hugging Face.

This dataset contains 3,000+ rows of human preferences between two model responses to identical questions, with each question sampled multiple times across different model pairs.

We generated two conversations per row, creating a 6,000+ conversation evaluation dataset that tests clustering quality with identical inputs and varying responses.

!!! note

    If you're interested in the full dataset, we've uploaded the processed dataset we used [here](https://huggingface.co/datasets/567-labs/kura-benchmark-dataset) to hugging face. We also have the full benchmarking scripts and datasets generated at [here](https://github.com/567-labs/kura/tree/main/benchmarks).

<!-- more -->

## Benchmarks

We focused on three critical factors for production usage:

1. **Summarization cost and latency** - This represents the biggest barrier to scaling since Kura uses AI models to extract core intent from each conversation before clustering, with this step accounting for roughly 80% of all processing work in the pipeline.

2. **Storage requirements** - This matters because Kura's checkpoint system caches results at each stage to enable resuming interrupted jobs, potentially creating significant storage overhead at scale.

3. **Cluster alignment** - We benchmarked this by using responses generated from identical questions, creating a straightforward test of whether similar conversations cluster together correctly.

We then used GPT-4o-mini and GPT-4.1-mini in two separate runs, seeing how the two different models might differ in terms of their performance and cost. We used a conservative semaphore limit of 20 concurrent tasks to stay within rate limits for most users - this can be increased for better performance if your API limits allow.

Even at 10,000 conversations, processing would cost approximately $10 and complete in less than 10 minutes, demonstrating Kura's scalability for large-scale production workloads.

Let's walk through our results below.

### Summarization cost and latency

Understanding where computational resources are allocated is crucial for optimizing Kura's pipeline. The data reveals that summarization dominates both time and cost, accounting for roughly 80-90% of total processing across all dataset sizes.

**Cost Breakdown ($)**

| Model            | Dataset Size | Input Tokens | Output Tokens | Summarisation | Other Steps | **Total** |
| ---------------- | ------------ | ------------ | ------------- | ------------- | ----------- | --------- |
| **GPT-4o-mini**  | 500          | 925,573      | 53,073        | 0.14          | 0.03        | **0.17**  |
|                  | 1000         | 1,748,032    | 104,218       | 0.27          | 0.05        | **0.32**  |
|                  | 6000         | 11,084,725   | 655,724       | 1.69          | 0.36        | **2.05**  |
| **GPT-4.1-mini** | 500          | 959,792      | 69,856        | 0.50          | 0.00        | **0.50**  |
|                  | 1000         | 1,809,783    | 132,094       | 0.75          | 0.18        | **0.93**  |
|                  | 6000         | 11,458,713   | 806,215       | 4.73          | 1.21        | **5.94**  |

**Time Breakdown (seconds)**

| Model            | Dataset Size | Concurrent Tasks | Summarisation | Other Steps | **Total**  |
| ---------------- | ------------ | ---------------- | ------------- | ----------- | ---------- |
| **GPT-4o-mini**  | 500          | 20               | 26.90         | 11.41       | **38.50**  |
|                  | 1000         | 20               | 52.50         | 12.57       | **65.10**  |
|                  | 6000         | 20               | 320.00        | 94.89       | **414.90** |
| **GPT-4.1-mini** | 500          | 20               | 33.20         | 4.99        | **38.20**  |
|                  | 1000         | 20               | 52.50         | 14.62       | **67.10**  |
|                  | 6000         | 20               | 346.00        | 120.02      | **466.00** |

We can see that summarisation is the primary bottleneck in the pipeline. It takes up roughly 70-90% of the time and budget that we spend processing the dataset. Even so, it's not prohibitively expensive with 6000 conversations costing us just under $2.05 with GPT-4o-mini.

We mostly also expect time required and cost to scale linearly with 100,000 conversations costing approximately $34 (GPT-4o-mini) or $98 (GPT-4.1-mini) and finishing in under 2.5 hours.

Scaling to even larger workloads with a vLLM server is also an option, allowing you to further optimise for cost and speed.

### Storage Requirements

Storage overhead is not a concern for production workloads. While Kura caches intermediate results at multiple stages to enable resuming interrupted jobs, our compression techniques ensure storage requirements remain minimal even for large conversation datasets.

Kura currently supports three forms of checkpoints - Parquet, Jsonl and Hugging Face's arrow dataset. We expect Parquet to dramatically outperform both JSONL and HuggingFace formats due to its binary encoding and columnar storage design. Ultimately we found that Parquet compression achieves 98-99% size reduction compared to HuggingFace datasets and JSONL formats.

Here are some of the results that we found below

| Dataset Size | HuggingFace | Parquet | JSONL    |
| ------------ | ----------- | ------- | -------- |
| 100          | 0.09 MB     | 0.03 MB | 0.10 MB  |
| 1,000        | 0.88 MB     | 0.15 MB | 0.99 MB  |
| 10,000       | 8.81 MB     | 0.15 MB | 9.86 MB  |
| 100,000      | 88.08 MB    | 0.20 MB | 98.61 MB |

Beyond checkpoint storage, Kura also supports resumable jobs and individual summary caching. This can be enabled with a `cache_dir` parameter. This can be used to resume interrupted jobs and to cache individual summaries for faster re-runs. The space requirements scale roughly linearly but even with 100k conversations cached, it's less than 50MB items

| Conversations | Cache Size |
| ------------- | ---------- |
| 100           | 48 KB      |
| 1,000         | 112 KB     |
| 6,000         | 556 KB     |
| 100,000\*     | 10MB\*     |

\*Extrapolated from observed scaling

With this automatic caching, we can resume interrupted jobs immediately without any additional effort as well as save on costs when we need to run analysis on the same conversations.

This has minimal storage requirements, with 100,000 conversations requiring less than 15MB of storage - storage is not a bottleneck for production use.

More importantly, as we cluster and summarize increasingly larger datasets, our compression techniques ensure storage never becomes a concern. Parquet enables us to store the same data in just a few megabytes while individual summary caching adds virtually no storage burden while providing massive performance improvements.

### Cluster Alignment

Since the MT-Bench dataset contains identical prompts answered by different AI models, all responses to the same question should be in the same clusters regardless of the model. This is because the underlying user intent and topic remains consistent.

In order to measure this, we defined a new metric called `spillover-rate`. This is the number of questions among our list of questions that have responses which appear in more than 1 cluster at each level.

This provides a direct test of whether Kura correctly groups similar conversations together

**Examples of Legitimate Spillovers:**

Even with low spillover rates, some questions legitimately belong in multiple categories:

- **Quantum Physics Question**: _"What is superposition and how does it relate to quantum entanglement?"_ appeared in both "Clarify scientific concepts" (65 occurrences) and "Provide accurate solutions" (7 occurrences)

- **Automotive Terminology**: _"Which word doesn't belong: tyre, steering wheel, car, engine"_ spanned "Help me correct automotive terminology errors" (62 occurrences) and "Enhance writing and communication strategies" (21 occurrences)

Kura generated clusters with an average spillover rate of 10-15% which means that roughly 85-90% of responses to the same question are clustered together in the same cluster. This is significantly lower than pure random clustering which would otherwise produce a ~100% spillover rate for all clusters.

Since we have approximately 6,000 conversations we're trying to cluster derived from 80 unique questions, each question appears roughly 75 times across different model pairs. This is a strong indicator of our cluster consistency.

## Model Performance Comparison

While GPT-4.1-mini costs approximately 3x more than GPT-4o-mini, our analysis revealed an interesting finding: both models perform similarly on cluster alignment.

At the top level clusters, GPT-4.1-mini edged out GPT-4o-mini by a small margin, having an average spillover rate of 9.6% relative to GPT-4o-mini's 12.3% when clustering 6,000 conversations.

Rather than agonizing over which “mini” to pick upfront, the most effective approach is to treat clustering as an iterative design exercise: run a batch, inspect the clusters and labels, tweak your summarization prompt or clustering hyperparameters, and run again.

Over a few cycles you’ll quickly learn which prompt formulations yield the cleanest topic separation for your particular dataset.

Because GPT-4o-mini comes in at roughly one-third the cost of GPT-4.1-mini yet delivers nearly identical cluster alignment, it’s a great choice for exploratory experimentation and large-scale runs where budget matters most.

Once you’ve locked in on prompt patterns and parameters that give you coherent clusters, you can optionally switch to GPT-4.1-mini if you need more detailed, nuanced summaries for stakeholder-facing reports. In short: experiment early and often with prompts, then choose the model that best balances your needs for cost, speed, and summary quality.

## Conclusion

Kura scales predictably for production workloads - Processing 6,000 conversations costs under $2 and completes in under 8 minutes. If we were to extrapolate this out, this means that Kura can process 100,000 conversations at approximately $34 dollars under 2.5 hours.

Kura provides extensive customization options for your specific use case. You can experiment with clustering hyperparameters, swap models for different cost-quality tradeoffs, and customize prompts for summaries and cluster generation to match your domain requirements.

For teams drowning in user feedback, support tickets, or community discussions, Kura offers a practical path forward. Instead of manually reading through thousands of conversations or relying on simple keyword searches, you can automatically discover what your users are actually talking about - and do it in minutes, not days.

Ready to uncover the hidden patterns in your conversational data? [Get started with Kura today](https://usekura.xyz) and transform your mountain of conversations into actionable insights.
