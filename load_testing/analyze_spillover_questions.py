#!/usr/bin/env python3
"""
Analyze question spillover across cluster levels.

This script reads cluster data from a checkpoint directory and identifies
questions that appear in multiple clusters at each level, analyzing spillover rates.
"""

from collections import defaultdict
from kura.checkpoints import JSONLCheckpointManager
from kura.types import ProjectedCluster, Conversation

checkpoint_mgr = JSONLCheckpointManager("./checkpoints_mt_bench_10k_hdbscan")
projected_clusters = checkpoint_mgr.load_checkpoint("dimensionality", ProjectedCluster)
conversations = checkpoint_mgr.load_checkpoint("conversations", Conversation)

question_id_to_query: dict[int, str] = {}

for conversation in conversations:
    query = conversation.messages[0].content
    question_id = conversation.metadata["question_id"]
    question_id_to_query[int(question_id)] = query

# Now we group by level
levels: dict[int, list[ProjectedCluster]] = defaultdict(list)

for cluster in projected_clusters:
    levels[cluster.level].append(cluster)


for level in [0, 1, 2]:
    # Now we get the unique questions at this level
    question_ids = set()

    for cluster in levels[level]:
        for chat_id in cluster.chat_ids:
            question_id = int(chat_id.split("_")[0])
            question_ids.add(question_id)

    spillovers = 0
    # Now we analyse spillover rates
    for question in question_ids:
        # Now we need to find the clusters that contain this question
        relevant_clusters = []

        for cluster in levels[level]:
            cluster_question_ids = [
                int(chat_id.split("_")[0]) for chat_id in cluster.chat_ids
            ]

            if question in cluster_question_ids:
                relevant_clusters.append(cluster)

        if len(relevant_clusters) > 1:
            spillovers += 1

            # Let's calculate the spillovers between each cluster for this question
            print(f"\nQuestion {question} spillover analysis:")
            print(f"Query: {question_id_to_query[question]}")
            print(f"Appears in {len(relevant_clusters)} clusters:")

            for cluster in relevant_clusters:
                # Count occurrences of this question_id in cluster
                question_count = sum(
                    1
                    for chat_id in cluster.chat_ids
                    if int(chat_id.split("_")[0]) == question
                )

                print(
                    f"  Cluster: {cluster.name if hasattr(cluster, 'name') else 'Unnamed'}"
                )
                print(f"    Occurrences: {question_count}")

        if len(relevant_clusters) == 0:
            print(f"Unable to find question in any cluster at level {level}")

    spillover_rate = spillovers / len(question_ids)
    print(
        f"Spillover rate at level {level}: {spillover_rate} [n={len(question_ids)}, c={len(levels[level])}]"
    )
    print("=" * 100)
