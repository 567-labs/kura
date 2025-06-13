from kura.checkpoints import JSONLCheckpointManager
from kura.types import ProjectedCluster, ConversationSummary, Conversation
from collections import defaultdict
import pandas as pd

checkpoint_mgr = JSONLCheckpointManager("./checkpoints_mt_bench_10k_hdbscan")

clusters = checkpoint_mgr.load_checkpoint("dimensionality", ProjectedCluster)
summaries = checkpoint_mgr.load_checkpoint("summaries", ConversationSummary)
conversations = checkpoint_mgr.load_checkpoint("conversations", Conversation)
summaries_by_chat_id = {summary.chat_id: summary for summary in summaries}
conversations_by_chat_id = {
    conversation.chat_id: conversation for conversation in conversations
}

level_to_clusters = defaultdict(list)
for cluster in clusters:
    level_to_clusters[cluster.level].append(cluster)

for level, clusters in level_to_clusters.items():
    print(f"Level {level}: {len(clusters)} clusters")


# Look at Level 0
level_0_clusters = level_to_clusters[0]

# Analyze each cluster with pandas
for cluster_idx, cluster in enumerate(level_0_clusters):
    print(f"\n--- Cluster {cluster_idx} ---")

    # Create DataFrame from chat_ids
    chat_data = []
    for chat_id in cluster.chat_ids:
        summary_data = summaries_by_chat_id[chat_id]
        parts = chat_id.split("_")
        chat_data.append(
            {
                "question_id": int(parts[0]),
                "question_text": conversations_by_chat_id[chat_id].messages[0].content,
                "conversation_id": parts[-1],
                "chat_id": chat_id,
                "winner": summary_data.metadata.get("winner", False),
                "model": summary_data.metadata.get("model", "unknown"),
            }
        )

    df = pd.DataFrame(chat_data)

    # Find valid conversations (appearing exactly twice)
    conv_counts = df["conversation_id"].value_counts()
    valid_conversations = conv_counts[conv_counts == 2].index
    print(
        f"Valid conversations (appearing twice): {len(valid_conversations)}/ len(cluster.chat_ids): {len(cluster.chat_ids)}"
    )

    # Filter to valid conversations and group by question
    valid_df = df[df["conversation_id"].isin(valid_conversations)]
    question_summary = (
        valid_df.groupby("question_id")["conversation_id"].nunique().sort_index()
    )

    # Group by model and calculate win rates
    win_rates = (
        valid_df.groupby(["question_id", "model", "winner"])
        .size()
        .reset_index(name="count")
    )

    # Calculate win rates per question per model
    model_stats = (
        valid_df.groupby(["question_id", "model"])
        .agg({"winner": ["count", "sum"], "conversation_id": "nunique"})
        .reset_index()
    )

    # Flatten column names
    model_stats.columns = [
        "question_id",
        "model",
        "total_games",
        "wins",
        "unique_conversations",
    ]
    model_stats["win_rate"] = model_stats["wins"] / model_stats["total_games"]

    print(f"\n=== Win Rates by Question and Model (Cluster {cluster_idx}) ===")
    if hasattr(cluster, "name") and cluster.name:
        print(f"Cluster Name: {cluster.name}")
    if hasattr(cluster, "description") and cluster.description:
        print(f"Cluster Description: {cluster.description}")

    for question_id in sorted(model_stats["question_id"].unique()):
        question_data = model_stats[model_stats["question_id"] == question_id].copy()
        question_data["losses"] = question_data["total_games"] - question_data["wins"]

        # Sort by wins descending
        question_data = question_data.sort_values("wins", ascending=False)

        # Get question text from summaries
        question_chats = valid_df[valid_df["question_id"] == question_id][
            "chat_id"
        ].iloc[0]
        question_summary = summaries_by_chat_id[question_chats]

        question_text = conversations_by_chat_id[question_chats].messages[0].content

        print(f"\nQuestion {question_id}:")
        print(f"Text: {question_text}")
        print(f"{'Model':<25} | {'Wins':<6} | {'Losses':<8} | {'Win Rate':<8}")
        print("-" * 55)

        for _, row in question_data.iterrows():
            print(
                f"{row['model']:<25} | {row['wins']:<6} | {row['losses']:<8} | {row['win_rate']:.1%}"
            )

    # Overall summary across all questions
    overall_stats = (
        valid_df.groupby("model")
        .agg(
            {
                "winner": ["count", "sum"],
            }
        )
        .reset_index()
    )

    overall_stats.columns = ["model", "total_games", "wins"]
    overall_stats["losses"] = overall_stats["total_games"] - overall_stats["wins"]
    overall_stats["win_rate"] = overall_stats["wins"] / overall_stats["total_games"]
    overall_stats = overall_stats.sort_values("wins", ascending=False)
