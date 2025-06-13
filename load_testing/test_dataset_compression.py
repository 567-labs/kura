from kura.checkpoints import (
    HFDatasetCheckpointManager,
    ParquetCheckpointManager,
    JSONLCheckpointManager,
)
import os
import shutil
import pandas as pd
from kura.types.summarisation import ConversationSummary


def test_dataset_compression(summaries: list[ConversationSummary]):
    checkpoint_dir = f"/Users/ivanleo/Documents/coding/kura/data/test/{len(summaries)}"

    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    hf_checkpoint_manager = HFDatasetCheckpointManager(
        checkpoint_dir=checkpoint_dir + "/hf",
    )
    parquet_checkpoint_manager = ParquetCheckpointManager(
        checkpoint_dir=checkpoint_dir + "/parquet",
    )
    jsonl_checkpoint_manager = JSONLCheckpointManager(
        checkpoint_dir=checkpoint_dir + "/jsonl",
    )

    hf_checkpoint_manager.save_checkpoint("hf_summaries", summaries)
    parquet_checkpoint_manager.save_checkpoint("parquet_summaries", summaries)
    jsonl_checkpoint_manager.save_checkpoint("jsonl_summaries", summaries)

    # Calculate size of HF dataset directory
    hf_size = sum(
        os.path.getsize(os.path.join(checkpoint_dir + "/hf/hf_summaries", f))
        for f in os.listdir(checkpoint_dir + "/hf/hf_summaries")
    )

    # Calculate size of Parquet file
    parquet_size = os.path.getsize(
        checkpoint_dir + "/parquet/parquet_summaries.parquet"
    )

    # Calculate size of JSONL file
    jsonl_size = os.path.getsize(checkpoint_dir + "/jsonl/jsonl_summaries.jsonl")

    # Load back the checkpoints and verify data integrity
    loaded_hf = hf_checkpoint_manager.load_checkpoint(
        "hf_summaries", ConversationSummary
    )
    loaded_parquet = parquet_checkpoint_manager.load_checkpoint(
        "parquet_summaries", ConversationSummary
    )
    loaded_jsonl = jsonl_checkpoint_manager.load_checkpoint(
        "jsonl_summaries", ConversationSummary
    )

    # Verify lengths match
    assert len(loaded_hf) == len(summaries), (
        f"HF dataset length mismatch: {len(loaded_hf)} != {len(summaries)}"
    )
    assert len(loaded_parquet) == len(summaries), (
        f"Parquet length mismatch: {len(loaded_parquet)} != {len(summaries)}"
    )
    assert len(loaded_jsonl) == len(summaries), (
        f"JSONL length mismatch: {len(loaded_jsonl)} != {len(summaries)}"
    )

    # Verify content matches by comparing chat_ids
    original_chat_ids = {s.chat_id for s in summaries}
    hf_chat_ids = {s.chat_id for s in loaded_hf}
    parquet_chat_ids = {s.chat_id for s in loaded_parquet}
    jsonl_chat_ids = {s.chat_id for s in loaded_jsonl}

    assert hf_chat_ids == original_chat_ids, "HF dataset content mismatch"
    assert parquet_chat_ids == original_chat_ids, "Parquet content mismatch"
    assert jsonl_chat_ids == original_chat_ids, "JSONL content mismatch"

    print(f"HuggingFace dataset size: {hf_size / (1024 * 1024):.2f} MB")
    print(f"Parquet file size: {parquet_size / (1024 * 1024):.2f} MB")
    print(f"JSONL file size: {jsonl_size / (1024 * 1024):.2f} MB")

    return {
        "size": len(summaries),
        "hf_size": hf_size,
        "parquet_size": parquet_size,
        "jsonl_size": jsonl_size,
    }


checkpoint_manager = JSONLCheckpointManager(
    checkpoint_dir="/Users/ivanleo/Documents/coding/kura/data/logfire_test_1",
)
summaries = checkpoint_manager.load_checkpoint("summaries", ConversationSummary)

test_sizes = [100, 1000, 5000, 10000, 100000]

results = []

for size in test_sizes:
    testing_summaries = []
    while len(testing_summaries) < size and summaries:
        testing_summaries.extend(
            summaries[: min(size - len(testing_summaries), len(summaries))]
        )
    results.append(test_dataset_compression(testing_summaries))

df = pd.DataFrame(results)
df_new = pd.DataFrame()
df_new["size"] = df["size"]
df_new = df_new.join(
    df[["hf_size", "parquet_size", "jsonl_size"]].map(
        lambda x: f"{x / (1024 * 1024):.2f} MB"
    )
)

print(df_new)
