#!/usr/bin/env python3
"""
Simple load testing script for Kura conversation summarisation.
"""

import os
import asyncio
import json
import time
import logging
import csv
from contextlib import contextmanager
from typing import Dict

from kura.types import Conversation
from kura import summarise_conversations
from kura.summarisation import SummaryModel
from kura.checkpoints import JSONLCheckpointManager

# Optional logfire import for tracing (one-off script dependency)
import logfire
import dotenv

dotenv.load_dotenv()


class TimingManager:
    """Manager for collecting detailed timing information."""

    def __init__(self):
        self.timings: Dict[str, float] = {}

    @contextmanager
    def timer(self, step_name: str):
        """Context manager to time a step."""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.timings[step_name] = duration
            logger.info(f"{step_name} took {duration:.2f} seconds")

    def get_timings(self) -> Dict[str, float]:
        """Get all recorded timings."""
        return self.timings.copy()

    def reset(self):
        """Reset all timings."""
        self.timings.clear()


# Configure logfire if available

logfire.configure(
    send_to_logfire=True,
    token=os.getenv("LOGFIRE_WRITE_TOKEN"),
    console=False,
)
logfire.instrument_openai()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def load_test_summarisation(conversation_count: int = 100, **kwargs) -> dict:
    """
    Load test the summarisation pipeline with detailed timing.

    Args:
        conversation_count: Number of conversations to test with
        **kwargs: Additional configuration options
    """
    timing_manager = TimingManager()

    logger.info(
        f"Starting summarisation load test with {conversation_count} conversations"
    )

    # Load conversations from the same dataset as main.py
    all_conversations = Conversation.from_conversation_dump(
        "mt_bench_human_judgments.json"
    )

    if not all_conversations:
        logger.error("No conversations found in dataset")
        return {"error": "No conversations found"}

    # Take subset for testing
    test_conversations = all_conversations[:conversation_count]
    logger.info(f"Using {len(test_conversations)} conversations for testing")

    if not kwargs.get("cache_dir"):
        raise ValueError("cache_dir must be specified")

    # Configure models similar to main.py
    summary_model = SummaryModel(cache_dir=kwargs.get("cache_dir"))

    checkpoint_manager = JSONLCheckpointManager(
        f"./data/benchmarks_{conversation_count}"
    )

    # Run summarisation with timing
    try:
        with timing_manager.timer("summarise_conversations"):
            with logfire.span("summarise_conversations"):
                summaries = await summarise_conversations(
                    test_conversations,
                    model=summary_model,
                    checkpoint_manager=checkpoint_manager,
                )

        # Calculate results
        timings = timing_manager.get_timings()

        results = {
            "conversation_count": len(test_conversations),
            "summary_count": len(summaries),
            "success": True,
            "cache_enabled": kwargs.get("cache_dir") is not None,
            "checkpointing_enabled": kwargs.get("checkpointing_enabled", False),
            # Add detailed timing information
            "total_time": sum(timings.values()),
            "load_conversations_time": timings.get("load_conversations", 0),
            "configure_models_time": timings.get("configure_models", 0),
            "summarise_conversations_time": timings.get("summarise_conversations", 0),
            # Calculate throughput
            "conversations_per_second": len(test_conversations)
            / timings.get("summarise_conversations", 1),
            "seconds_per_conversation": timings.get("summarise_conversations", 0)
            / len(test_conversations),
        }

    except Exception as e:
        timings = timing_manager.get_timings()
        results = {
            "conversation_count": len(test_conversations)
            if "test_conversations" in locals()
            else conversation_count,
            "error": str(e),
            "success": False,
            "total_time": sum(timings.values()),
            "load_conversations_time": timings.get("load_conversations", 0),
            "configure_models_time": timings.get("configure_models", 0),
            "summarise_conversations_time": timings.get("summarise_conversations", 0),
        }

    return results


def save_results_to_csv(results: list[dict], timestamp: str):
    """Save timing results to a CSV file."""
    csv_filename = f"summarisation_load_test_results_{timestamp}.csv"

    if not results:
        logger.warning("No results to save to CSV")
        return

    # Get all keys from the first result for CSV headers
    fieldnames = list(results[0].keys())

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Detailed timing results saved to {csv_filename}")


async def main():
    """Run load tests with different configurations."""
    test_configs = [
        {
            "conversation_count": 100,
            "cache_dir": "./.summary_cache_100",
        },
        {
            "conversation_count": 1000,
            "cache_dir": "./.summary_cache_1000",
        },
        {
            "conversation_count": 6000,
            "cache_dir": "./.summary_cache_6000",
        },
    ]

    all_results = []

    for config in test_configs:
        with logfire.span(f"test_summarisation_{config['conversation_count']}"):
            logger.info(f"Running test: {config}")
            result = await load_test_summarisation(**config)
            all_results.append(result)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save JSON results (original format)
    json_output_file = f"summarisation_load_test_results_{timestamp}.json"
    with open(json_output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {json_output_file}")

    # Save CSV results (detailed timing)
    save_results_to_csv(all_results, timestamp)

    # Print summary
    print("\n=== SUMMARISATION BENCHMARK RESULTS ===")
    for result in all_results:
        if result.get("success"):
            print(f"Conversations: {result['conversation_count']}")
            print(f"Total time: {result['total_time']:.2f}s")
            print(f"Summarisation time: {result['summarise_conversations_time']:.2f}s")
            print(
                f"Throughput: {result['conversations_per_second']:.2f} conversations/sec"
            )
            print(
                f"Per conversation: {result['seconds_per_conversation']:.3f}s/conversation"
            )
            print("---")
        else:
            print(
                f"FAILED - Conversations: {result['conversation_count']}, Error: {result.get('error')}"
            )
            print("---")


if __name__ == "__main__":
    asyncio.run(main())
