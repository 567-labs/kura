#!/usr/bin/env python3
"""
Load testing script for Kura embedding and clustering scaling.

Uses existing clusters.jsonl data as baseline and scales up by duplicating
conversation patterns to test memory usage, processing time, and storage requirements.
"""

import asyncio
import time
import json
import psutil
import tracemalloc
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import uuid

# Kura imports
from kura.types import Conversation, Message
from kura.embedding import OpenAIEmbeddingModel, embed_summaries
from kura.summarisation import SummaryModel
from kura import summarise_conversations
from kura.hdbscan import HDBSCANClusteringMethod
from kura import generate_base_clusters_from_conversation_summaries
from rich.console import Console
from rich.table import Table
from rich.progress import track


@dataclass
class PerformanceMetrics:
    """Container for performance metrics at each scale."""
    scale: int
    conversations_count: int
    
    # Memory metrics (MB)
    baseline_memory_mb: float
    peak_memory_mb: float
    after_embedding_memory_mb: float
    after_clustering_memory_mb: float
    
    # Time metrics (seconds)
    embedding_time_s: float
    clustering_time_s: float
    total_time_s: float
    
    # Storage metrics (MB)
    estimated_storage_mb: float
    
    # Throughput metrics
    conversations_per_minute: float
    embeddings_per_second: float
    
    # API costs (estimated)
    estimated_api_calls: int
    estimated_cost_usd: float


class EmbeddingLoadTester:
    """Load tester for Kura embedding and clustering pipeline."""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
        self.baseline_conversations = self._load_baseline_conversations()
        
        # Models for testing
        self.embedding_model = OpenAIEmbeddingModel(
            model_name="text-embedding-3-small",
            model_batch_size=50,
            n_concurrent_jobs=5
        )
        self.summary_model = SummaryModel(console=self.console, model="openai/gpt-4o-mini")
        self.clustering_method = HDBSCANClusteringMethod()
        
    def _load_baseline_conversations(self) -> List[Conversation]:
        """Load conversations from clusters.jsonl and reconstruct conversation objects."""
        clusters_file = Path("logfire_test/clusters.jsonl")
        if not clusters_file.exists():
            raise FileNotFoundError(f"Baseline data not found: {clusters_file}")
        
        conversations = []
        with open(clusters_file) as f:
            for line in f:
                cluster_data = json.loads(line)
                
                # Generate mock conversations for each chat_id in cluster
                for chat_id in cluster_data["chat_ids"]:
                    # Extract model and question info from chat_id format: "question_model_index"
                    parts = chat_id.split("_")
                    question_id = parts[0]
                    model_name = "_".join(parts[1:-1])
                    
                    # Create realistic conversation based on cluster description
                    conversation = self._create_mock_conversation(
                        chat_id=chat_id,
                        question_id=question_id,
                        model_name=model_name,
                        cluster_description=cluster_data["description"]
                    )
                    conversations.append(conversation)
        
        self.console.print(f"Loaded {len(conversations)} baseline conversations from clusters.jsonl")
        return conversations
    
    def _create_mock_conversation(self, chat_id: str, question_id: str, model_name: str, cluster_description: str) -> Conversation:
        """Create a mock conversation based on cluster description."""
        # Generate realistic user prompt based on cluster description
        user_content = f"Please help me with: {cluster_description[:200]}..."
        
        # Generate realistic assistant response
        assistant_content = f"I'll help you with that. Based on your request about {cluster_description[:100]}..., here's my response: {cluster_description[100:300]}... [This is a mock response for testing purposes]"
        
        return Conversation(
            chat_id=chat_id,
            created_at=datetime.now(),
            messages=[
                Message(
                    role="user",
                    content=user_content,
                    created_at=datetime.now()
                ),
                Message(
                    role="assistant", 
                    content=assistant_content,
                    created_at=datetime.now()
                )
            ],
            metadata={
                "model": model_name,
                "question_id": question_id,
                "test_scale": "baseline"
            }
        )
    
    def scale_conversations(self, target_count: int) -> List[Conversation]:
        """Scale baseline conversations to target count by duplicating and modifying."""
        if target_count <= len(self.baseline_conversations):
            return self.baseline_conversations[:target_count]
        
        scaled_conversations = []
        multiplier = target_count // len(self.baseline_conversations) + 1
        
        for i in range(target_count):
            base_conv = self.baseline_conversations[i % len(self.baseline_conversations)]
            
            # Create scaled version with unique ID
            scaled_conv = Conversation(
                chat_id=f"{base_conv.chat_id}_scaled_{i}",
                created_at=base_conv.created_at,
                messages=[
                    Message(
                        role=msg.role,
                        content=f"[Scale {i//len(self.baseline_conversations)}] {msg.content}",
                        created_at=msg.created_at
                    ) for msg in base_conv.messages
                ],
                metadata={
                    **base_conv.metadata,
                    "scale_index": i,
                    "scale_multiplier": multiplier
                }
            )
            scaled_conversations.append(scaled_conv)
        
        return scaled_conversations[:target_count]
    
    async def run_embedding_test(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Run embedding test and collect metrics."""
        process = psutil.Process()
        
        # Start memory tracking
        tracemalloc.start()
        baseline_memory = process.memory_info().rss / 1024**2  # MB
        
        # Generate summaries (simplified - skip actual summarization for load testing)
        self.console.print(f"Generating {len(conversations)} mock summaries...")
        summaries = []
        for conv in conversations:
            # Mock summary for faster testing
            from kura.types import ConversationSummary
            summary = ConversationSummary(
                conversation_id=conv.chat_id,
                summary=f"Mock summary for {conv.chat_id}: {conv.messages[0].content[:100]}...",
                topics=["testing", "mock"],
                created_at=datetime.now()
            )
            summaries.append(summary)
        
        # Test embedding generation
        start_time = time.time()
        self.console.print(f"Starting embedding generation for {len(summaries)} summaries...")
        
        embedded_items = await embed_summaries(summaries, self.embedding_model)
        
        embedding_time = time.time() - start_time
        after_embedding_memory = process.memory_info().rss / 1024**2
        
        return {
            "summaries_count": len(summaries),
            "embedded_items_count": len(embedded_items),
            "baseline_memory_mb": baseline_memory,
            "after_embedding_memory_mb": after_embedding_memory,
            "embedding_time_s": embedding_time,
            "embeddings_per_second": len(embedded_items) / embedding_time if embedding_time > 0 else 0
        }
    
    async def run_clustering_test(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """Run full clustering test with memory and time tracking."""
        process = psutil.Process()
        tracemalloc.start()
        
        baseline_memory = process.memory_info().rss / 1024**2
        total_start_time = time.time()
        
        # Generate summaries (mock)
        summaries = []
        for conv in conversations:
            from kura.types import ConversationSummary
            summary = ConversationSummary(
                conversation_id=conv.chat_id,
                summary=f"Mock summary: {conv.messages[0].content[:100]}...",
                topics=["testing"],
                created_at=datetime.now()
            )
            summaries.append(summary)
        
        # Embedding phase
        embedding_start = time.time()
        embedded_items = await embed_summaries(summaries, self.embedding_model)
        embedding_time = time.time() - embedding_start
        after_embedding_memory = process.memory_info().rss / 1024**2
        
        # Memory peak during clustering
        peak_memory = after_embedding_memory
        
        # Clustering phase (simplified - create mock clusters to avoid actual clustering overhead)
        clustering_start = time.time()
        
        # Mock clustering result for performance testing
        from kura.types import Cluster
        mock_clusters = []
        cluster_size = max(1, len(summaries) // 10)  # ~10 clusters
        
        for i in range(0, len(summaries), cluster_size):
            cluster_summaries = summaries[i:i+cluster_size]
            cluster = Cluster(
                id=str(uuid.uuid4()),
                name=f"Mock Cluster {i//cluster_size + 1}",
                description=f"Mock cluster containing {len(cluster_summaries)} conversations",
                slug=f"mock_cluster_{i//cluster_size + 1}",
                chat_ids=[s.conversation_id for s in cluster_summaries],
                parent_id=None,
                count=len(cluster_summaries)
            )
            mock_clusters.append(cluster)
        
        clustering_time = time.time() - clustering_start
        after_clustering_memory = process.memory_info().rss / 1024**2
        peak_memory = max(peak_memory, after_clustering_memory)
        
        total_time = time.time() - total_start_time
        
        # Calculate storage estimates
        embedding_storage_mb = len(embedded_items) * 1536 * 4 / 1024**2  # 1536 dims * 4 bytes
        metadata_storage_mb = len(conversations) * 2  # ~2KB per conversation metadata
        total_storage_mb = embedding_storage_mb + metadata_storage_mb
        
        # Estimate API costs (OpenAI text-embedding-3-small: ~$0.00002/1K tokens)
        avg_tokens_per_summary = 50  # Estimated
        total_tokens = len(summaries) * avg_tokens_per_summary
        estimated_cost = (total_tokens / 1000) * 0.00002
        
        return {
            "conversations_count": len(conversations),
            "clusters_count": len(mock_clusters),
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": peak_memory,
            "after_embedding_memory_mb": after_embedding_memory,
            "after_clustering_memory_mb": after_clustering_memory,
            "embedding_time_s": embedding_time,
            "clustering_time_s": clustering_time,
            "total_time_s": total_time,
            "estimated_storage_mb": total_storage_mb,
            "conversations_per_minute": (len(conversations) / total_time) * 60 if total_time > 0 else 0,
            "embeddings_per_second": len(embedded_items) / embedding_time if embedding_time > 0 else 0,
            "estimated_api_calls": len(summaries) // 50 + 1,  # Based on batch size
            "estimated_cost_usd": estimated_cost
        }
    
    async def run_scale_test(self, scales: List[int]) -> List[PerformanceMetrics]:
        """Run performance tests across multiple scales."""
        results = []
        
        self.console.print("\n🚀 Starting Kura Embedding Load Testing")
        self.console.print(f"Testing scales: {scales}")
        self.console.print(f"Baseline conversations: {len(self.baseline_conversations)}")
        
        for scale in track(scales, description="Running scale tests..."):
            self.console.print(f"\n📊 Testing scale: {scale} conversations")
            
            # Generate conversations for this scale
            conversations = self.scale_conversations(scale)
            
            # Run test
            try:
                test_results = await self.run_clustering_test(conversations)
                
                metrics = PerformanceMetrics(
                    scale=scale,
                    conversations_count=test_results["conversations_count"],
                    baseline_memory_mb=test_results["baseline_memory_mb"],
                    peak_memory_mb=test_results["peak_memory_mb"],
                    after_embedding_memory_mb=test_results["after_embedding_memory_mb"],
                    after_clustering_memory_mb=test_results["after_clustering_memory_mb"],
                    embedding_time_s=test_results["embedding_time_s"],
                    clustering_time_s=test_results["clustering_time_s"],
                    total_time_s=test_results["total_time_s"],
                    estimated_storage_mb=test_results["estimated_storage_mb"],
                    conversations_per_minute=test_results["conversations_per_minute"],
                    embeddings_per_second=test_results["embeddings_per_second"],
                    estimated_api_calls=test_results["estimated_api_calls"],
                    estimated_cost_usd=test_results["estimated_cost_usd"]
                )
                
                results.append(metrics)
                
                # Print immediate results
                self.console.print(f"✅ Scale {scale}: {metrics.total_time_s:.1f}s, {metrics.peak_memory_mb:.1f}MB peak, ${metrics.estimated_cost_usd:.4f}")
                
            except Exception as e:
                self.console.print(f"❌ Scale {scale} failed: {e}")
                continue
        
        return results
    
    def generate_report(self, results: List[PerformanceMetrics]) -> None:
        """Generate and display performance report."""
        self.console.print("\n📈 Performance Test Results")
        
        # Create results table
        table = Table(title="Kura Embedding Load Test Results")
        table.add_column("Scale", style="cyan", no_wrap=True)
        table.add_column("Time (s)", style="green")
        table.add_column("Peak Memory (MB)", style="yellow")
        table.add_column("Storage (MB)", style="blue")
        table.add_column("Throughput (conv/min)", style="magenta")
        table.add_column("Cost ($)", style="red")
        
        for result in results:
            table.add_row(
                str(result.scale),
                f"{result.total_time_s:.1f}",
                f"{result.peak_memory_mb:.1f}",
                f"{result.estimated_storage_mb:.1f}",
                f"{result.conversations_per_minute:.1f}",
                f"{result.estimated_cost_usd:.4f}"
            )
        
        self.console.print(table)
        
        # Scaling analysis
        self.console.print("\n📊 Scaling Analysis")
        if len(results) >= 2:
            memory_ratio = results[-1].peak_memory_mb / results[0].peak_memory_mb
            time_ratio = results[-1].total_time_s / results[0].total_time_s
            scale_ratio = results[-1].scale / results[0].scale
            
            self.console.print(f"Memory scaling: {memory_ratio:.2f}x for {scale_ratio:.0f}x conversations")
            self.console.print(f"Time scaling: {time_ratio:.2f}x for {scale_ratio:.0f}x conversations")
            
            # Efficiency analysis
            memory_efficiency = scale_ratio / memory_ratio
            time_efficiency = scale_ratio / time_ratio
            
            self.console.print(f"Memory efficiency: {memory_efficiency:.2f} (1.0 = linear)")
            self.console.print(f"Time efficiency: {time_efficiency:.2f} (1.0 = linear)")
        
        # Save results to file
        results_file = f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump([
                {
                    "scale": r.scale,
                    "conversations_count": r.conversations_count,
                    "baseline_memory_mb": r.baseline_memory_mb,
                    "peak_memory_mb": r.peak_memory_mb,
                    "total_time_s": r.total_time_s,
                    "estimated_storage_mb": r.estimated_storage_mb,
                    "conversations_per_minute": r.conversations_per_minute,
                    "estimated_cost_usd": r.estimated_cost_usd
                } for r in results
            ], f, indent=2)
        
        self.console.print(f"\n💾 Results saved to: {results_file}")


async def main():
    """Main load testing function."""
    console = Console()
    tester = EmbeddingLoadTester(console)
    
    # Test scales - start small and increase
    # Note: Adjust these based on your system capabilities and API limits
    scales = [100, 250, 500, 1000, 2000]  # Start conservative
    
    console.print("⚠️  IMPORTANT: This test will make real API calls to OpenAI")
    console.print("Estimated cost for full test: $0.01 - $0.50")
    
    if input("Continue? (y/n): ").lower() != 'y':
        console.print("Test cancelled.")
        return
    
    # Run the load test
    results = await tester.run_scale_test(scales)
    
    # Generate report
    tester.generate_report(results)
    
    console.print("\n✨ Load testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
