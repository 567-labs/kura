#!/usr/bin/env python3
"""
Simple test to verify that embedding caching works correctly.
"""
import asyncio
from kura.embedding import OpenAIEmbeddingModel
import os


class MockCache:
    """Simple in-memory cache for testing."""
    def __init__(self):
        self.data = {}
    
    def get(self, key: str):
        return self.data.get(key)
    
    def set(self, key: str, value):
        self.data[key] = value
        print(f"Cached key: {key[:16]}... with embedding of length {len(value)}")


async def test_caching():
    """Test that caching works with OpenAI embedding model."""
    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY found, skipping test")
        return
    
    cache = MockCache()
    
    # Create embedding model with cache
    model = OpenAIEmbeddingModel(
        model_name="text-embedding-3-small",
        cache=cache
    )
    
    texts = ["Hello world", "This is a test", "Hello world"]  # Note duplicate
    
    print("First embedding (should cache 'Hello world' and 'This is a test'):")
    embeddings1 = await model.embed(texts)
    print(f"Got {len(embeddings1)} embeddings")
    
    print("\nSecond embedding (should use cache for 'Hello world'):")
    embeddings2 = await model.embed(["Hello world", "Another text"])
    print(f"Got {len(embeddings2)} embeddings")
    
    print(f"\nCache now contains {len(cache.data)} items")
    
    # Verify the embeddings are the same for the duplicate text
    assert embeddings1[0] == embeddings1[2], "Duplicate texts should have same embedding"
    assert embeddings1[0] == embeddings2[0], "Cached embedding should match original"
    
    print("✅ Caching test passed!")


if __name__ == "__main__":
    asyncio.run(test_caching())
