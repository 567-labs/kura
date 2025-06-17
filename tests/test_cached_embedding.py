"""Tests for CachedEmbeddingModel."""

import pytest
import tempfile
import shutil
from pathlib import Path

from kura.cached_embedding import CachedEmbeddingModel
from kura.base_classes import BaseEmbeddingModel, BaseCheckpointManager


class MockEmbeddingModel(BaseEmbeddingModel):
    """Mock embedding model for testing."""
    
    def __init__(self, model_name: str = "test-model"):
        self.model_name = model_name
        self.embed_call_count = 0
        
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Mock embed that returns simple embeddings."""
        self.embed_call_count += 1
        return [[float(i) for i in range(len(text))] for text in texts]
    
    def slug(self) -> str:
        return f"mock:{self.model_name}"


class MockCheckpointManager(BaseCheckpointManager):
    """Mock checkpoint manager for testing."""
    
    def __init__(self, checkpoint_dir: str, *, enabled: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enabled = enabled
        self.storage = {}
        
    def setup_checkpoint_dir(self) -> None:
        """Setup is handled by base test setup."""
        pass
        
    def load_checkpoint(self, filename: str, model_class: type, **kwargs):
        """Load from in-memory storage."""
        return self.storage.get(filename, None)
        
    def save_checkpoint(self, filename: str, data: list, **kwargs) -> None:
        """Save to in-memory storage."""
        self.storage[filename] = data
        
    def list_checkpoints(self) -> list[str]:
        """List stored checkpoints."""
        return list(self.storage.keys())


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_embedding_model():
    """Create mock embedding model."""
    return MockEmbeddingModel()


@pytest.fixture
def mock_cache_manager(temp_dir):
    """Create mock cache manager."""
    return MockCheckpointManager(temp_dir)


class TestCachedEmbeddingModel:
    """Test cases for CachedEmbeddingModel."""

    def test_init_requires_model_name(self, temp_dir):
        """Test that initialization requires model_name attribute."""
        # Create model without model_name
        class BadModel(BaseEmbeddingModel):
            async def embed(self, texts):
                return []
            def slug(self):
                return "bad"
        
        bad_model = BadModel()
        cache_manager = MockCheckpointManager(temp_dir)
        
        with pytest.raises(ValueError, match="must have a 'model_name' attribute"):
            CachedEmbeddingModel(bad_model, cache_manager)

    def test_init_with_valid_model(self, mock_embedding_model, mock_cache_manager):
        """Test successful initialization."""
        cached_model = CachedEmbeddingModel(mock_embedding_model, mock_cache_manager)
        
        assert cached_model.embedding_model == mock_embedding_model
        assert cached_model.cache_manager == mock_cache_manager
        
    def test_slug_with_cache(self, mock_embedding_model, mock_cache_manager):
        """Test slug generation with caching enabled."""
        cached_model = CachedEmbeddingModel(mock_embedding_model, mock_cache_manager)
        
        expected_slug = "cached:mock:test-model"
        assert cached_model.slug() == expected_slug
        
    def test_slug_without_cache(self, mock_embedding_model):
        """Test slug generation without caching."""
        cached_model = CachedEmbeddingModel(mock_embedding_model, None)
        
        expected_slug = "mock:test-model"
        assert cached_model.slug() == expected_slug

    @pytest.mark.asyncio
    async def test_embed_without_cache(self, mock_embedding_model):
        """Test embedding without cache manager."""
        cached_model = CachedEmbeddingModel(mock_embedding_model, None)
        
        texts = ["hello", "world"]
        result = await cached_model.embed(texts)
        
        assert len(result) == 2
        assert mock_embedding_model.embed_call_count == 1

    @pytest.mark.asyncio
    async def test_embed_with_empty_cache(self, mock_embedding_model, mock_cache_manager):
        """Test embedding with empty cache."""
        cached_model = CachedEmbeddingModel(mock_embedding_model, mock_cache_manager)
        
        texts = ["hello", "world"]
        result = await cached_model.embed(texts)
        
        assert len(result) == 2
        assert mock_embedding_model.embed_call_count == 1
        # Check that embeddings were cached
        assert len(mock_cache_manager.storage) == 2

    @pytest.mark.asyncio
    async def test_embed_with_partial_cache(self, mock_embedding_model, mock_cache_manager):
        """Test embedding with some cached results."""
        cached_model = CachedEmbeddingModel(mock_embedding_model, mock_cache_manager)
        
        # First call - cache "hello"
        await cached_model.embed(["hello"])
        assert mock_embedding_model.embed_call_count == 1
        
        # Second call - "hello" cached, "world" new
        result = await cached_model.embed(["hello", "world"])
        assert len(result) == 2
        assert mock_embedding_model.embed_call_count == 2  # Only one more call

    @pytest.mark.asyncio
    async def test_embed_fully_cached(self, mock_embedding_model, mock_cache_manager):
        """Test embedding with all results cached."""
        cached_model = CachedEmbeddingModel(mock_embedding_model, mock_cache_manager)
        
        texts = ["hello", "world"]
        
        # First call
        result1 = await cached_model.embed(texts)
        assert mock_embedding_model.embed_call_count == 1
        
        # Second call - should use cache entirely
        result2 = await cached_model.embed(texts)
        assert mock_embedding_model.embed_call_count == 1  # No additional calls
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, mock_embedding_model, mock_cache_manager):
        """Test embedding empty list."""
        cached_model = CachedEmbeddingModel(mock_embedding_model, mock_cache_manager)
        
        result = await cached_model.embed([])
        assert result == []
        assert mock_embedding_model.embed_call_count == 0

    def test_cache_key_generation(self, mock_embedding_model, mock_cache_manager):
        """Test cache key generation."""
        cached_model = CachedEmbeddingModel(
            mock_embedding_model, 
            mock_cache_manager,
            temperature=0.5,
            batch_size=10
        )
        
        key1 = cached_model._generate_cache_key("hello")
        key2 = cached_model._generate_cache_key("hello")
        key3 = cached_model._generate_cache_key("world")
        
        # Same text should generate same key
        assert key1 == key2
        # Different text should generate different key
        assert key1 != key3
        # Keys should be consistent format (hex string)
        assert len(key1) == 64  # SHA256 hex string length
        assert all(c in '0123456789abcdef' for c in key1)

    @pytest.mark.asyncio
    async def test_cache_with_kwargs(self, temp_dir):
        """Test that cache keys include kwargs."""
        model1 = MockEmbeddingModel("test-model")
        model2 = MockEmbeddingModel("test-model")
        cache_manager = MockCheckpointManager(temp_dir)
        
        # Same model, different kwargs
        cached_model1 = CachedEmbeddingModel(model1, cache_manager, temperature=0.5)
        cached_model2 = CachedEmbeddingModel(model2, cache_manager, temperature=0.7)
        
        # Should generate different cache keys due to different kwargs
        key1 = cached_model1._generate_cache_key("hello")
        key2 = cached_model2._generate_cache_key("hello")
        
        assert key1 != key2
