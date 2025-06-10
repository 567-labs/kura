from kura.base_classes import BaseEmbeddingModel
from kura.embedding.utils import batch_texts
from asyncio import Semaphore, gather
from tenacity import retry, wait_fixed, stop_after_attempt
from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        model_batch_size: int = 50,
        n_concurrent_jobs: int = 5,
    ):
        self.client = AsyncOpenAI()
        self.model_name = model_name
        self._model_batch_size = model_batch_size
        self._n_concurrent_jobs = n_concurrent_jobs
        self._semaphore = Semaphore(n_concurrent_jobs)
        logger.info(
            f"Initialized OpenAIEmbeddingModel with model={model_name}, batch_size={model_batch_size}, concurrent_jobs={n_concurrent_jobs}"
        )

    def slug(self):
        return f"openai:{self.model_name}-batchsize:{self._model_batch_size}-concurrent:{self._n_concurrent_jobs}"

    @retry(wait=wait_fixed(3), stop=stop_after_attempt(3))
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch of texts."""
        async with self._semaphore:
            try:
                logger.debug(
                    f"Embedding batch of {len(texts)} texts using model {self.model_name}"
                )
                resp = await self.client.embeddings.create(
                    input=texts, model=self.model_name
                )
                embeddings = [item.embedding for item in resp.data]
                logger.debug(
                    f"Successfully embedded batch of {len(texts)} texts, got {len(embeddings)} embeddings"
                )
                return embeddings
            except Exception as e:
                logger.error(f"Failed to embed batch of {len(texts)} texts: {e}")
                raise

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.debug("Empty text list provided, returning empty embeddings")
            return []

        logger.info(f"Starting embedding of {len(texts)} texts using {self.model_name}")

        # Create batches
        batches = batch_texts(texts, self._model_batch_size)
        logger.debug(
            f"Split {len(texts)} texts into {len(batches)} batches of size {self._model_batch_size}"
        )

        # Process all batches concurrently
        tasks = [self._embed_batch(batch) for batch in batches]
        try:
            results_list_of_lists = await gather(*tasks)
            logger.debug(f"Completed embedding {len(batches)} batches")
        except Exception as e:
            logger.error(f"Failed to embed texts: {e}")
            raise

        # Flatten results
        embeddings = []
        for result_batch in results_list_of_lists:
            embeddings.extend(result_batch)

        logger.info(
            f"Successfully embedded {len(texts)} texts, produced {len(embeddings)} embeddings"
        )
        return embeddings
