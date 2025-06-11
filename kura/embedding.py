from kura.base_classes import BaseEmbeddingModel
from kura.types import ConversationSummary
import logging
from typing import Union
from kura.utils import batch_texts
from asyncio import Semaphore, gather
from tenacity import retry, wait_fixed, stop_after_attempt
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


async def embed_summaries(
    summaries: list[ConversationSummary], embedding_model: BaseEmbeddingModel
) -> list[dict[str, Union[ConversationSummary, list[float]]]]:
    """Embeds conversation summaries and returns items ready for clustering."""
    if not summaries:
        return []

    logger.info(f"Processing {len(summaries)} summaries")
    texts_to_embed = [str(item) for item in summaries]

    try:
        embeddings = await embedding_model.embed(texts_to_embed)
    except Exception as e:
        logger.error(f"Error embedding summaries: {e}")
        raise

    return [
        {"item": summary, "embedding": embedding}
        for summary, embedding in zip(summaries, embeddings)
    ]


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


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_batch_size: int = 128,
        device: str = "cpu",
    ):
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info(
            f"Initializing SentenceTransformerEmbeddingModel with model={model_name}, batch_size={model_batch_size}"
        )
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.model_name = model_name
            self._model_batch_size = model_batch_size
            logger.info(f"Successfully loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
            raise

    def slug(self) -> str:
        return f"sentence-transformers:{self.model_name}-batchsize:{self._model_batch_size}"

    @retry(wait=wait_fixed(3), stop=stop_after_attempt(3))
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.debug("Empty text list provided, returning empty embeddings")
            return []

        logger.info(
            f"Starting embedding of {len(texts)} texts using SentenceTransformer"
        )

        # Create batches
        batches = batch_texts(texts, self._model_batch_size)
        logger.debug(
            f"Split {len(texts)} texts into {len(batches)} batches of size {self._model_batch_size}"
        )

        # Process all batches
        embeddings = []
        try:
            for i, batch in enumerate(batches):
                logger.debug(
                    f"Processing batch {i + 1}/{len(batches)} with {len(batch)} texts"
                )
                batch_embeddings = self.model.encode(batch).tolist()
                embeddings.extend(batch_embeddings)
                logger.debug(f"Completed batch {i + 1}/{len(batches)}")

            logger.info(
                f"Successfully embedded {len(texts)} texts using SentenceTransformer, produced {len(embeddings)} embeddings"
            )
        except Exception as e:
            logger.error(f"Failed to embed texts using SentenceTransformer: {e}")
            raise

        return embeddings
