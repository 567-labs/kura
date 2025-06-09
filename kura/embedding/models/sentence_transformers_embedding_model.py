from kura.base_classes import BaseEmbeddingModel
from kura.embedding.utils import batch_texts
from tenacity import retry, wait_fixed, stop_after_attempt
import logging

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_batch_size: int = 128,
    ):
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info(
            f"Initializing SentenceTransformerEmbeddingModel with model={model_name}, batch_size={model_batch_size}"
        )
        try:
            self.model = SentenceTransformer(model_name)
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
