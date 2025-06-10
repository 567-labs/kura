from kura.base_classes import BaseEmbeddingModel
from kura.types import ConversationSummary
import logging
from typing import Union

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
