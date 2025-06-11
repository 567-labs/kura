def batch_texts(texts: list[str], batch_size: int) -> list[list[str]]:
    """Helper function to divide a list of texts into batches.

    Args:
        texts: List of texts to batch
        batch_size: Maximum size of each batch

    Returns:
        List of batches, where each batch is a list of texts
    """
    if not texts:
        return []

    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batches.append(batch)
    return batches
