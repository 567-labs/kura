"""
HuggingFace Datasets-based checkpoint manager.

This checkpoint system uses HuggingFace datasets for storing intermediate
pipeline results, providing better performance, scalability, and features
like streaming, versioning, and cloud storage integration.
"""

import logging
import importlib.util
from typing import Optional, TypeVar, List, Dict, Any, Callable
from pathlib import Path
from datetime import datetime
import json
from pydantic import BaseModel
from kura.types import Conversation, ConversationSummary, Cluster
from kura.types.dimensionality import ProjectedCluster
from kura.base_classes import BaseCheckpointManager

# Check if HuggingFace datasets is available
HF_DATASETS_AVAILABLE = importlib.util.find_spec("datasets") is not None

# Conditional imports to avoid shadowing
if HF_DATASETS_AVAILABLE:
    from datasets import (
        Dataset,
        Features,
        Value,
        Sequence,
        load_from_disk,
        load_dataset,
    )
    from datasets.features import ClassLabel


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class HFDatasetCheckpointManager(BaseCheckpointManager):
    """Handles checkpoint loading and saving using HuggingFace datasets.

    This checkpoint system provides better performance, scalability, and features
    compared to the JSONL-based system:
    - Memory-mapped files for efficient access without loading everything into memory
    - Streaming support for datasets larger than available RAM
    - Built-in compression and optimization
    - Version control and dataset cards via HuggingFace Hub
    - Rich querying and filtering capabilities
    - Schema validation and type safety
    """

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        enabled: bool = True,
        hub_repo: Optional[str] = None,
        hub_token: Optional[str] = None,
        streaming: bool = False,
        compression: Optional[str] = "gzip",
    ):
        """Initialize HuggingFace dataset checkpoint manager.

        Args:
            checkpoint_dir: Directory for saving checkpoints locally
            enabled: Whether checkpointing is enabled
            hub_repo: Optional HuggingFace Hub repository name for cloud storage
            hub_token: Optional HuggingFace Hub token for authentication
            streaming: Whether to use streaming mode by default
            compression: Compression algorithm to use ('gzip', 'lz4', 'zstd', None)
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets is required for HFDatasetCheckpointManager. "
                "Install with: pip install datasets"
            )

        super().__init__(checkpoint_dir, enabled=enabled)

        self.hub_repo = hub_repo
        self.hub_token = hub_token
        self.streaming = streaming
        self.compression = compression

        # Define schemas for different data types
        self._schemas = self._define_schemas()

    def setup_checkpoint_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist."""
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")

    def _define_schemas(self) -> Dict[str, Features]:
        """Define HuggingFace dataset schemas for different data types."""

        # Schema for conversations
        conversation_schema = Features(
            {
                "chat_id": Value("string"),
                "created_at": Value("timestamp[ns]"),
                "messages": Sequence(
                    {
                        "created_at": Value("timestamp[ns]"),
                        "role": ClassLabel(names=["user", "assistant"]),
                        "content": Value("string"),
                    }
                ),
                "metadata": {
                    # Dynamic metadata fields - we'll handle this specially
                    "keys": Sequence(Value("string")),
                    "values": Sequence(Value("string")),
                },
            }
        )

        # Schema for conversation summaries
        summary_schema = Features(
            {
                "chat_id": Value("string"),
                "summary": Value("string"),
                "request": Value("string"),
                "topic": Value("string"),
                "languages": Sequence(Value("string")),
                "task": Value("string"),
                "concerning_score": Value("int8"),
                "user_frustration": Value("int8"),
                "assistant_errors": Sequence(Value("string")),
                "metadata": {
                    "keys": Sequence(Value("string")),
                    "values": Sequence(Value("string")),
                },
                "embedding": Sequence(Value("float32")),
            }
        )

        # Schema for clusters
        cluster_schema = Features(
            {
                "id": Value("string"),
                "name": Value("string"),
                "description": Value("string"),
                "slug": Value("string"),
                "chat_ids": Sequence(Value("string")),
                "parent_id": Value("string"),
                "count": Value("int32"),
            }
        )

        # Schema for projected clusters (extends cluster schema)
        projected_cluster_schema = Features(
            {
                "id": Value("string"),
                "name": Value("string"),
                "description": Value("string"),
                "slug": Value("string"),
                "chat_ids": Sequence(Value("string")),
                "parent_id": Value("string"),
                "count": Value("int32"),
                "x_coord": Value("float32"),
                "y_coord": Value("float32"),
                "level": Value("int32"),
            }
        )

        return {
            "conversations": conversation_schema,
            "summaries": summary_schema,
            "clusters": cluster_schema,
            "projected_clusters": projected_cluster_schema,
        }

    def _get_checkpoint_path(self, checkpoint_name: str) -> Path:
        """Get the local path for a checkpoint dataset."""
        return self.checkpoint_dir / checkpoint_name

    def _serialize_metadata(self, metadata: dict) -> Dict[str, List[str]]:
        """Convert metadata dict to HF-compatible format."""
        keys = []
        values = []

        for k, v in metadata.items():
            keys.append(k)
            if isinstance(v, (list, tuple)):
                values.append(json.dumps(v))
            else:
                values.append(str(v))

        return {"keys": keys, "values": values}

    def _deserialize_metadata(self, serialized: Dict[str, List[str]]) -> dict:
        """Convert HF metadata format back to dict."""
        metadata = {}

        for k, v in zip(serialized["keys"], serialized["values"]):
            try:
                # Try to parse as JSON first (for lists)
                metadata[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                metadata[k] = v

        return metadata

    def _model_to_dict(self, model: BaseModel, checkpoint_type: str) -> dict:
        """Convert Pydantic model to dictionary suitable for HF datasets."""
        data = model.model_dump()

        if checkpoint_type == "conversations":
            # Handle Conversation model
            data["created_at"] = int(
                model.created_at.timestamp() * 1_000_000_000  # ty: ignore
            )  # Convert to nanoseconds
            for msg in data["messages"]:
                # Handle datetime objects correctly - they might already be datetime objects
                if isinstance(msg["created_at"], str):
                    msg["created_at"] = int(
                        datetime.fromisoformat(msg["created_at"]).timestamp()
                        * 1_000_000_000
                    )
                else:
                    # Already a datetime object
                    msg["created_at"] = int(
                        msg["created_at"].timestamp() * 1_000_000_000
                    )
            data["metadata"] = self._serialize_metadata(data["metadata"])

        elif checkpoint_type == "summaries":
            # Handle ConversationSummary model
            data["metadata"] = self._serialize_metadata(data["metadata"])
            # Handle optional fields
            for field in ["request", "topic", "task"]:
                if data.get(field) is None:
                    data[field] = ""
            if data.get("languages") is None:
                data["languages"] = []
            if data.get("assistant_errors") is None:
                data["assistant_errors"] = []
            if data.get("embedding") is None:
                data["embedding"] = []
            for field in ["concerning_score", "user_frustration"]:
                if data.get(field) is None:
                    data[field] = 0

        elif checkpoint_type in ["clusters", "projected_clusters"]:
            # Handle Cluster and ProjectedCluster models
            # parent_id can be None, convert to empty string for HF datasets
            if data.get("parent_id") is None:
                data["parent_id"] = ""
            # Add computed field 'count'
            data["count"] = len(data["chat_ids"])

        return data

    def _dict_to_model(
        self, data: dict, model_class: type[T], checkpoint_type: str
    ) -> T:
        """Convert dictionary from HF datasets back to Pydantic model."""

        # Make a copy to avoid modifying the original data
        data = dict(data)

        if checkpoint_type == "conversations":
            # Handle Conversation model
            # Handle timestamp conversion - could be int, pandas Timestamp, or already datetime
            if isinstance(data["created_at"], (int, float)):
                data["created_at"] = datetime.fromtimestamp(
                    data["created_at"] / 1_000_000_000
                )
            elif hasattr(data["created_at"], "timestamp"):
                # Pandas Timestamp object
                data["created_at"] = data["created_at"].to_pydatetime()
            # Otherwise it's already a datetime object

            # Handle messages - HF datasets converts list of dicts to dict of lists
            messages = data.get("messages", [])
            processed_messages = []

            # Check if messages is in HF datasets format (dict with lists) or regular format
            if isinstance(messages, dict) and "created_at" in messages:
                # Convert from column-oriented format back to row-oriented format
                num_messages = len(messages["created_at"])
                for i in range(num_messages):
                    msg_dict = {}

                    # Extract each field for this message
                    for field in ["created_at", "role", "content"]:
                        if field in messages:
                            msg_dict[field] = messages[field][i]

                    # Convert timestamp
                    if isinstance(msg_dict["created_at"], (int, float)):
                        msg_dict["created_at"] = datetime.fromtimestamp(
                            msg_dict["created_at"] / 1_000_000_000
                        )
                    elif hasattr(msg_dict["created_at"], "timestamp"):
                        # Pandas Timestamp object
                        msg_dict["created_at"] = msg_dict["created_at"].to_pydatetime()
                    # Otherwise it's already a datetime object

                    # Convert role from ClassLabel index back to string
                    if isinstance(msg_dict.get("role"), int):
                        role_names = ["user", "assistant"]
                        if 0 <= msg_dict["role"] < len(role_names):
                            msg_dict["role"] = role_names[msg_dict["role"]]

                    processed_messages.append(msg_dict)
            else:
                # Handle regular list format
                for i, msg in enumerate(messages):
                    # Handle case where msg might be a dict-like object but not exactly a dict
                    if hasattr(msg, "keys"):
                        msg_dict = dict(msg)
                    elif isinstance(msg, dict):
                        msg_dict = msg
                    else:
                        # Debug: print the actual type and content
                        logger.error(
                            f"Unexpected message type at index {i}: {type(msg)} = {msg}"
                        )
                        raise ValueError(f"Unexpected message format: {type(msg)}")

                    # Convert timestamp
                    if isinstance(msg_dict["created_at"], (int, float)):
                        msg_dict["created_at"] = datetime.fromtimestamp(
                            msg_dict["created_at"] / 1_000_000_000
                        )
                    elif hasattr(msg_dict["created_at"], "timestamp"):
                        # Pandas Timestamp object
                        msg_dict["created_at"] = msg_dict["created_at"].to_pydatetime()
                    # Otherwise it's already a datetime object

                    processed_messages.append(msg_dict)

            data["messages"] = processed_messages
            data["metadata"] = self._deserialize_metadata(data["metadata"])

        elif checkpoint_type == "summaries":
            # Handle ConversationSummary model
            data["metadata"] = self._deserialize_metadata(data["metadata"])
            # Convert empty strings back to None for optional fields
            for field in ["request", "topic", "task"]:
                if data.get(field) == "":
                    data[field] = None
            for field in ["concerning_score", "user_frustration"]:
                if data.get(field) == 0:
                    data[field] = None
            if not data.get("languages"):
                data["languages"] = None
            if not data.get("assistant_errors"):
                data["assistant_errors"] = None
            if not data.get("embedding"):
                data["embedding"] = None

        elif checkpoint_type in ["clusters", "projected_clusters"]:
            # Handle Cluster and ProjectedCluster models
            if data.get("parent_id") == "":
                data["parent_id"] = None
            # Remove computed field 'count' as it's computed in the model
            data.pop("count", None)

        return model_class.model_validate(data)

    def save_checkpoint(self, filename: str, data: List[T], **kwargs) -> None:
        """Save data to a checkpoint using HuggingFace datasets.

        Args:
            filename: Name of the checkpoint (without extension)
            data: List of model instances to save
            **kwargs: Additional arguments, including:
                checkpoint_type: Type of checkpoint for schema selection
        """
        if not self.enabled or not data:
            return

        # Get checkpoint_type from kwargs or infer from data
        checkpoint_type = kwargs.get("checkpoint_type")
        if not checkpoint_type:
            checkpoint_type = self._infer_checkpoint_type(data[0])

        # Convert models to dictionaries
        dict_data = [self._model_to_dict(item, checkpoint_type) for item in data]

        # Create dataset with appropriate schema
        schema = self._schemas.get(checkpoint_type)
        if schema:
            dataset = Dataset.from_list(dict_data, features=schema)
        else:
            dataset = Dataset.from_list(dict_data)

        # Save locally
        checkpoint_path = self._get_checkpoint_path(filename)

        # Note: HuggingFace datasets handles compression internally
        # The compression parameter is for reference only
        dataset.save_to_disk(str(checkpoint_path))
        logger.info(
            f"Saved HF dataset checkpoint to {checkpoint_path} with {len(data)} items"
        )

        # Save to hub if configured
        if self.hub_repo:
            try:
                dataset.push_to_hub(
                    self.hub_repo,
                    config_name=filename,
                    token=self.hub_token,
                    commit_message=f"Update {filename} checkpoint with {len(data)} items",
                )
                logger.info(
                    f"Pushed checkpoint to HuggingFace Hub: {self.hub_repo}/{filename}"
                )
            except Exception as e:
                logger.warning(f"Failed to push to hub: {e}")

    def _infer_checkpoint_type(self, data_item: BaseModel) -> str:
        """Infer checkpoint type from the data model class."""
        from kura.types import Conversation, ConversationSummary, Cluster
        from kura.types.dimensionality import ProjectedCluster

        if isinstance(data_item, Conversation):
            return "conversations"
        elif isinstance(data_item, ConversationSummary):
            return "summaries"
        elif isinstance(data_item, ProjectedCluster):
            return "projected_clusters"
        elif isinstance(data_item, Cluster):
            return "clusters"
        else:
            # Default fallback
            return "unknown"

    def load_checkpoint(
        self, filename: str, model_class: type[T], **kwargs
    ) -> Optional[List[T]]:
        """Load data from a checkpoint using HuggingFace datasets.

        Args:
            filename: Name of the checkpoint (without extension)
            model_class: Pydantic model class for deserializing the data
            **kwargs: Additional arguments, including:
                streaming: Whether to use streaming mode (overrides default)
                checkpoint_type: Type of checkpoint for proper deserialization

        Returns:
            List of model instances if checkpoint exists, None otherwise
        """
        if not self.enabled:
            return None

        # Get parameters from kwargs
        streaming = kwargs.get("streaming")
        checkpoint_type = kwargs.get("checkpoint_type", "")

        use_streaming = streaming if streaming is not None else self.streaming

        # Try to load from hub first if configured
        if self.hub_repo:
            try:
                if use_streaming:
                    dataset = load_dataset(
                        self.hub_repo,
                        name=filename,
                        streaming=True,
                        token=self.hub_token,
                        split=kwargs.get("split", "train"),
                    )
                    # For streaming, we need to collect all items
                    dict_data = list(dataset)
                else:
                    dataset = load_dataset(
                        self.hub_repo,
                        name=filename,
                        token=self.hub_token,
                        split=kwargs.get("split", "train"),
                    )
                    dict_data = dataset

                logger.info(
                    f"Loaded checkpoint from HuggingFace Hub: {self.hub_repo}/{filename}"
                )

            except Exception as e:
                logger.warning(f"Failed to load from hub, trying local: {e}")
                dataset = None
        else:
            dataset = None

        # Try local checkpoint if hub failed or not configured
        if dataset is None:
            checkpoint_path = self._get_checkpoint_path(filename)
            if not checkpoint_path.exists():
                return None

            try:
                if use_streaming:
                    # HF datasets doesn't support streaming from local disk directly
                    # Load normally but process in chunks if needed
                    dataset = load_from_disk(str(checkpoint_path))
                    dict_data = dataset
                else:
                    dataset = load_from_disk(str(checkpoint_path))
                    dict_data = dataset

                logger.info(f"Loaded checkpoint from local disk: {checkpoint_path}")

            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
                return None

        # Convert back to Pydantic models
        try:
            # Infer checkpoint type from model class if not provided
            if not checkpoint_type:
                if model_class == Conversation:
                    checkpoint_type = "conversations"
                elif model_class == ConversationSummary:
                    checkpoint_type = "summaries"
                elif model_class == ProjectedCluster:
                    checkpoint_type = "projected_clusters"
                elif model_class == Cluster:
                    checkpoint_type = "clusters"

            models = [
                self._dict_to_model(item, model_class, checkpoint_type)
                for item in dict_data
            ]
            logger.info(f"Converted {len(models)} items to {model_class.__name__}")
            return models

        except Exception as e:
            logger.error(f"Failed to convert data to {model_class.__name__}: {e}")
            return None

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint datasets."""
        if not self.enabled:
            return []

        checkpoints = []

        # List local checkpoints
        if self.checkpoint_dir.exists():
            for item in self.checkpoint_dir.iterdir():
                if item.is_dir():
                    # Check if it's a valid HF dataset
                    if (item / "dataset_info.json").exists():
                        checkpoints.append(item.name)

        return checkpoints

    def delete_checkpoint(self, filename: str) -> bool:
        """Delete a checkpoint dataset.

        Args:
            filename: Name of the checkpoint to delete

        Returns:
            True if checkpoint was deleted, False if it didn't exist
        """
        if not self.enabled:
            return False

        checkpoint_path = self._get_checkpoint_path(filename)
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            import shutil

            shutil.rmtree(checkpoint_path)
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
            return True
        return False

    def get_checkpoint_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get information about a checkpoint dataset.

        Args:
            filename: Name of the checkpoint

        Returns:
            Dictionary with checkpoint metadata and statistics
        """
        if not self.enabled:
            return None

        checkpoint_path = self._get_checkpoint_path(filename)
        if not checkpoint_path.exists():
            return None

        try:
            dataset = load_from_disk(str(checkpoint_path))
            info = {
                "num_rows": len(dataset),
                "num_columns": len(dataset.column_names),
                "column_names": dataset.column_names,
                "features": str(dataset.features),  # ty: ignore
                "size_bytes": sum(
                    f.stat().st_size for f in checkpoint_path.rglob("*") if f.is_file()
                ),
            }

            # Add dataset info if available
            info_file = checkpoint_path / "dataset_info.json"
            if info_file.exists():
                with open(info_file) as f:
                    dataset_info = json.load(f)
                    info.update(dataset_info)

            return info

        except Exception as e:
            logger.error(f"Failed to get info for checkpoint {filename}: {e}")
            return None

    def filter_checkpoint(
        self,
        filename: str,
        filter_fn: Callable[[Dict[str, Any]], bool],
        model_class: type[T],
        **kwargs,
    ) -> Optional[List[T]]:
        """Filter a checkpoint dataset without loading everything into memory.

        Args:
            filename: Name of the checkpoint
            filter_fn: Function to filter rows (takes dict, returns bool)
            model_class: Pydantic model class for results
            **kwargs: Additional arguments, including:
                checkpoint_type: Type of checkpoint for proper deserialization

        Returns:
            List of filtered model instances
        """
        if not self.enabled:
            return None

        # Get checkpoint_type from kwargs or infer
        checkpoint_type = kwargs.get("checkpoint_type", "")

        checkpoint_path = self._get_checkpoint_path(filename)
        if not checkpoint_path.exists():
            return None

        try:
            dataset = load_from_disk(str(checkpoint_path))
            filtered_dataset = dataset.filter(filter_fn)

            # Convert to models
            if not checkpoint_type:
                if model_class == Conversation:
                    checkpoint_type = "conversations"
                elif model_class == ConversationSummary:
                    checkpoint_type = "summaries"
                elif model_class == ProjectedCluster:
                    checkpoint_type = "projected_clusters"
                elif model_class == Cluster:
                    checkpoint_type = "clusters"

            models = [
                self._dict_to_model(item, model_class, checkpoint_type)
                for item in filtered_dataset
            ]

            logger.info(f"Filtered {len(models)} items from {filename}")
            return models

        except Exception as e:
            logger.error(f"Failed to filter checkpoint {filename}: {e}")
            return None
