"""
Migration utilities for converting checkpoints between different formats.

This module provides functions to migrate from JSONL checkpoints to
HuggingFace dataset checkpoints, verify migrations, and analyze
migration benefits.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
import os

logger = logging.getLogger(__name__)

# Optional imports for HuggingFace functionality
try:
    from .hf_dataset import HFDatasetCheckpointManager, HF_DATASETS_AVAILABLE
except ImportError:
    HFDatasetCheckpointManager = None
    HF_DATASETS_AVAILABLE = False

from .jsonl import JSONLCheckpointManager
from kura.types import Conversation, ConversationSummary, Cluster
from kura.types.dimensionality import ProjectedCluster


def migrate_jsonl_to_hf_dataset(
    source_dir: str,
    target_dir: str,
    hub_repo: Optional[str] = None,
    hub_token: Optional[str] = None,
    compression: Optional[str] = "gzip",
    delete_source: bool = False,
) -> Dict[str, bool]:
    """Migrate JSONL checkpoints to HuggingFace dataset format.
    
    Args:
        source_dir: Directory containing JSONL checkpoints
        target_dir: Directory for HuggingFace dataset checkpoints
        hub_repo: Optional HuggingFace Hub repository name
        hub_token: Optional HuggingFace Hub token
        compression: Compression algorithm ('gzip', 'lz4', 'zstd', None)
        delete_source: Whether to delete source files after migration
        
    Returns:
        Dictionary mapping checkpoint names to success status
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets is required for migration. "
            "Install with: uv pip install datasets"
        )
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist")
    
    # Initialize checkpoint managers
    jsonl_manager = JSONLCheckpointManager(str(source_path))
    hf_manager = HFDatasetCheckpointManager(
        str(target_path),
        hub_repo=hub_repo,
        hub_token=hub_token,
        compression=compression,
    )
    
    # Get list of JSONL checkpoints
    jsonl_files = jsonl_manager.list_checkpoints()
    if not jsonl_files:
        logger.warning(f"No JSONL checkpoints found in {source_dir}")
        return {}
    
    results = {}
    
    for filename in jsonl_files:
        checkpoint_name = Path(filename).stem  # Remove .jsonl extension
        
        try:
            logger.info(f"Migrating checkpoint: {checkpoint_name}")
            
            # Determine the data type and load accordingly
            model_class, checkpoint_type = _infer_checkpoint_type(filename, source_path)
            
            if model_class is None:
                logger.warning(f"Could not determine type for {filename}, skipping")
                results[checkpoint_name] = False
                continue
            
            # Load data from JSONL
            data = jsonl_manager.load_checkpoint(checkpoint_name, model_class)
            
            if data is None or len(data) == 0:
                logger.warning(f"No data found in {filename}, skipping")
                results[checkpoint_name] = False
                continue
            
            # Save to HuggingFace dataset format
            hf_manager.save_checkpoint(
                checkpoint_name, 
                data, 
                checkpoint_type=checkpoint_type
            )
            
            logger.info(f"Successfully migrated {checkpoint_name} ({len(data)} items)")
            results[checkpoint_name] = True
            
            # Delete source file if requested
            if delete_source:
                jsonl_manager.delete_checkpoint(filename)
                logger.info(f"Deleted source file: {filename}")
                
        except Exception as e:
            logger.error(f"Failed to migrate {checkpoint_name}: {e}")
            results[checkpoint_name] = False
    
    return results


def _infer_checkpoint_type(filename: str, source_path: Path) -> tuple[Optional[type], str]:
    """Infer the checkpoint type from filename and sample data.
    
    Returns:
        Tuple of (model_class, checkpoint_type_string)
    """
    # First try to infer from filename patterns
    filename_lower = filename.lower()
    
    if "conversation" in filename_lower and "summary" not in filename_lower:
        return Conversation, "conversations"
    elif "summary" in filename_lower or "summaries" in filename_lower:
        return ConversationSummary, "summaries"
    elif "projected" in filename_lower or "dimensionality" in filename_lower:
        return ProjectedCluster, "projected_clusters"
    elif "cluster" in filename_lower:
        return Cluster, "clusters"
    
    # If filename doesn't give us a clue, try to sample the data
    try:
        checkpoint_path = source_path / filename
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    data = json.loads(first_line)
                    
                    # Check for conversation-specific fields
                    if "messages" in data:
                        return Conversation, "conversations"
                    # Check for summary-specific fields
                    elif "summary" in data and "chat_id" in data:
                        return ConversationSummary, "summaries"
                    # Check for cluster fields
                    elif "chat_ids" in data:
                        # Check if it has projection coordinates
                        if "x_coord" in data or "y_coord" in data:
                            return ProjectedCluster, "projected_clusters"
                        else:
                            return Cluster, "clusters"
    except Exception as e:
        logger.error(f"Error sampling {filename}: {e}")
    
    return None, ""


def verify_migration(
    source_dir: str, 
    target_dir: str, 
    detailed: bool = False
) -> Dict[str, Any]:
    """Verify that migration was successful by comparing data.
    
    Args:
        source_dir: Directory with original JSONL checkpoints
        target_dir: Directory with HuggingFace dataset checkpoints
        detailed: Whether to return detailed failure information
        
    Returns:
        Dictionary with verification results
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets is required for verification. "
            "Install with: uv pip install datasets"
        )
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Initialize managers
    jsonl_manager = JSONLCheckpointManager(str(source_path))
    hf_manager = HFDatasetCheckpointManager(str(target_path))
    
    # Get lists of checkpoints
    jsonl_checkpoints = set(Path(f).stem for f in jsonl_manager.list_checkpoints())
    hf_checkpoints = set(hf_manager.list_checkpoints())
    
    results = {
        "total_checkpoints": len(jsonl_checkpoints),
        "verified_checkpoints": 0,
        "failed_checkpoints": [],
        "missing_checkpoints": [],
    }
    
    # Check for missing checkpoints
    missing = jsonl_checkpoints - hf_checkpoints
    if missing:
        results["missing_checkpoints"] = list(missing)
        logger.warning(f"Missing checkpoints in target: {missing}")
    
    # Verify each checkpoint
    for checkpoint_name in jsonl_checkpoints:
        if checkpoint_name not in hf_checkpoints:
            continue
            
        try:
            # Infer type and load data
            original_file = next(
                (f for f in jsonl_manager.list_checkpoints() 
                 if Path(f).stem == checkpoint_name), 
                None
            )
            
            if not original_file:
                continue
                
            model_class, checkpoint_type = _infer_checkpoint_type(original_file, source_path)
            
            if model_class is None:
                continue
            
            # Load from both sources
            jsonl_data = jsonl_manager.load_checkpoint(checkpoint_name, model_class)
            hf_data = hf_manager.load_checkpoint(
                checkpoint_name, 
                model_class, 
                checkpoint_type=checkpoint_type
            )
            
            if jsonl_data is None and hf_data is None:
                results["verified_checkpoints"] += 1
                continue
            
            if jsonl_data is None or hf_data is None:
                results["failed_checkpoints"].append(
                    f"{checkpoint_name}: Data mismatch (one is None)"
                )
                continue
            
            if len(jsonl_data) != len(hf_data):
                results["failed_checkpoints"].append(
                    f"{checkpoint_name}: Length mismatch ({len(jsonl_data)} vs {len(hf_data)})"
                )
                continue
            
            # For detailed verification, compare actual content
            if detailed:
                try:
                    # Compare serialized versions for consistency
                    jsonl_serialized = {item.model_dump_json() for item in jsonl_data}
                    hf_serialized = {item.model_dump_json() for item in hf_data}
                    
                    if jsonl_serialized != hf_serialized:
                        results["failed_checkpoints"].append(
                            f"{checkpoint_name}: Content mismatch"
                        )
                        continue
                except Exception as e:
                    results["failed_checkpoints"].append(
                        f"{checkpoint_name}: Comparison error: {e}"
                    )
                    continue
            
            results["verified_checkpoints"] += 1
            logger.info(f"Verified checkpoint: {checkpoint_name}")
            
        except Exception as e:
            results["failed_checkpoints"].append(
                f"{checkpoint_name}: Verification error: {e}"
            )
            logger.error(f"Failed to verify {checkpoint_name}: {e}")
    
    return results


def estimate_migration_benefits(checkpoint_dir: str) -> Dict[str, Any]:
    """Analyze checkpoints and estimate migration benefits.
    
    Args:
        checkpoint_dir: Directory containing checkpoints to analyze
        
    Returns:
        Dictionary with analysis results and estimated benefits
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return {
            "error": f"Directory {checkpoint_dir} does not exist",
            "current_format": "unknown",
            "total_files": 0,
            "total_size_mb": 0,
        }
    
    # Initialize manager to detect format
    jsonl_manager = JSONLCheckpointManager(str(checkpoint_path))
    jsonl_files = jsonl_manager.list_checkpoints()
    
    # Check if we have HF datasets
    hf_checkpoints = []
    if HF_DATASETS_AVAILABLE:
        hf_manager = HFDatasetCheckpointManager(str(checkpoint_path))
        hf_checkpoints = hf_manager.list_checkpoints()
    
    # Determine current format
    if jsonl_files and not hf_checkpoints:
        current_format = "jsonl"
        files_to_analyze = jsonl_files
    elif hf_checkpoints and not jsonl_files:
        current_format = "hf-dataset"
        files_to_analyze = hf_checkpoints
    elif jsonl_files and hf_checkpoints:
        current_format = "mixed"
        files_to_analyze = jsonl_files + hf_checkpoints
    else:
        current_format = "empty"
        files_to_analyze = []
    
    # Calculate total size
    total_size = 0
    for item in checkpoint_path.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
    
    total_size_mb = total_size / (1024 * 1024)
    
    results = {
        "current_format": current_format,
        "total_files": len(files_to_analyze),
        "total_size_mb": round(total_size_mb, 2),
        "estimated_benefits": {},
        "migration_priority": "low",
    }
    
    # Estimate benefits based on current format
    if current_format == "jsonl":
        results["estimated_benefits"] = {
            "performance": "50-80% faster loading for large datasets",
            "memory_efficiency": "Memory-mapped access, no need to load everything",
            "compression": "Built-in compression reduces file size by 30-60%",
            "features": "Advanced filtering, streaming, and querying capabilities",
            "scalability": "Better handling of datasets larger than available RAM",
            "cloud_integration": "Direct integration with HuggingFace Hub",
        }
        
        # Determine migration priority based on size
        if total_size_mb > 1000:  # > 1GB
            results["migration_priority"] = "high"
        elif total_size_mb > 100:  # > 100MB
            results["migration_priority"] = "medium"
        else:
            results["migration_priority"] = "low"
            
        # Estimate post-migration size (rough estimate)
        compression_ratio = 0.4  # Assume 60% compression
        results["estimated_compressed_size_mb"] = round(total_size_mb * compression_ratio, 2)
        results["estimated_space_savings_mb"] = round(total_size_mb * (1 - compression_ratio), 2)
        
    elif current_format == "hf-dataset":
        results["estimated_benefits"] = {
            "status": "Already using HuggingFace datasets format",
            "current_benefits": "Optimized performance and features already active",
        }
        results["migration_priority"] = "none"
        
    elif current_format == "mixed":
        results["estimated_benefits"] = {
            "consolidation": "Standardize on single, optimized format",
            "consistency": "Uniform API and performance characteristics",
            "maintenance": "Simplified checkpoint management",
        }
        results["migration_priority"] = "medium"
        
    else:  # empty
        results["estimated_benefits"] = {
            "none": "No checkpoints found to analyze",
        }
        results["migration_priority"] = "none"
    
    return results