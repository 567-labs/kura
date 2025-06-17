import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from kura import StandardLogger, create_logger
from kura.base_classes import BaseClusterLogger


class TestStandardLogger:
    """Test the StandardLogger implementation."""

    def test_logger_creation(self):
        """Test basic logger creation."""
        logger = StandardLogger("test_logger")
        assert logger.name == "test_logger"
        assert not logger.supports_artifacts()  # No artifact dir configured

    def test_logger_with_artifact_dir(self):
        """Test logger with artifact directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = StandardLogger("test_logger", artifact_dir=temp_dir)
            assert logger.supports_artifacts()
            assert logger.artifact_dir == Path(temp_dir)

    def test_log_params(self):
        """Test logging parameters."""
        with patch('kura.logger.logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            
            logger = StandardLogger("test")
            params = {"n_clusters": 8, "algorithm": "kmeans"}
            
            logger.log_params(params)
            
            # Verify logger was called with JSON-formatted params
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "PARAMS:" in call_args
            assert "n_clusters" in call_args
            assert "kmeans" in call_args

    def test_log_metrics(self):
        """Test logging metrics."""
        with patch('kura.logger.logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            
            logger = StandardLogger("test")
            metrics = {"silhouette_score": 0.42, "inertia": 2847.3}
            
            logger.log_metrics(metrics)
            
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "METRICS:" in call_args
            assert "silhouette_score" in call_args

    def test_log_metrics_with_step(self):
        """Test logging metrics with step."""
        with patch('kura.logger.logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            
            logger = StandardLogger("test")
            metrics = {"score": 0.8}
            
            logger.log_metrics(metrics, step=5)
            
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "METRICS:" in call_args
            assert "step" in call_args

    def test_log_errors_with_exception(self):
        """Test logging errors with Exception object."""
        with patch('kura.logger.logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            
            logger = StandardLogger("test")
            error = ValueError("Test error")
            context = {"operation": "clustering"}
            
            logger.log_errors(error, context)
            
            # Should call error with exc_info
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "ERROR with context:" in call_args[0][0]
            assert call_args[1]['exc_info'] == error

    def test_log_errors_with_string(self):
        """Test logging errors with string message."""
        with patch('kura.logger.logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            
            logger = StandardLogger("test")
            error = "Something went wrong"
            
            logger.log_errors(error)
            
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "ERROR: Something went wrong" in call_args

    def test_log_generic(self):
        """Test generic logging method."""
        with patch('kura.logger.logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            
            logger = StandardLogger("test")
            data = "Processing conversations"
            key = "status"
            
            logger.log(data, key)
            
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "LOG:" in call_args
            assert "status" in call_args

    def test_log_artifact_without_support(self):
        """Test artifact logging when artifacts aren't supported."""
        with patch('kura.logger.logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            
            logger = StandardLogger("test")  # No artifact_dir
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                
                try:
                    logger.log_artifact(temp_path, "test_plot")
                    
                    # Should fall back to logging file path
                    mock_logger.info.assert_called()
                    call_args = mock_logger.info.call_args[0][0]
                    assert "artifact_info" in call_args
                finally:
                    temp_path.unlink()

    def test_log_artifact_with_support(self):
        """Test artifact logging when artifacts are supported."""
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir) / "artifacts"
            logger = StandardLogger("test", artifact_dir=artifact_dir)
            
            # Create a test file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file.write(b"fake image data")
                temp_path = Path(temp_file.name)
                
                try:
                    with patch('kura.logger.logging.getLogger') as mock_get_logger:
                        mock_logger = mock_get_logger.return_value
                        
                        logger.log_artifact(temp_path, "test_plot")
                        
                        # Should copy file and log success
                        expected_dest = artifact_dir / "test_plot"
                        assert expected_dest.exists()
                        
                        mock_logger.info.assert_called()
                        call_args = mock_logger.info.call_args[0][0]
                        assert "artifact_copied" in call_args
                        
                finally:
                    temp_path.unlink()

    def test_cluster_summary(self):
        """Test cluster summary logging."""
        with patch('kura.logger.logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            
            logger = StandardLogger("test")
            cluster_data = {
                0: {"size": 45, "coherence": 0.8},
                1: {"size": 32, "coherence": 0.7}
            }
            
            logger.log_cluster_summary(cluster_data)
            
            # Should log each cluster separately
            assert mock_logger.info.call_count == 2

    def test_conversation_sample(self):
        """Test conversation sample logging."""
        with patch('kura.logger.logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            
            logger = StandardLogger("test")
            conversations = ["Conv 1", "Conv 2", "Conv 3", "Conv 4"]
            
            logger.log_conversation_sample(0, conversations, max_samples=2)
            
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "cluster_0_samples" in call_args

    def test_context_manager(self):
        """Test context manager functionality."""
        with patch('kura.logger.logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            
            logger = StandardLogger("test")
            
            with logger:
                pass
            
            # Should log run start and end
            assert mock_logger.info.call_count >= 2

    def test_context_manager_with_exception(self):
        """Test context manager with exception."""
        with patch('kura.logger.logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            
            logger = StandardLogger("test")
            
            with pytest.raises(ValueError):
                with logger:
                    raise ValueError("Test error")
            
            # Should log error and end run
            mock_logger.error.assert_called()
            mock_logger.info.assert_called()

    def test_checkpoint_filename(self):
        """Test checkpoint filename property."""
        logger = StandardLogger("test_experiment")
        filename = logger.checkpoint_filename
        assert filename == "logger_config_test_experiment.json"


class TestCreateLogger:
    """Test the create_logger factory function."""

    def test_create_standard_logger(self):
        """Test creating standard logger."""
        logger = create_logger("standard", "test")
        assert isinstance(logger, StandardLogger)
        assert isinstance(logger, BaseClusterLogger)

    def test_create_logger_with_kwargs(self):
        """Test creating logger with additional arguments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = create_logger("standard", "test", artifact_dir=temp_dir)
            assert isinstance(logger, StandardLogger)
            assert logger.supports_artifacts()

    def test_create_logger_invalid_provider(self):
        """Test creating logger with invalid provider."""
        with pytest.raises(ValueError, match="Unsupported logger provider"):
            create_logger("invalid_provider", "test")


class TestBaseClusterLogger:
    """Test the abstract base class behavior."""

    def test_abstract_methods(self):
        """Test that abstract methods cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseClusterLogger()

    def test_default_artifact_logging(self):
        """Test default artifact logging fallback."""
        # Create a concrete subclass for testing
        class TestLogger(BaseClusterLogger):
            def __init__(self):
                self.logged_data = []
            
            def log_params(self, params):
                pass
            
            def log_metrics(self, metrics, step=None):
                pass
            
            def log_errors(self, error, context=None):
                pass
            
            def log(self, data, key, **metadata):
                self.logged_data.append((data, key, metadata))
            
            @property
            def checkpoint_filename(self):
                return "test.json"
        
        logger = TestLogger()
        
        with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
            temp_path = Path(temp_file.name)
            logger.log_artifact(temp_path, "test")
            
            # Should fall back to logging file path
            assert len(logger.logged_data) == 1
            data, key, metadata = logger.logged_data[0]
            assert key == "artifact_info"
            assert str(temp_path) in data