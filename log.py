#!/usr/bin/env python3
"""
Simple demo script showing the Kura logging system in action.
"""

from kura.logger import StandardLogger

def main():
    # Create a logger
    logger = StandardLogger("demo_experiment")
    
    print("Running Kura logging demo...")
    
    # Test standard logging methods
    logger.info("Starting demo experiment")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    
    # Test clustering-specific convenience methods
    logger.log_params({
        "n_clusters": 8,
        "algorithm": "kmeans",
        "embedding_model": "text-embedding-3-large"
    })
    
    logger.log_metrics({
        "silhouette_score": 0.42,
        "inertia": 2847.3,
        "processing_time": 45.2
    })
    
    # Log with step tracking
    for i in range(3):
        logger.log_metrics({"epoch_loss": 0.8 - i * 0.1}, step=i)
    
    # Generic log method
    logger.log("Processing completed successfully", "status")
    logger.log({"progress": 100, "items_processed": 1500}, "final_stats", 
               algorithm="kmeans", timestamp="2024-01-15T10:30:00")
    
    # Demonstrate error logging
    try:
        # Simulate an error
        raise ValueError("This is a demo error")
    except Exception as e:
        logger.log_errors(e, {"operation": "clustering", "data_size": 1500})
    
    # Log a string error
    logger.log_errors("Configuration file not found", {"config_path": "/tmp/config.json"})
    
    # Test direct error method
    logger.error("Direct error message", exc_info=False)
    
    logger.info("Demo completed! Check the console output above.")

if __name__ == "__main__":
    main()
