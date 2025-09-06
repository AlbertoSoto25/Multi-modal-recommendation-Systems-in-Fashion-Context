"""
Main training script for Multimodal Fashion Recommender System
Optimized for Google Colab environment
"""
import os
import sys
import torch
import random
import numpy as np
import logging
from datetime import datetime
import argparse
import json
import gc
import psutil

# Import our modules
from config import Config
from data_loader import create_data_loaders
from trainer import FashionRecommenderTrainer
from model import MultiModalFashionRecommender

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_gpu_memory():
    """Check GPU memory status"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        logger.info(f"GPU Total Memory: {gpu_memory:.2f} GB")
        logger.info(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
        logger.info(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} GB")
        logger.info(f"GPU Memory Available: {gpu_memory - gpu_memory_reserved:.2f} GB")
    
    # RAM memory
    ram = psutil.virtual_memory()
    logger.info(f"RAM Total: {ram.total / 1024**3:.2f} GB")
    logger.info(f"RAM Available: {ram.available / 1024**3:.2f} GB")
    logger.info(f"RAM Used: {ram.used / 1024**3:.2f} GB")

def setup_colab_environment():
    """Setup Google Colab environment"""
    logger.info("Setting up Google Colab environment...")
    
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        logger.info("Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        logger.info("Not running in Google Colab")
    
    # Mount Google Drive if in Colab
    if IN_COLAB:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("Google Drive mounted successfully")
        except Exception as e:
            logger.warning(f"Failed to mount Google Drive: {e}")
    
    # Install required packages if needed
    required_packages = [
        'transformers', 'datasets', 'wandb', 'accelerate', 'einops'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            logger.info(f"Installing {package}...")
            os.system(f"pip install {package}")
    
    return IN_COLAB

def download_data_if_needed(config):
    """Download or setup data if needed"""
    data_dir = config.data.data_dir
    
    if not os.path.exists(data_dir):
        logger.info(f"Data directory {data_dir} not found")
        
        # In a real scenario, you would download the data here
        # For now, we assume the data is already available
        logger.warning("Please ensure the data is available in the specified directory")
        return False
    
    # Check required files
    required_files = [
        "transactions_clean_small.csv",
        "articles_clean.csv", 
        "customers_clean.csv"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    # Check images directory
    if not os.path.exists(config.data.images_dir):
        logger.error(f"Images directory not found: {config.data.images_dir}")
        return False
    
    logger.info("All required data files found")
    return True

def optimize_for_colab(config):
    """Optimize configuration for Google Colab constraints"""
    logger.info("Optimizing configuration for Google Colab...")
    
    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Available GPU memory: {gpu_memory:.2f} GB")
        
        # Adjust batch size based on GPU memory
        if gpu_memory < 12:  # Less than 12GB (T4)
            config.data.batch_size = min(config.data.batch_size, 16)
            config.training.eval_batch_size = min(config.training.eval_batch_size, 32)
            config.data.streaming_buffer_size = min(config.data.streaming_buffer_size, 500)
            config.training.gradient_accumulation_steps = max(config.training.gradient_accumulation_steps, 8)
            logger.info("Adjusted for T4 GPU (< 12GB)")
            
        elif gpu_memory < 16:  # 12-16GB (P100, V100)
            config.data.batch_size = min(config.data.batch_size, 24)
            config.training.eval_batch_size = min(config.training.eval_batch_size, 48)
            config.data.streaming_buffer_size = min(config.data.streaming_buffer_size, 750)
            config.training.gradient_accumulation_steps = max(config.training.gradient_accumulation_steps, 6)
            logger.info("Adjusted for P100/V100 GPU (12-16GB)")
            
        else:  # > 16GB (A100)
            logger.info("High-end GPU detected, using default settings")
    
    else:
        logger.warning("No GPU available, training will be very slow")
        config.device = "cpu"
        config.mixed_precision = False
        config.data.batch_size = min(config.data.batch_size, 8)
        config.training.eval_batch_size = min(config.training.eval_batch_size, 16)
    
    # Adjust other settings for Colab
    config.data.num_workers = 0  # Colab doesn't handle multiprocessing well
    config.training.save_every_n_steps = max(config.training.save_every_n_steps, 2000)  # Save less frequently
    config.training.eval_every_n_steps = max(config.training.eval_every_n_steps, 1000)  # Evaluate less frequently
    
    logger.info(f"Final configuration:")
    logger.info(f"  Batch size: {config.data.batch_size}")
    logger.info(f"  Eval batch size: {config.training.eval_batch_size}")
    logger.info(f"  Streaming buffer size: {config.data.streaming_buffer_size}")
    logger.info(f"  Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
    logger.info(f"  Mixed precision: {config.mixed_precision}")
    
    return config

def clear_memory():
    """Clear memory to prevent OOM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Multimodal Fashion Recommender')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/MUIIA/tfm', help='Data directory')
    parser.add_argument('--images_dir', type=str, default='/content/drive/MyDrive/MUIIA/tfm/images', help='Images directory')
    parser.add_argument('--checkpoint_dir', type=str, default='/content/checkpoints', help='Checkpoint directory')
    parser.add_argument('--wandb_project', type=str, default='fashion-multimodal-recommender', help='Wandb project name')
    parser.add_argument('--max_epochs', type=int, default=5, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Setup environment
    logger.info("Starting Multimodal Fashion Recommender Training")
    logger.info(f"Arguments: {args}")
    
    IN_COLAB = setup_colab_environment()
    check_gpu_memory()
    
    # Load configuration
    config = Config()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        # Override config values with loaded config
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        # Override top-level config values
        for key in ['device', 'mixed_precision', 'seed']:
            if key in config_dict and hasattr(config, key):
                setattr(config, key, config_dict[key])
    
    # Override config with command line arguments
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.images_dir:
        config.data.images_dir = args.images_dir
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir
    if args.wandb_project:
        config.training.project_name = args.wandb_project
    if args.max_epochs:
        config.training.max_epochs = args.max_epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.no_wandb:
        config.training.use_wandb = False
    
    # Optimize for Colab
    config = optimize_for_colab(config)
    
    # Set random seed
    set_seed(config.seed)
    
    # Check data availability
    if not download_data_if_needed(config):
        logger.error("Data setup failed. Please check data availability.")
        return
    
    # Save config
    config_path = os.path.join(config.training.checkpoint_dir, 'config.json')
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Configuration saved to {config_path}")
    
    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, test_loader = create_data_loaders(config)
        logger.info("Data loaders created successfully")
        
        clear_memory()  # Clear memory before model creation
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = FashionRecommenderTrainer(config, train_loader, test_loader)
        logger.info("Trainer initialized successfully")
        
        # Resume from checkpoint if specified
        if args.resume and os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
        
        # Start training
        logger.info("Starting training process...")
        final_metrics = trainer.train()
        
        # Log final results
        logger.info("Training completed successfully!")
        logger.info("Final metrics:")
        for metric_name, value in final_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Save final results
        results_path = os.path.join(config.training.checkpoint_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        logger.info(f"Final results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    finally:
        # Cleanup
        clear_memory()
        logger.info("Training script completed")

if __name__ == "__main__":
    main()
