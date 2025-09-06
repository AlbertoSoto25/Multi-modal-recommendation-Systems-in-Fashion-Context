"""
Google Colab Setup Script for Multimodal Fashion Recommender
Run this in the first cell of your Colab notebook
"""

import os
import sys
import subprocess
import logging

def install_requirements():
    """Install required packages"""
    packages = [
        'torch>=2.0.0',
        'torchvision>=0.15.0', 
        'transformers>=4.25.0',
        'datasets>=2.8.0',
        'pillow>=9.0.0',
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.2.0',
        'tqdm>=4.64.0',
        'wandb>=0.13.0',
        'accelerate>=0.16.0',
        'einops>=0.6.0',
        'matplotlib>=3.6.0',
        'seaborn>=0.12.0',
        'psutil'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")

def setup_directories():
    """Setup required directories"""
    directories = [
        '/content/checkpoints',
        '/content/logs',
        '/content/outputs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_data():
    """Download and setup data (customize this for your data source)"""
    print("Setting up data...")
    
    # Example for downloading from Google Drive
    # You'll need to replace this with your actual data source
    print("Please upload your data to /content/cleaned-data/")
    print("Required files:")
    print("  - transactions_clean.csv")
    print("  - articles_clean.csv")
    print("  - customers_clean.csv")
    print("  - normalized_images/ (directory with image subdirectories)")
    
    return True

def check_gpu():
    """Check GPU availability"""
    import torch
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Available: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        
        # Recommendations based on GPU
        if gpu_memory < 12:
            print("⚠️  T4 GPU detected. Consider using smaller batch sizes.")
            recommended_batch_size = 16
        elif gpu_memory < 16:
            print("✅ P100/V100 GPU detected. Good for training.")
            recommended_batch_size = 24
        else:
            print("🚀 A100 GPU detected. Excellent for training!")
            recommended_batch_size = 32
            
        print(f"Recommended batch size: {recommended_batch_size}")
        
    else:
        print("❌ No GPU available. Training will be very slow.")

def setup_wandb():
    """Setup Weights & Biases"""
    try:
        import wandb
        print("Weights & Biases is available.")
        print("Run 'wandb login' if you want to use experiment tracking.")
        return True
    except ImportError:
        print("Installing wandb...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        return True

def main():
    """Main setup function"""
    print("🚀 Setting up Multimodal Fashion Recommender in Google Colab")
    print("=" * 60)
    
    # Install requirements
    print("\n📦 Installing requirements...")
    install_requirements()
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Check GPU
    print("\n🎮 Checking GPU...")
    check_gpu()
    
    # Setup wandb
    print("\n📊 Setting up Weights & Biases...")
    setup_wandb()
    
    # Data setup
    print("\n💾 Data setup...")
    download_data()
    
    print("\n✅ Setup completed!")
    print("\nNext steps:")
    print("1. Upload your data to /content/cleaned-data/")
    print("2. Run the training script with: python main.py")
    print("3. Monitor training progress in the logs")
    
    # Create a sample configuration
    sample_config = '''
# Sample configuration for Google Colab
{
    "model": {
        "vit_model_name": "google/vit-base-patch16-224",
        "bert_model_name": "bert-base-uncased",
        "cross_modal_num_layers": 2,
        "temperature": 0.07
    },
    "data": {
        "data_dir": "/content/cleaned-data",
        "images_dir": "/content/cleaned-data/normalized_images", 
        "batch_size": 16,
        "streaming_buffer_size": 500
    },
    "training": {
        "learning_rate": 5e-5,
        "max_epochs": 3,
        "patience": 2,
        "use_wandb": true
    }
}
    '''
    
    with open('/content/sample_config.json', 'w') as f:
        f.write(sample_config)
    
    print("\n📋 Sample configuration saved to /content/sample_config.json")

if __name__ == "__main__":
    main()
