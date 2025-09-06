"""
Configuration file for the multimodal fashion recommender system
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    # Vision Transformer config
    vit_model_name: str = "google/vit-base-patch16-224"
    vit_hidden_size: int = 768
    vit_freeze_layers: int = 6  # Number of layers to freeze
    
    # BERT config
    bert_model_name: str = "bert-base-uncased"
    bert_hidden_size: int = 768
    bert_freeze_layers: int = 6  # Number of layers to freeze
    
    # Cross-modal attention config
    cross_modal_hidden_size: int = 768
    cross_modal_num_heads: int = 12
    cross_modal_num_layers: int = 4
    cross_modal_dropout: float = 0.1
    
    # Fusion and output config
    fusion_hidden_size: int = 512
    output_embedding_size: int = 256
    
    # Temperature for contrastive learning
    temperature: float = 0.07

@dataclass
class DataConfig:
    """Configuration for data processing"""
    # Paths (optimized for Colab with Google Drive access)
    data_dir: str = "/content/data"  # CSV files copied locally for fast access
    images_dir: str = "/content/drive/MyDrive/MUIIA/tfm/images_small"  # Images accessed directly from Drive
    
    # Image processing
    image_size: int = 224
    image_channels: int = 3
    
    # Text processing
    max_text_length: int = 128
    
    # Streaming config
    batch_size: int = 32
    streaming_buffer_size: int = 1000
    num_workers: int = 2
    
    # Data split
    train_start_date: str = "2018-09-20"
    train_end_date: str = "2020-09-15"  # 2 years for training
    test_start_date: str = "2020-09-16"
    test_end_date: str = "2020-09-22"   # 7 days for testing
    
    # Negative sampling
    num_negative_samples: int = 5

@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 5
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Scheduling
    scheduler_type: str = "cosine"
    
    # Checkpointing
    save_every_n_steps: int = 1000
    checkpoint_dir: str = "/content/checkpoints"
    
    # Evaluation
    eval_every_n_steps: int = 500
    eval_batch_size: int = 64
    
    # Early stopping
    patience: int = 3
    min_delta: float = 0.001
    
    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = True
    project_name: str = "fashion-multimodal-recommender"

@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Device config
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    mixed_precision: bool = True
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed
        }
