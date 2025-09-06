"""
Multimodal Fashion Recommender Training Package
"""

from config import Config, ModelConfig, DataConfig, TrainingConfig
from model import MultiModalFashionRecommender
from data_loader import StreamingFashionDataset, create_data_loaders
from trainer import FashionRecommenderTrainer
from metrics import RecommendationMetrics, EvaluationManager

__version__ = "1.0.0"
__author__ = "Fashion Recommender Team"

__all__ = [
    "Config",
    "ModelConfig", 
    "DataConfig",
    "TrainingConfig",
    "MultiModalFashionRecommender",
    "StreamingFashionDataset",
    "create_data_loaders",
    "FashionRecommenderTrainer",
    "RecommendationMetrics",
    "EvaluationManager"
]
