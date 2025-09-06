"""
Training loop for multimodal fashion recommender system
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import logging
import json
import time
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm
import wandb
from datetime import datetime
import pandas as pd

from model import MultiModalFashionRecommender
from metrics import EvaluationManager
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FashionRecommenderTrainer:
    """Trainer class for the multimodal fashion recommender"""
    
    def __init__(self, config: Config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = MultiModalFashionRecommender(config).to(self.device)
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Evaluation manager
        self.evaluation_manager = EvaluationManager(self.model, self.device, config)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        
        # Initialize wandb if enabled
        if config.training.use_wandb:
            self._init_wandb()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with different learning rates for different components"""
        # Separate parameters for different components
        vision_params = []
        text_params = []
        fusion_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'vision_model' in name:
                vision_params.append(param)
            elif 'text_model' in name:
                text_params.append(param)
            else:
                fusion_params.append(param)
        
        # Different learning rates for different components
        param_groups = [
            {'params': vision_params, 'lr': self.config.training.learning_rate * 0.1},  # Lower LR for pre-trained vision
            {'params': text_params, 'lr': self.config.training.learning_rate * 0.1},   # Lower LR for pre-trained text
            {'params': fusion_params, 'lr': self.config.training.learning_rate}        # Full LR for fusion layers
        ]
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        # Calculate total training steps more conservatively
        # For streaming data, we estimate based on typical batch counts
        estimated_batches_per_epoch = 100  # Conservative estimate
        total_training_steps = self.config.training.max_epochs * estimated_batches_per_epoch
        
        if self.config.training.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=min(self.config.training.warmup_steps, total_training_steps // 10),
                num_training_steps=total_training_steps
            )
        elif self.config.training.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=min(self.config.training.warmup_steps, total_training_steps // 10),
                num_training_steps=total_training_steps
            )
        else:
            scheduler = None
        
        logger.info(f"Scheduler created with {total_training_steps} total steps, {min(self.config.training.warmup_steps, total_training_steps // 10)} warmup steps")
        return scheduler
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        try:
            wandb.init(
                project=self.config.training.project_name,
                config=self.config.to_dict(),
                name=f"multimodal_fashion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            logger.info("Wandb initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.config.training.use_wandb = False
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            if not batch:  # Empty batch from streaming
                logger.warning(f"Empty batch {batch_idx}, skipping...")
                continue
            
            # Debug: Log batch info
            if batch_idx == 0:
                logger.info(f"First batch keys: {list(batch.keys())}")
                logger.info(f"Batch size: {len(batch.get('customer_ids', []))}")
            
            try:
                loss = self.train_step(batch)
                
                if loss > 0:  # Only count non-zero losses
                    total_loss += loss
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': f'{loss:.4f}', 'valid_batches': num_batches})
                else:
                    logger.warning(f"Skipping batch {batch_idx} due to zero loss")
                    
                # Memory cleanup every 10 batches
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Logging
                if self.global_step % self.config.training.log_every_n_steps == 0:
                    self._log_metrics({'train/loss': loss, 'train/step': self.global_step})
                
                # Evaluation
                if self.global_step % self.config.training.eval_every_n_steps == 0:
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics)
                    
                    # Check for best model
                    current_metric = eval_metrics.get('eval/ndcg@10', 0.0)
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self.patience_counter = 0
                        self.save_checkpoint(is_best=True)
                    else:
                        self.patience_counter += 1
                
                # Save checkpoint
                if self.global_step % self.config.training.save_every_n_steps == 0:
                    self.save_checkpoint(is_best=False)
                
                # Early stopping check
                if self.patience_counter >= self.config.training.patience:
                    logger.info(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
                    return {'avg_loss': total_loss / max(num_batches, 1)}
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA out of memory in batch {batch_idx}. Clearing cache and skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    logger.error(f"Runtime error in batch {batch_idx}: {e}")
                    continue
            except Exception as e:
                logger.warning(f"Unexpected error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'avg_loss': avg_loss}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        # Move batch to device
        positive_images = batch['positive_images'].to(self.device)
        positive_text_input_ids = batch['positive_text_input_ids'].to(self.device)
        positive_text_attention_masks = batch['positive_text_attention_masks'].to(self.device)
        
        negative_images = batch['negative_images'].to(self.device)
        negative_text_input_ids = batch['negative_text_input_ids'].to(self.device)
        negative_text_attention_masks = batch['negative_text_attention_masks'].to(self.device)
        
        batch_size, num_negatives = negative_images.size(0), negative_images.size(1)
        
        # Forward pass with mixed precision
        if self.config.mixed_precision:
            with autocast(device_type='cuda'):  # Fix deprecated warning
                logger.debug(f"Processing batch with {batch_size} samples and {num_negatives} negatives per sample")
                
                # Positive embeddings
                try:
                    positive_embeddings = self.model(
                        positive_images, 
                        positive_text_input_ids, 
                        positive_text_attention_masks
                    )
                    logger.debug(f"Positive embeddings shape: {positive_embeddings.shape}")
                except Exception as e:
                    logger.error(f"Error in positive forward pass: {e}")
                    return 0.0
                
                # Negative embeddings - reshape for batch processing
                negative_images_flat = negative_images.view(-1, *negative_images.shape[2:])
                negative_text_input_ids_flat = negative_text_input_ids.view(-1, negative_text_input_ids.size(-1))
                negative_text_attention_masks_flat = negative_text_attention_masks.view(-1, negative_text_attention_masks.size(-1))
                
                # Filter out zero-padded negatives
                valid_mask = negative_text_attention_masks_flat.sum(dim=1) > 0
                valid_count = valid_mask.sum().item()
                logger.debug(f"Valid negatives: {valid_count} out of {len(valid_mask)}")
                
                if valid_count > 0:
                    try:
                        negative_embeddings_flat = self.model(
                            negative_images_flat[valid_mask],
                            negative_text_input_ids_flat[valid_mask],
                            negative_text_attention_masks_flat[valid_mask]
                        )
                        logger.debug(f"Negative embeddings shape: {negative_embeddings_flat.shape}")
                    except Exception as e:
                        logger.error(f"Error in negative forward pass: {e}")
                        return 0.0
                    
                    # Reconstruct negative embeddings tensor
                    negative_embeddings = torch.zeros(
                        batch_size * num_negatives, 
                        positive_embeddings.size(-1),
                        device=self.device
                    )
                    negative_embeddings[valid_mask] = negative_embeddings_flat
                    negative_embeddings = negative_embeddings.view(batch_size, num_negatives, -1)
                else:
                    # Skip this batch if no valid negatives
                    return 0.0
                
                # Compute contrastive loss
                try:
                    loss = self.model.compute_contrastive_loss(positive_embeddings, negative_embeddings)
                    
                    # Debug loss computation
                    if self.global_step % 10 == 0:  # Log every 10 steps
                        logger.info(f"Step {self.global_step} - Loss: {loss.item():.6f}")
                        logger.info(f"  Positive embeddings norm: {positive_embeddings.norm().item():.6f}")
                        logger.info(f"  Negative embeddings norm: {negative_embeddings.norm().item():.6f}")
                        logger.info(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.8f}")
                        
                        # Check if gradients are flowing
                        if hasattr(self.model, 'fusion_layer'):
                            fusion_params = list(self.model.fusion_layer.parameters())
                            if fusion_params and fusion_params[0].grad is not None:
                                logger.info(f"  Fusion layer grad norm: {fusion_params[0].grad.norm().item():.6f}")
                    
                except Exception as e:
                    logger.error(f"Error computing contrastive loss: {e}")
                    return 0.0
        else:
            # Same forward pass without autocast
            positive_embeddings = self.model(
                positive_images, 
                positive_text_input_ids, 
                positive_text_attention_masks
            )
            
            negative_images_flat = negative_images.view(-1, *negative_images.shape[2:])
            negative_text_input_ids_flat = negative_text_input_ids.view(-1, negative_text_input_ids.size(-1))
            negative_text_attention_masks_flat = negative_text_attention_masks.view(-1, negative_text_attention_masks.size(-1))
            
            valid_mask = negative_text_attention_masks_flat.sum(dim=1) > 0
            if valid_mask.sum() > 0:
                negative_embeddings_flat = self.model(
                    negative_images_flat[valid_mask],
                    negative_text_input_ids_flat[valid_mask],
                    negative_text_attention_masks_flat[valid_mask]
                )
                
                negative_embeddings = torch.zeros(
                    batch_size * num_negatives, 
                    positive_embeddings.size(-1),
                    device=self.device
                )
                negative_embeddings[valid_mask] = negative_embeddings_flat
                negative_embeddings = negative_embeddings.view(batch_size, num_negatives, -1)
            else:
                return 0.0
            
            loss = self.model.compute_contrastive_loss(positive_embeddings, negative_embeddings)
        
        # Backward pass
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.training.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        self.global_step += 1
        
        return loss.item()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model"""
        logger.info("Starting evaluation...")
        
        # Load articles for embedding computation
        articles_path = os.path.join(self.config.data.data_dir, "articles_clean.csv")
        try:
            articles_df = pd.read_csv(articles_path)
            
            # Compute article embeddings (cached for efficiency)
            if not hasattr(self, '_article_embeddings_cache'):
                self._article_embeddings_cache = self.evaluation_manager.compute_article_embeddings(
                    articles_df, self.config.data.images_dir
                )
            
            # Evaluate model
            metrics = self.evaluation_manager.evaluate_model(
                self.test_loader, 
                self._article_embeddings_cache
            )
            
            # Add eval prefix
            eval_metrics = {f'eval/{k}': v for k, v in metrics.items()}
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            eval_metrics = {'eval/error': 1.0}
        
        # Switch back to training mode
        self.model.train()
        
        return eval_metrics
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb and console"""
        # Console logging
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")
        
        # Wandb logging
        if self.config.training.use_wandb:
            try:
                wandb.log(metrics, step=self.global_step)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir, 
            f'checkpoint_epoch_{self.current_epoch}_step_{self.global_step}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.training.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        logger.info(f"Checkpoint loaded: epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.training.max_epochs):
            self.current_epoch = epoch
            logger.info(f"Starting epoch {epoch}")
            
            # Train for one epoch
            epoch_metrics = self.train_epoch()
            
            # Log epoch metrics
            epoch_metrics.update({
                'epoch': epoch,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            self._log_metrics(epoch_metrics)
            
            # Early stopping check
            if self.patience_counter >= self.config.training.patience:
                logger.info("Early stopping triggered")
                break
        
        # Final evaluation
        logger.info("Training completed. Running final evaluation...")
        final_metrics = self.evaluate()
        self._log_metrics(final_metrics)
        
        # Save final model
        self.save_checkpoint(is_best=False)
        
        if self.config.training.use_wandb:
            wandb.finish()
        
        logger.info("Training finished!")
        return final_metrics
