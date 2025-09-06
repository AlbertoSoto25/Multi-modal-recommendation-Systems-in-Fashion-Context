"""
Multimodal Fashion Recommender Model
Combines Vision Transformer, BERT, and Cross-Modal Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Tuple, Optional
import math

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for vision-text fusion"""
    
    def __init__(
        self, 
        hidden_size: int = 768, 
        num_heads: int = 12, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len_q, hidden_size]
            key: [batch_size, seq_len_k, hidden_size]  
            value: [batch_size, seq_len_v, hidden_size]
            attention_mask: [batch_size, seq_len_k]
        
        Returns:
            output: [batch_size, seq_len_q, hidden_size]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k = key.size(1)
        
        # Linear projections
        Q = self.q_proj(query)  # [batch_size, seq_len_q, hidden_size]
        K = self.k_proj(key)    # [batch_size, seq_len_k, hidden_size]
        V = self.v_proj(value)  # [batch_size, seq_len_v, hidden_size]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention scores shape
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len_k]
            attention_scores = attention_scores.masked_fill(expanded_mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len_q, head_dim]
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_size
        )
        output = self.out_proj(context)
        
        return output, attention_weights

class CrossModalFusionLayer(nn.Module):
    """Single layer of cross-modal fusion with self-attention and feed-forward"""
    
    def __init__(self, hidden_size: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        
        # Cross-modal attention (vision to text and text to vision)
        self.vision_to_text_attention = CrossModalAttention(hidden_size, num_heads, dropout)
        self.text_to_vision_attention = CrossModalAttention(hidden_size, num_heads, dropout)
        
        # Layer normalization
        self.vision_norm1 = nn.LayerNorm(hidden_size)
        self.text_norm1 = nn.LayerNorm(hidden_size)
        self.vision_norm2 = nn.LayerNorm(hidden_size)
        self.text_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward networks
        self.vision_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(
        self, 
        vision_features: torch.Tensor, 
        text_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_features: [batch_size, num_patches, hidden_size]
            text_features: [batch_size, seq_len, hidden_size]
            text_attention_mask: [batch_size, seq_len]
        
        Returns:
            Enhanced vision and text features
        """
        # Cross-modal attention
        # Vision attends to text
        vision_attended, _ = self.vision_to_text_attention(
            vision_features, text_features, text_features, text_attention_mask
        )
        vision_features = self.vision_norm1(vision_features + vision_attended)
        
        # Text attends to vision  
        text_attended, _ = self.text_to_vision_attention(
            text_features, vision_features, vision_features
        )
        text_features = self.text_norm1(text_features + text_attended)
        
        # Feed-forward networks
        vision_features = self.vision_norm2(vision_features + self.vision_ffn(vision_features))
        text_features = self.text_norm2(text_features + self.text_ffn(text_features))
        
        return vision_features, text_features

class MultiModalFashionRecommender(nn.Module):
    """
    Multimodal Fashion Recommender combining Vision Transformer, BERT, and Cross-Modal Attention
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Vision Transformer
        self.vision_model = AutoModel.from_pretrained(
            config.model.vit_model_name,
            add_pooling_layer=False
        )
        
        # Freeze some layers of ViT
        if config.model.vit_freeze_layers > 0:
            for param in list(self.vision_model.parameters())[:config.model.vit_freeze_layers * 12]:
                param.requires_grad = False
        
        # BERT for text processing
        self.text_model = AutoModel.from_pretrained(
            config.model.bert_model_name,
            add_pooling_layer=False
        )
        
        # Freeze some layers of BERT
        if config.model.bert_freeze_layers > 0:
            for param in list(self.text_model.parameters())[:config.model.bert_freeze_layers * 12]:
                param.requires_grad = False
        
        # Cross-modal fusion layers
        self.cross_modal_layers = nn.ModuleList([
            CrossModalFusionLayer(
                hidden_size=config.model.cross_modal_hidden_size,
                num_heads=config.model.cross_modal_num_heads,
                dropout=config.model.cross_modal_dropout
            )
            for _ in range(config.model.cross_modal_num_layers)
        ])
        
        # Projection layers to align dimensions if needed
        self.vision_projection = nn.Linear(
            config.model.vit_hidden_size, 
            config.model.cross_modal_hidden_size
        ) if config.model.vit_hidden_size != config.model.cross_modal_hidden_size else nn.Identity()
        
        self.text_projection = nn.Linear(
            config.model.bert_hidden_size, 
            config.model.cross_modal_hidden_size
        ) if config.model.bert_hidden_size != config.model.cross_modal_hidden_size else nn.Identity()
        
        # Fusion and output layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.model.cross_modal_hidden_size * 2, config.model.fusion_hidden_size),
            nn.GELU(),
            nn.Dropout(config.model.cross_modal_dropout),
            nn.Linear(config.model.fusion_hidden_size, config.model.output_embedding_size)
        )
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(config.model.temperature))
        
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using Vision Transformer
        
        Args:
            images: [batch_size, channels, height, width]
            
        Returns:
            vision_features: [batch_size, num_patches, hidden_size]
        """
        vision_outputs = self.vision_model(pixel_values=images)
        vision_features = vision_outputs.last_hidden_state  # [batch_size, num_patches+1, hidden_size]
        
        # Remove CLS token, keep only patch embeddings
        vision_features = vision_features[:, 1:, :]  # [batch_size, num_patches, hidden_size]
        
        # Project to cross-modal dimension
        vision_features = self.vision_projection(vision_features)
        
        return vision_features
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode text using BERT
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            text_features: [batch_size, seq_len, hidden_size]
        """
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Project to cross-modal dimension
        text_features = self.text_projection(text_features)
        
        return text_features
    
    def cross_modal_fusion(
        self, 
        vision_features: torch.Tensor, 
        text_features: torch.Tensor,
        text_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention layers
        
        Args:
            vision_features: [batch_size, num_patches, hidden_size]
            text_features: [batch_size, seq_len, hidden_size]
            text_attention_mask: [batch_size, seq_len]
            
        Returns:
            Enhanced vision and text features
        """
        for layer in self.cross_modal_layers:
            vision_features, text_features = layer(
                vision_features, text_features, text_attention_mask
            )
        
        return vision_features, text_features
    
    def forward(
        self, 
        images: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the multimodal model
        
        Args:
            images: [batch_size, channels, height, width]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            item_embeddings: [batch_size, output_embedding_size]
        """
        # Encode modalities
        vision_features = self.encode_image(images)  # [batch_size, num_patches, hidden_size]
        text_features = self.encode_text(input_ids, attention_mask)  # [batch_size, seq_len, hidden_size]
        
        # Cross-modal fusion
        vision_features, text_features = self.cross_modal_fusion(
            vision_features, text_features, attention_mask
        )
        
        # Global pooling
        vision_pooled = vision_features.mean(dim=1)  # [batch_size, hidden_size]
        
        # Text pooling (weighted by attention mask)
        text_mask_expanded = attention_mask.unsqueeze(-1).expand(text_features.size()).float()
        text_pooled = (text_features * text_mask_expanded).sum(dim=1) / text_mask_expanded.sum(dim=1)
        
        # Fusion
        fused_features = torch.cat([vision_pooled, text_pooled], dim=-1)  # [batch_size, hidden_size*2]
        item_embeddings = self.fusion_layer(fused_features)  # [batch_size, output_embedding_size]
        
        # L2 normalize for similarity computation
        item_embeddings = F.normalize(item_embeddings, p=2, dim=-1)
        
        return item_embeddings
    
    def compute_contrastive_loss(
        self, 
        positive_embeddings: torch.Tensor, 
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss for recommendation
        
        Args:
            positive_embeddings: [batch_size, embedding_size]
            negative_embeddings: [batch_size, num_negatives, embedding_size]
            
        Returns:
            loss: scalar tensor
        """
        batch_size, num_negatives = negative_embeddings.size(0), negative_embeddings.size(1)
        
        # InfoNCE Loss Implementation
        # We want to maximize similarity between positive pairs and minimize with negatives
        
        # Normalize embeddings for better stability
        positive_norm = F.normalize(positive_embeddings, p=2, dim=-1)
        negative_norm = F.normalize(negative_embeddings, p=2, dim=-1)
        
        # Compute cosine similarities
        # Positive similarity: for now, we'll use a fixed high target (this is a simplification)
        # In a proper implementation, you'd have actual positive pairs
        pos_sim = torch.ones(batch_size, device=positive_embeddings.device) / self.temperature
        
        # Negative similarities: cosine similarity between positive and each negative
        neg_sim = torch.bmm(
            positive_norm.unsqueeze(1),  # [batch_size, 1, embedding_size]
            negative_norm.transpose(1, 2)  # [batch_size, embedding_size, num_negatives]
        ).squeeze(1) / self.temperature  # [batch_size, num_negatives]
        
        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1 + num_negatives]
        
        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Debug info for troubleshooting
        if torch.isnan(loss):
            print(f"NaN Loss detected!")
            print(f"  pos_sim: {pos_sim}")
            print(f"  neg_sim stats: min={neg_sim.min()}, max={neg_sim.max()}, mean={neg_sim.mean()}")
            print(f"  logits stats: min={logits.min()}, max={logits.max()}, mean={logits.mean()}")
        
        # If loss is exactly zero, it might indicate a problem
        if loss.item() < 1e-8:
            print(f"Very small loss detected: {loss.item()}")
            print(f"  Logits: {logits[0]}")  # Print first sample's logits
        
        return loss
