"""
Streaming data loader for multimodal fashion recommendation system
Designed to handle large datasets efficiently in Google Colab
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from PIL import Image
import numpy as np
from typing import Iterator, Tuple, Dict, Any, Optional, List
import random
from datetime import datetime, timedelta
import json
from transformers import AutoTokenizer
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamingFashionDataset(IterableDataset):
    """
    Streaming dataset that loads data in chunks to avoid memory issues
    """
    
    def __init__(
        self,
        transactions_path: str,
        articles_path: str,
        customers_path: str,
        images_dir: str,
        tokenizer_name: str = "bert-base-uncased",
        image_size: int = 224,
        max_text_length: int = 128,
        chunk_size: int = 10000,
        num_negative_samples: int = 5,
        start_date: str = None,
        end_date: str = None,
        mode: str = "train"
    ):
        self.transactions_path = transactions_path
        self.articles_path = articles_path
        self.customers_path = customers_path
        self.images_dir = images_dir
        self.chunk_size = chunk_size
        self.num_negative_samples = num_negative_samples
        self.start_date = start_date
        self.end_date = end_date
        self.mode = mode
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.max_text_length = max_text_length
        
        # Load articles metadata (small enough to keep in memory)
        logger.info("Loading articles metadata...")
        self.articles_df = pd.read_csv(articles_path)
        # Ensure article IDs are zero-padded to 10 digits for consistency
        self.articles_df['article_id'] = self.articles_df['article_id'].astype(str).str.zfill(10)
        self.article_to_idx = {str(article_id): idx for idx, article_id in enumerate(self.articles_df['article_id'].unique())}
        self.idx_to_article = {idx: str(article_id) for article_id, idx in self.article_to_idx.items()}
        
        # Create article text descriptions
        self.article_texts = self._create_article_texts()
        
        # Get all article IDs for negative sampling
        self.all_articles = list(self.article_to_idx.keys())
        
        logger.info(f"Loaded {len(self.articles_df)} articles")
        logger.info(f"Created text descriptions for {len(self.article_texts)} articles")
        logger.info(f"Sample articles: {list(self.article_texts.keys())[:5]}")
        
    def _create_article_texts(self) -> Dict[str, str]:
        """Create text descriptions for articles"""
        article_texts = {}
        
        for _, row in self.articles_df.iterrows():
            # Article ID is already zero-padded from the DataFrame processing
            article_id = str(row['article_id'])
            
            # Combine multiple text fields
            text_parts = []
            
            if pd.notna(row.get('prod_name')):
                text_parts.append(str(row['prod_name']))
                
            if pd.notna(row.get('product_type_name')):
                text_parts.append(str(row['product_type_name']))
                
            if pd.notna(row.get('product_group_name')):
                text_parts.append(str(row['product_group_name']))
                
            if pd.notna(row.get('colour_group_name')):
                text_parts.append(str(row['colour_group_name']))
                
            if pd.notna(row.get('department_name')):
                text_parts.append(str(row['department_name']))
                
            if pd.notna(row.get('detail_desc')):
                text_parts.append(str(row['detail_desc']))
            
            article_texts[article_id] = " ".join(text_parts)
            
        return article_texts
    
    def _load_image(self, article_id: str) -> Optional[torch.Tensor]:
        """Load and preprocess image"""
        try:
            # Images are now stored in a single directory without subfolders
            # Article ID should be zero-padded to 10 digits
            article_id_padded = str(article_id).zfill(10)
            image_path = os.path.join(self.images_dir, f"{article_id_padded}.jpg")
            
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                return self.image_transform(image)
            else:
                # Return a black image if not found
                logger.debug(f"Image not found for article {article_id_padded}: {image_path}")
                return torch.zeros(3, 224, 224)
                
        except Exception as e:
            logger.warning(f"Error loading image for article {article_id}: {e}")
            return torch.zeros(3, 224, 224)
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text description"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    def _get_negative_samples(self, positive_article: str, customer_articles: set) -> List[str]:
        """Get negative samples for contrastive learning"""
        negative_articles = []
        available_articles = [a for a in self.all_articles if a not in customer_articles]
        
        if len(available_articles) < self.num_negative_samples:
            # If not enough articles, sample with replacement
            negative_articles = random.choices(available_articles, k=self.num_negative_samples)
        else:
            negative_articles = random.sample(available_articles, self.num_negative_samples)
            
        return negative_articles
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through the dataset in chunks"""
        logger.info(f"Starting {self.mode} data iteration...")
        logger.info(f"Reading from: {self.transactions_path}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Chunk size: {self.chunk_size}")
        
        # Read transactions in chunks
        try:
            chunk_iter = pd.read_csv(
                self.transactions_path, 
                chunksize=self.chunk_size,
                parse_dates=['t_dat']
            )
        except Exception as e:
            logger.error(f"Failed to read transactions file: {e}")
            return
        
        total_examples_yielded = 0
        for chunk_idx, chunk in enumerate(chunk_iter):
            logger.info(f"Loaded chunk {chunk_idx + 1} with {len(chunk)} transactions")
            
            # Filter by date range if specified
            if self.start_date and self.end_date:
                original_size = len(chunk)
                chunk = chunk[
                    (chunk['t_dat'] >= self.start_date) & 
                    (chunk['t_dat'] <= self.end_date)
                ]
                logger.info(f"After date filtering: {len(chunk)} transactions (was {original_size})")
            
            if len(chunk) == 0:
                logger.warning(f"Chunk {chunk_idx + 1} is empty after filtering")
                continue
                
            logger.info(f"Processing chunk {chunk_idx + 1} with {len(chunk)} transactions")
            
            # Group by customer to get purchase history
            customer_groups = chunk.groupby('customer_id')
            
            examples_in_chunk = 0
            skipped_no_text = 0
            skipped_no_image = 0
            skipped_no_negatives = 0
            
            for customer_id, customer_data in customer_groups:
                customer_articles = set(customer_data['article_id'].astype(str).str.zfill(10))
                
                # For each positive article, create training examples
                for _, transaction in customer_data.iterrows():
                    # Ensure article_id is zero-padded to 10 digits
                    article_id = str(transaction['article_id']).zfill(10)
                    
                    # Skip if we don't have text for this article
                    if article_id not in self.article_texts:
                        skipped_no_text += 1
                        continue
                    
                    # Load positive example
                    positive_image = self._load_image(article_id)
                    positive_text = self._tokenize_text(self.article_texts[article_id])
                    
                    if positive_image is None:
                        skipped_no_image += 1
                        continue
                    
                    # Get negative samples
                    negative_articles = self._get_negative_samples(article_id, customer_articles)
                    
                    # Create training example
                    example = {
                        'customer_id': customer_id,
                        'positive_article_id': article_id,
                        'positive_image': positive_image,
                        'positive_text_input_ids': positive_text['input_ids'],
                        'positive_text_attention_masks': positive_text['attention_mask'],
                        'negative_articles': [],
                        'negative_images': [],
                        'negative_text_input_ids': [],
                        'negative_text_attention_masks': []
                    }
                    
                    # Add negative examples
                    for neg_article_id in negative_articles:
                        if neg_article_id in self.article_texts:
                            neg_image = self._load_image(neg_article_id)
                            neg_text = self._tokenize_text(self.article_texts[neg_article_id])
                            
                            if neg_image is not None:
                                example['negative_articles'].append(neg_article_id)
                                example['negative_images'].append(neg_image)
                                example['negative_text_input_ids'].append(neg_text['input_ids'])
                                example['negative_text_attention_masks'].append(neg_text['attention_mask'])
                    
                    # Only yield if we have at least one negative sample
                    if len(example['negative_articles']) > 0:
                        # Convert lists to tensors
                        if example['negative_images']:
                            example['negative_images'] = torch.stack(example['negative_images'])
                        if example['negative_text_input_ids']:
                            example['negative_text_input_ids'] = torch.stack(example['negative_text_input_ids'])
                        if example['negative_text_attention_masks']:
                            example['negative_text_attention_masks'] = torch.stack(example['negative_text_attention_masks'])
                        
                        total_examples_yielded += 1
                        examples_in_chunk += 1
                        yield example
                    else:
                        skipped_no_negatives += 1
            
            # Log chunk statistics
            logger.info(f"Chunk {chunk_idx + 1} statistics:")
            logger.info(f"  Examples created: {examples_in_chunk}")
            logger.info(f"  Skipped (no text): {skipped_no_text}")
            logger.info(f"  Skipped (no image): {skipped_no_image}")
            logger.info(f"  Skipped (no negatives): {skipped_no_negatives}")
        
        logger.info(f"Total examples yielded for {self.mode}: {total_examples_yielded}")

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    if not batch:
        return {}
    
    # Extract all the components
    positive_images = torch.stack([item['positive_image'] for item in batch])
    positive_text_input_ids = torch.stack([item['positive_text_input_ids'] for item in batch])
    positive_text_attention_masks = torch.stack([item['positive_text_attention_masks'] for item in batch])
    
    # Handle negative samples - pad to same length
    max_negatives = max(len(item['negative_articles']) for item in batch)
    
    batch_size = len(batch)
    negative_images = torch.zeros(batch_size, max_negatives, 3, 224, 224)
    negative_text_input_ids = torch.zeros(batch_size, max_negatives, positive_text_input_ids.size(-1), dtype=torch.long)
    negative_text_attention_masks = torch.zeros(batch_size, max_negatives, positive_text_attention_masks.size(-1), dtype=torch.long)
    
    for i, item in enumerate(batch):
        n_negs = len(item['negative_articles'])
        if n_negs > 0:
            negative_images[i, :n_negs] = item['negative_images']
            negative_text_input_ids[i, :n_negs] = item['negative_text_input_ids']
            negative_text_attention_masks[i, :n_negs] = item['negative_text_attention_masks']
    
    return {
        'positive_images': positive_images,
        'positive_text_input_ids': positive_text_input_ids,
        'positive_text_attention_masks': positive_text_attention_masks,
        'negative_images': negative_images,
        'negative_text_input_ids': negative_text_input_ids,
        'negative_text_attention_masks': negative_text_attention_masks,
        'customer_ids': [item['customer_id'] for item in batch],
        'positive_article_ids': [item['positive_article_id'] for item in batch]
    }

def create_data_loaders(config) -> Tuple[DataLoader, DataLoader]:
    """Create train and test data loaders"""
    
    # Use transactions_clean_small.csv for training as requested
    transactions_file = "transactions_clean_small.csv"
    transactions_path = os.path.join(config.data.data_dir, transactions_file)
    
    # Fallback to regular file if small file doesn't exist
    if not os.path.exists(transactions_path):
        transactions_file = "transactions_clean.csv"
        transactions_path = os.path.join(config.data.data_dir, transactions_file)
        logger.info(f"Small transactions file not found, using: {transactions_file}")
    else:
        logger.info(f"Using small transactions file: {transactions_file}")
    
    # Training dataset
    train_dataset = StreamingFashionDataset(
        transactions_path=transactions_path,
        articles_path=os.path.join(config.data.data_dir, "articles_clean.csv"),
        customers_path=os.path.join(config.data.data_dir, "customers_clean.csv"),
        images_dir=config.data.images_dir,
        tokenizer_name=config.model.bert_model_name,
        image_size=config.data.image_size,
        max_text_length=config.data.max_text_length,
        chunk_size=config.data.streaming_buffer_size,
        num_negative_samples=config.data.num_negative_samples,
        start_date=config.data.train_start_date,
        end_date=config.data.train_end_date,
        mode="train"
    )
    
    # Test dataset
    test_dataset = StreamingFashionDataset(
        transactions_path=os.path.join(config.data.data_dir, "transactions_clean.csv"),
        articles_path=os.path.join(config.data.data_dir, "articles_clean.csv"),
        customers_path=os.path.join(config.data.data_dir, "customers_clean.csv"),
        images_dir=config.data.images_dir,
        tokenizer_name=config.model.bert_model_name,
        image_size=config.data.image_size,
        max_text_length=config.data.max_text_length,
        chunk_size=config.data.streaming_buffer_size,
        num_negative_samples=config.data.num_negative_samples,
        start_date=config.data.test_start_date,
        end_date=config.data.test_end_date,
        mode="test"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        num_workers=0,  # Set to 0 for Colab compatibility
        collate_fn=collate_fn,
        pin_memory=True if config.device == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.eval_batch_size,
        num_workers=0,  # Set to 0 for Colab compatibility
        collate_fn=collate_fn,
        pin_memory=True if config.device == "cuda" else False
    )
    
    return train_loader, test_loader
