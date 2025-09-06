"""
Evaluation metrics for fashion recommendation system
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import pandas as pd
from collections import defaultdict
import logging
from sklearn.metrics import ndcg_score
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationMetrics:
    """Class to compute various recommendation metrics"""
    
    def __init__(self, k_values: List[int] = [5, 10, 20, 50]):
        self.k_values = k_values
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.ground_truths = []
        self.customer_ids = []
        
    def update(
        self, 
        customer_ids: List[str], 
        predictions: torch.Tensor, 
        ground_truths: List[List[str]]
    ):
        """
        Update metrics with new batch
        
        Args:
            customer_ids: List of customer IDs
            predictions: [batch_size, num_items] similarity scores
            ground_truths: List of lists of ground truth article IDs for each customer
        """
        self.customer_ids.extend(customer_ids)
        self.predictions.extend(predictions.cpu().numpy())
        self.ground_truths.extend(ground_truths)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all recommendation metrics"""
        if not self.predictions:
            return {}
        
        metrics = {}
        
        # Convert predictions to rankings
        rankings = []
        for pred in self.predictions:
            # Get top-k items (indices)
            ranking = np.argsort(pred)[::-1]  # Sort in descending order
            rankings.append(ranking)
        
        # Compute metrics for each k
        for k in self.k_values:
            # Precision@K
            precision_k = self._compute_precision_at_k(rankings, k)
            metrics[f'precision@{k}'] = precision_k
            
            # Recall@K
            recall_k = self._compute_recall_at_k(rankings, k)
            metrics[f'recall@{k}'] = recall_k
            
            # F1@K
            if precision_k + recall_k > 0:
                f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                f1_k = 0.0
            metrics[f'f1@{k}'] = f1_k
            
            # NDCG@K
            ndcg_k = self._compute_ndcg_at_k(rankings, k)
            metrics[f'ndcg@{k}'] = ndcg_k
            
            # Hit Rate@K
            hit_rate_k = self._compute_hit_rate_at_k(rankings, k)
            metrics[f'hit_rate@{k}'] = hit_rate_k
        
        # Mean Reciprocal Rank
        mrr = self._compute_mrr(rankings)
        metrics['mrr'] = mrr
        
        # Mean Average Precision
        map_score = self._compute_map(rankings)
        metrics['map'] = map_score
        
        # Coverage metrics
        coverage = self._compute_coverage(rankings)
        metrics['coverage'] = coverage
        
        # Diversity metrics
        diversity = self._compute_diversity(rankings)
        metrics['diversity'] = diversity
        
        return metrics
    
    def _compute_precision_at_k(self, rankings: List[np.ndarray], k: int) -> float:
        """Compute Precision@K"""
        precisions = []
        
        for i, ranking in enumerate(rankings):
            top_k = set(ranking[:k])
            ground_truth = set(self.ground_truths[i])
            
            if len(top_k) == 0:
                precision = 0.0
            else:
                precision = len(top_k.intersection(ground_truth)) / len(top_k)
            
            precisions.append(precision)
        
        return np.mean(precisions)
    
    def _compute_recall_at_k(self, rankings: List[np.ndarray], k: int) -> float:
        """Compute Recall@K"""
        recalls = []
        
        for i, ranking in enumerate(rankings):
            top_k = set(ranking[:k])
            ground_truth = set(self.ground_truths[i])
            
            if len(ground_truth) == 0:
                recall = 0.0
            else:
                recall = len(top_k.intersection(ground_truth)) / len(ground_truth)
            
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def _compute_ndcg_at_k(self, rankings: List[np.ndarray], k: int) -> float:
        """Compute NDCG@K"""
        ndcgs = []
        
        for i, ranking in enumerate(rankings):
            top_k = ranking[:k]
            ground_truth = set(self.ground_truths[i])
            
            # Create relevance scores (1 for relevant, 0 for not relevant)
            relevance_scores = [1 if item in ground_truth else 0 for item in top_k]
            
            if sum(relevance_scores) == 0:
                ndcg = 0.0
            else:
                # Compute DCG
                dcg = relevance_scores[0]
                for j in range(1, len(relevance_scores)):
                    dcg += relevance_scores[j] / math.log2(j + 1)
                
                # Compute IDCG (ideal DCG)
                ideal_relevance = sorted(relevance_scores, reverse=True)
                idcg = ideal_relevance[0]
                for j in range(1, len(ideal_relevance)):
                    idcg += ideal_relevance[j] / math.log2(j + 1)
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
            
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs)
    
    def _compute_hit_rate_at_k(self, rankings: List[np.ndarray], k: int) -> float:
        """Compute Hit Rate@K"""
        hits = []
        
        for i, ranking in enumerate(rankings):
            top_k = set(ranking[:k])
            ground_truth = set(self.ground_truths[i])
            
            hit = 1.0 if len(top_k.intersection(ground_truth)) > 0 else 0.0
            hits.append(hit)
        
        return np.mean(hits)
    
    def _compute_mrr(self, rankings: List[np.ndarray]) -> float:
        """Compute Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for i, ranking in enumerate(rankings):
            ground_truth = set(self.ground_truths[i])
            
            reciprocal_rank = 0.0
            for rank, item in enumerate(ranking, 1):
                if item in ground_truth:
                    reciprocal_rank = 1.0 / rank
                    break
            
            reciprocal_ranks.append(reciprocal_rank)
        
        return np.mean(reciprocal_ranks)
    
    def _compute_map(self, rankings: List[np.ndarray]) -> float:
        """Compute Mean Average Precision"""
        average_precisions = []
        
        for i, ranking in enumerate(rankings):
            ground_truth = set(self.ground_truths[i])
            
            if len(ground_truth) == 0:
                average_precisions.append(0.0)
                continue
            
            relevant_found = 0
            precision_sum = 0.0
            
            for rank, item in enumerate(ranking, 1):
                if item in ground_truth:
                    relevant_found += 1
                    precision_at_rank = relevant_found / rank
                    precision_sum += precision_at_rank
            
            if relevant_found > 0:
                average_precision = precision_sum / len(ground_truth)
            else:
                average_precision = 0.0
            
            average_precisions.append(average_precision)
        
        return np.mean(average_precisions)
    
    def _compute_coverage(self, rankings: List[np.ndarray], k: int = 50) -> float:
        """Compute catalog coverage"""
        all_recommended = set()
        
        for ranking in rankings:
            top_k = ranking[:k]
            all_recommended.update(top_k)
        
        # Assuming we know the total number of items in catalog
        # This should be passed as a parameter in a real implementation
        total_items = max(max(ranking) for ranking in rankings) + 1
        
        coverage = len(all_recommended) / total_items
        return coverage
    
    def _compute_diversity(self, rankings: List[np.ndarray], k: int = 20) -> float:
        """Compute intra-list diversity (simplified version)"""
        # This is a simplified diversity measure
        # In practice, you'd want to compute semantic diversity using item features
        diversities = []
        
        for ranking in rankings:
            top_k = ranking[:k]
            
            # Simple diversity: ratio of unique items to total items
            diversity = len(set(top_k)) / len(top_k) if len(top_k) > 0 else 0.0
            diversities.append(diversity)
        
        return np.mean(diversities)

class EvaluationManager:
    """Manager for evaluation process"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.metrics_calculator = RecommendationMetrics()
        
    def evaluate_model(self, test_loader, article_embeddings_cache: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate the model on test set
        
        Args:
            test_loader: DataLoader for test data
            article_embeddings_cache: Pre-computed embeddings for all articles
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        self.metrics_calculator.reset()
        
        logger.info("Starting model evaluation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if not batch:  # Empty batch
                    continue
                    
                try:
                    # Move batch to device
                    positive_images = batch['positive_images'].to(self.device)
                    positive_text_input_ids = batch['positive_text_input_ids'].to(self.device)
                    positive_text_attention_masks = batch['positive_text_attention_masks'].to(self.device)
                    customer_ids = batch['customer_ids']
                    positive_article_ids = batch['positive_article_ids']
                    
                    batch_size = positive_images.size(0)
                    
                    # Get embeddings for positive items (ground truth)
                    positive_embeddings = self.model(
                        positive_images, 
                        positive_text_input_ids, 
                        positive_text_attention_masks
                    )
                    
                    # Compute similarities with all articles
                    all_article_ids = list(article_embeddings_cache.keys())
                    all_embeddings = torch.stack([article_embeddings_cache[aid] for aid in all_article_ids])
                    
                    # Compute similarity scores
                    similarities = torch.mm(positive_embeddings, all_embeddings.t())  # [batch_size, num_articles]
                    
                    # Prepare ground truth
                    ground_truths = []
                    for i in range(batch_size):
                        # In this case, ground truth is just the positive article
                        ground_truths.append([positive_article_ids[i]])
                    
                    # Update metrics
                    self.metrics_calculator.update(customer_ids, similarities, ground_truths)
                    
                    if batch_idx % 100 == 0:
                        logger.info(f"Evaluated {batch_idx} batches")
                        
                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        # Compute final metrics
        metrics = self.metrics_calculator.compute_metrics()
        
        logger.info("Evaluation completed")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        return metrics
    
    def compute_article_embeddings(self, articles_df: pd.DataFrame, images_dir: str) -> Dict[str, torch.Tensor]:
        """
        Pre-compute embeddings for all articles
        
        Args:
            articles_df: DataFrame with article information
            images_dir: Directory containing images
            
        Returns:
            Dictionary mapping article_id to embedding tensor
        """
        logger.info("Computing article embeddings...")
        
        from data_loader import StreamingFashionDataset
        
        # Create a temporary dataset for article embedding computation
        temp_dataset = StreamingFashionDataset(
            transactions_path="dummy",  # Not used for this purpose
            articles_path="dummy",      # Not used for this purpose
            customers_path="dummy",     # Not used for this purpose
            images_dir=images_dir,
            tokenizer_name=self.config.model.bert_model_name,
            image_size=self.config.data.image_size,
            max_text_length=self.config.data.max_text_length,
            mode="eval"
        )
        
        article_embeddings = {}
        
        self.model.eval()
        with torch.no_grad():
            for _, row in articles_df.iterrows():
                article_id = str(row['article_id'])
                
                try:
                    # Load image
                    image = temp_dataset._load_image(article_id)
                    if image is None:
                        continue
                    
                    # Create text description
                    text_parts = []
                    for col in ['prod_name', 'product_type_name', 'product_group_name', 
                               'colour_group_name', 'department_name', 'detail_desc']:
                        if col in row and pd.notna(row[col]):
                            text_parts.append(str(row[col]))
                    
                    text = " ".join(text_parts)
                    text_encoding = temp_dataset._tokenize_text(text)
                    
                    # Move to device
                    image = image.unsqueeze(0).to(self.device)
                    input_ids = text_encoding['input_ids'].unsqueeze(0).to(self.device)
                    attention_mask = text_encoding['attention_mask'].unsqueeze(0).to(self.device)
                    
                    # Get embedding
                    embedding = self.model(image, input_ids, attention_mask)
                    article_embeddings[article_id] = embedding.squeeze(0).cpu()
                    
                except Exception as e:
                    logger.warning(f"Error computing embedding for article {article_id}: {e}")
                    continue
        
        logger.info(f"Computed embeddings for {len(article_embeddings)} articles")
        return article_embeddings
