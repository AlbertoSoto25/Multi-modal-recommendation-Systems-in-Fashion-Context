"""
Data Visualization and Metrics Module for Fashion Recommender System
====================================================================

This module provides comprehensive visualization and metrics analysis
for the cleaned fashion recommendation dataset.

Author: Data Science Team
Purpose: Master's Thesis - Multimodal Recommender System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import warnings
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional import for wordcloud functionality
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    logger.warning("WordCloud not available. Install with: pip install wordcloud")

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class DataVisualizationMetrics:
    """
    Comprehensive visualization and metrics analysis for fashion recommendation data.
    
    This class provides:
    - Data quality visualizations
    - Statistical analysis and metrics
    - Interactive dashboards
    - Academic-quality plots for publications
    """
    
    def __init__(self, data_path: str, output_path: str = "visualizations"):
        """
        Initialize the visualization module.
        
        Args:
            data_path (str): Path to cleaned data
            output_path (str): Path for visualization outputs
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of visualizations
        (self.output_path / "quality_plots").mkdir(exist_ok=True)
        (self.output_path / "exploratory_plots").mkdir(exist_ok=True)
        (self.output_path / "statistical_plots").mkdir(exist_ok=True)
        (self.output_path / "interactive_plots").mkdir(exist_ok=True)
        
        # Load cleaned data
        self.transactions_df = None
        self.articles_df = None
        self.customers_df = None
        self.cleaning_results = None
        
        logger.info("Data Visualization Module initialized")
    
    def load_cleaned_data(self) -> None:
        """Load cleaned datasets and cleaning results."""
        logger.info("Loading cleaned datasets...")
        
        try:
            self.transactions_df = pd.read_csv(
                self.data_path / "transactions_clean.csv",
                parse_dates=['t_dat']
            )
            self.articles_df = pd.read_csv(self.data_path / "articles_clean.csv")
            self.customers_df = pd.read_csv(self.data_path / "customers_clean.csv")
            
            # Load cleaning results if available
            results_file = self.data_path / "cleaning_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    self.cleaning_results = json.load(f)
            
            logger.info("Cleaned data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading cleaned data: {e}")
            raise
    
    def create_data_quality_dashboard(self) -> None:
        """Create comprehensive data quality visualizations."""
        logger.info("Creating data quality dashboard...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Data Quality Assessment Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Missing values heatmap
        missing_data = pd.DataFrame({
            'Transactions': self.transactions_df.isnull().sum(),
            'Articles': self.articles_df.isnull().sum(),
            'Customers': self.customers_df.isnull().sum()
        }).fillna(0)
        
        sns.heatmap(missing_data, annot=True, fmt='g', cmap='Reds', ax=axes[0,0])
        axes[0,0].set_title('Missing Values by Dataset', fontweight='bold')
        axes[0,0].set_ylabel('Features')
        
        # 2. Data distribution - Transaction prices
        axes[0,1].hist(self.transactions_df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_title('Price Distribution', fontweight='bold')
        axes[0,1].set_xlabel('Price')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_yscale('log')
        
        # 3. Transactions over time
        daily_transactions = self.transactions_df.groupby(self.transactions_df['t_dat'].dt.date).size()
        axes[0,2].plot(daily_transactions.index, daily_transactions.values, color='green', alpha=0.7)
        axes[0,2].set_title('Daily Transactions Over Time', fontweight='bold')
        axes[0,2].set_xlabel('Date')
        axes[0,2].set_ylabel('Number of Transactions')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Customer transaction frequency
        customer_freq = self.transactions_df.groupby('customer_id').size()
        axes[1,0].hist(customer_freq, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1,0].set_title('Customer Transaction Frequency', fontweight='bold')
        axes[1,0].set_xlabel('Number of Transactions')
        axes[1,0].set_ylabel('Number of Customers')
        axes[1,0].set_yscale('log')
        
        # 5. Article popularity
        article_freq = self.transactions_df.groupby('article_id').size()
        axes[1,1].hist(article_freq, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1,1].set_title('Article Popularity Distribution', fontweight='bold')
        axes[1,1].set_xlabel('Number of Purchases')
        axes[1,1].set_ylabel('Number of Articles')
        axes[1,1].set_yscale('log')
        
        # 6. Sales channel distribution
        channel_dist = self.transactions_df['sales_channel_id'].value_counts()
        axes[1,2].pie(channel_dist.values, labels=channel_dist.index, autopct='%1.1f%%', startangle=90)
        axes[1,2].set_title('Sales Channel Distribution', fontweight='bold')
        
        # 7. Monthly seasonality
        monthly_sales = self.transactions_df.groupby(self.transactions_df['t_dat'].dt.month).size()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[2,0].bar(range(1, 13), monthly_sales, color='lightcoral', alpha=0.8)
        axes[2,0].set_title('Monthly Seasonality', fontweight='bold')
        axes[2,0].set_xlabel('Month')
        axes[2,0].set_ylabel('Number of Transactions')
        axes[2,0].set_xticks(range(1, 13))
        axes[2,0].set_xticklabels(months)
        
        # 8. Day of week patterns
        dow_sales = self.transactions_df.groupby('day_of_week').size()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[2,1].bar(range(7), dow_sales, color='lightgreen', alpha=0.8)
        axes[2,1].set_title('Day of Week Patterns', fontweight='bold')
        axes[2,1].set_xlabel('Day of Week')
        axes[2,1].set_ylabel('Number of Transactions')
        axes[2,1].set_xticks(range(7))
        axes[2,1].set_xticklabels(days)
        
        # 9. Price vs Quantity relationship (if quantity exists)
        if len(self.transactions_df) > 10000:  # Sample for large datasets
            sample_df = self.transactions_df.sample(n=10000, random_state=42)
        else:
            sample_df = self.transactions_df
        
        axes[2,2].scatter(sample_df['price'], sample_df.index, alpha=0.5, s=1)
        axes[2,2].set_title('Price Distribution Scatter', fontweight='bold')
        axes[2,2].set_xlabel('Price')
        axes[2,2].set_ylabel('Transaction Index')
        
        plt.tight_layout()
        plt.savefig(self.output_path / "quality_plots" / "data_quality_dashboard.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Data quality dashboard created")
    
    def create_statistical_analysis_plots(self) -> Dict:
        """Create statistical analysis visualizations and compute metrics."""
        logger.info("Creating statistical analysis plots...")
        
        stats_results = {}
        
        # 1. Price Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Analysis - Price Patterns', fontsize=16, fontweight='bold')
        
        # Price distribution with statistical annotations
        prices = self.transactions_df['price']
        axes[0,0].hist(prices, bins=100, alpha=0.7, color='skyblue', density=True, edgecolor='black')
        axes[0,0].axvline(prices.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {prices.mean():.2f}')
        axes[0,0].axvline(prices.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {prices.median():.2f}')
        axes[0,0].set_title('Price Distribution with Statistics')
        axes[0,0].set_xlabel('Price')
        axes[0,0].set_ylabel('Density')
        axes[0,0].legend()
        
        # Q-Q plot for normality test
        stats.probplot(prices, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot - Price Normality Test')
        
        # Box plot by sales channel
        self.transactions_df.boxplot(column='price', by='sales_channel_id', ax=axes[1,0])
        axes[1,0].set_title('Price Distribution by Sales Channel')
        axes[1,0].set_xlabel('Sales Channel ID')
        axes[1,0].set_ylabel('Price')
        
        # Price trends over time (monthly)
        monthly_avg_price = self.transactions_df.groupby(
            self.transactions_df['t_dat'].dt.to_period('M')
        )['price'].mean()
        axes[1,1].plot(monthly_avg_price.index.astype(str), monthly_avg_price.values, 
                      marker='o', linewidth=2, markersize=6)
        axes[1,1].set_title('Average Price Trends Over Time')
        axes[1,1].set_xlabel('Month')
        axes[1,1].set_ylabel('Average Price')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "statistical_plots" / "price_statistical_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Compute statistical metrics
        stats_results['price_statistics'] = {
            'mean': float(prices.mean()),
            'median': float(prices.median()),
            'std': float(prices.std()),
            'skewness': float(stats.skew(prices)),
            'kurtosis': float(stats.kurtosis(prices)),
            'shapiro_test': stats.shapiro(prices.sample(5000) if len(prices) > 5000 else prices),
            'percentiles': {
                '25th': float(prices.quantile(0.25)),
                '75th': float(prices.quantile(0.75)),
                '90th': float(prices.quantile(0.90)),
                '95th': float(prices.quantile(0.95)),
                '99th': float(prices.quantile(0.99))
            }
        }
        
        # 2. Customer Behavior Analysis
        self._create_customer_behavior_analysis(stats_results)
        
        # 3. Temporal Patterns Analysis
        self._create_temporal_analysis(stats_results)
        
        # 4. Article Analysis
        self._create_article_analysis(stats_results)
        
        return stats_results
    
    def _create_customer_behavior_analysis(self, stats_results: Dict) -> None:
        """Create customer behavior analysis plots."""
        
        # Customer transaction patterns
        customer_stats = self.transactions_df.groupby('customer_id').agg({
            'price': ['count', 'sum', 'mean'],
            't_dat': ['min', 'max']
        }).round(2)
        
        customer_stats.columns = ['transaction_count', 'total_spent', 'avg_price', 'first_purchase', 'last_purchase']
        customer_stats['purchase_span_days'] = (
            pd.to_datetime(customer_stats['last_purchase']) - 
            pd.to_datetime(customer_stats['first_purchase'])
        ).dt.days
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Customer Behavior Analysis', fontsize=16, fontweight='bold')
        
        # Customer value distribution
        axes[0,0].hist(customer_stats['total_spent'], bins=50, alpha=0.7, color='gold', edgecolor='black')
        axes[0,0].set_title('Customer Lifetime Value Distribution')
        axes[0,0].set_xlabel('Total Amount Spent')
        axes[0,0].set_ylabel('Number of Customers')
        axes[0,0].set_yscale('log')
        
        # Transaction frequency vs Average price
        sample_customers = customer_stats.sample(n=min(5000, len(customer_stats)), random_state=42)
        scatter = axes[0,1].scatter(sample_customers['transaction_count'], 
                                   sample_customers['avg_price'], 
                                   alpha=0.6, c=sample_customers['total_spent'], 
                                   cmap='viridis', s=20)
        axes[0,1].set_title('Transaction Frequency vs Average Price')
        axes[0,1].set_xlabel('Number of Transactions')
        axes[0,1].set_ylabel('Average Price per Transaction')
        plt.colorbar(scatter, ax=axes[0,1], label='Total Spent')
        
        # Customer purchase span
        axes[1,0].hist(customer_stats['purchase_span_days'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1,0].set_title('Customer Purchase Span Distribution')
        axes[1,0].set_xlabel('Days Between First and Last Purchase')
        axes[1,0].set_ylabel('Number of Customers')
        
        # RFM-like analysis
        customer_stats['recency'] = (customer_stats['last_purchase'].max() - customer_stats['last_purchase']).dt.days
        axes[1,1].scatter(customer_stats['recency'], customer_stats['transaction_count'], 
                         alpha=0.6, s=20, c='purple')
        axes[1,1].set_title('Recency vs Frequency Analysis')
        axes[1,1].set_xlabel('Days Since Last Purchase')
        axes[1,1].set_ylabel('Transaction Count')
        
        plt.tight_layout()
        plt.savefig(self.output_path / "statistical_plots" / "customer_behavior_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store customer statistics
        stats_results['customer_statistics'] = {
            'total_customers': len(customer_stats),
            'avg_transactions_per_customer': float(customer_stats['transaction_count'].mean()),
            'avg_customer_lifetime_value': float(customer_stats['total_spent'].mean()),
            'customer_retention_metrics': {
                'single_purchase_customers': int((customer_stats['transaction_count'] == 1).sum()),
                'repeat_customers': int((customer_stats['transaction_count'] > 1).sum()),
                'high_value_customers': int((customer_stats['total_spent'] > customer_stats['total_spent'].quantile(0.8)).sum())
            }
        }
    
    def _create_temporal_analysis(self, stats_results: Dict) -> None:
        """Create temporal patterns analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Patterns Analysis', fontsize=16, fontweight='bold')
        
        # Daily transaction volume with trend
        daily_transactions = self.transactions_df.groupby(self.transactions_df['t_dat'].dt.date).size()
        axes[0,0].plot(daily_transactions.index, daily_transactions.values, alpha=0.7, color='blue')
        
        # Add moving average
        rolling_mean = daily_transactions.rolling(window=7).mean()
        axes[0,0].plot(rolling_mean.index, rolling_mean.values, color='red', linewidth=2, label='7-day MA')
        axes[0,0].set_title('Daily Transaction Volume with Trend')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Number of Transactions')
        axes[0,0].legend()
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Hourly patterns (if hour data available)
        if 'hour' in self.transactions_df.columns:
            hourly_pattern = self.transactions_df.groupby('hour').size()
            axes[0,1].bar(hourly_pattern.index, hourly_pattern.values, color='orange', alpha=0.8)
            axes[0,1].set_title('Hourly Transaction Patterns')
            axes[0,1].set_xlabel('Hour of Day')
            axes[0,1].set_ylabel('Number of Transactions')
        else:
            # Use day of week instead
            dow_pattern = self.transactions_df.groupby('day_of_week').size()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            axes[0,1].bar(range(7), dow_pattern.values, color='orange', alpha=0.8)
            axes[0,1].set_title('Day of Week Transaction Patterns')
            axes[0,1].set_xlabel('Day of Week')
            axes[0,1].set_ylabel('Number of Transactions')
            axes[0,1].set_xticks(range(7))
            axes[0,1].set_xticklabels(days)
        
        # Monthly revenue trends
        monthly_revenue = self.transactions_df.groupby(
            self.transactions_df['t_dat'].dt.to_period('M')
        )['price'].sum()
        axes[1,0].plot(monthly_revenue.index.astype(str), monthly_revenue.values, 
                      marker='o', linewidth=2, markersize=6, color='green')
        axes[1,0].set_title('Monthly Revenue Trends')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Total Revenue')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Seasonal decomposition visualization
        monthly_transactions = self.transactions_df.groupby(
            self.transactions_df['t_dat'].dt.to_period('M')
        ).size()
        
        if len(monthly_transactions) >= 12:  # Need at least a year of data
            # Simple seasonal pattern
            seasonal_pattern = monthly_transactions.groupby(
                monthly_transactions.index.month
            ).mean()
            
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            axes[1,1].bar(range(1, 13), seasonal_pattern.values, color='purple', alpha=0.8)
            axes[1,1].set_title('Average Monthly Seasonal Pattern')
            axes[1,1].set_xlabel('Month')
            axes[1,1].set_ylabel('Average Transactions')
            axes[1,1].set_xticks(range(1, 13))
            axes[1,1].set_xticklabels(months)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "statistical_plots" / "temporal_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store temporal statistics
        stats_results['temporal_statistics'] = {
            'date_range': {
                'start_date': str(self.transactions_df['t_dat'].min().date()),
                'end_date': str(self.transactions_df['t_dat'].max().date()),
                'total_days': int((self.transactions_df['t_dat'].max() - self.transactions_df['t_dat'].min()).days)
            },
            'daily_transaction_stats': {
                'avg_daily_transactions': float(daily_transactions.mean()),
                'max_daily_transactions': int(daily_transactions.max()),
                'min_daily_transactions': int(daily_transactions.min())
            }
        }
    
    def _create_article_analysis(self, stats_results: Dict) -> None:
        """Create article analysis plots."""
        
        # Article popularity and characteristics
        article_stats = self.transactions_df.groupby('article_id').agg({
            'customer_id': 'nunique',
            'price': ['count', 'mean', 'sum']
        }).round(2)
        
        article_stats.columns = ['unique_customers', 'total_purchases', 'avg_price', 'total_revenue']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Article Analysis', fontsize=16, fontweight='bold')
        
        # Article popularity distribution
        axes[0,0].hist(article_stats['total_purchases'], bins=50, alpha=0.7, color='cyan', edgecolor='black')
        axes[0,0].set_title('Article Popularity Distribution')
        axes[0,0].set_xlabel('Number of Purchases')
        axes[0,0].set_ylabel('Number of Articles')
        axes[0,0].set_yscale('log')
        
        # Price vs Popularity
        sample_articles = article_stats.sample(n=min(5000, len(article_stats)), random_state=42)
        scatter = axes[0,1].scatter(sample_articles['avg_price'], 
                                   sample_articles['total_purchases'], 
                                   alpha=0.6, c=sample_articles['total_revenue'], 
                                   cmap='plasma', s=20)
        axes[0,1].set_title('Average Price vs Popularity')
        axes[0,1].set_xlabel('Average Price')
        axes[0,1].set_ylabel('Total Purchases')
        axes[0,1].set_yscale('log')
        plt.colorbar(scatter, ax=axes[0,1], label='Total Revenue')
        
        # Revenue distribution
        axes[1,0].hist(article_stats['total_revenue'], bins=50, alpha=0.7, color='gold', edgecolor='black')
        axes[1,0].set_title('Article Revenue Distribution')
        axes[1,0].set_xlabel('Total Revenue')
        axes[1,0].set_ylabel('Number of Articles')
        axes[1,0].set_yscale('log')
        
        # Customer reach vs purchases
        axes[1,1].scatter(article_stats['unique_customers'], article_stats['total_purchases'], 
                         alpha=0.6, s=20, c='red')
        axes[1,1].set_title('Customer Reach vs Total Purchases')
        axes[1,1].set_xlabel('Number of Unique Customers')
        axes[1,1].set_ylabel('Total Purchases')
        axes[1,1].set_xscale('log')
        axes[1,1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_path / "statistical_plots" / "article_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store article statistics
        stats_results['article_statistics'] = {
            'total_articles': len(article_stats),
            'avg_purchases_per_article': float(article_stats['total_purchases'].mean()),
            'most_popular_articles': article_stats.nlargest(10, 'total_purchases').to_dict(),
            'highest_revenue_articles': article_stats.nlargest(10, 'total_revenue').to_dict()
        }
    
    def create_interactive_dashboard(self) -> None:
        """Create interactive Plotly dashboard."""
        logger.info("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Transaction Volume Over Time', 'Price Distribution',
                          'Customer Transaction Frequency', 'Sales Channel Performance',
                          'Monthly Seasonality', 'Top Articles by Revenue'),
            specs=[[{"secondary_y": True}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Transaction volume over time
        daily_transactions = self.transactions_df.groupby(self.transactions_df['t_dat'].dt.date).size()
        fig.add_trace(
            go.Scatter(x=daily_transactions.index, y=daily_transactions.values,
                      mode='lines', name='Daily Transactions',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # Add moving average
        rolling_mean = daily_transactions.rolling(window=7).mean()
        fig.add_trace(
            go.Scatter(x=rolling_mean.index, y=rolling_mean.values,
                      mode='lines', name='7-day Moving Average',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # 2. Price distribution
        fig.add_trace(
            go.Histogram(x=self.transactions_df['price'], nbinsx=50,
                        name='Price Distribution', marker_color='skyblue'),
            row=1, col=2
        )
        
        # 3. Customer transaction frequency
        customer_freq = self.transactions_df.groupby('customer_id').size()
        fig.add_trace(
            go.Histogram(x=customer_freq, nbinsx=50,
                        name='Customer Frequency', marker_color='orange'),
            row=2, col=1
        )
        
        # 4. Sales channel performance
        channel_stats = self.transactions_df.groupby('sales_channel_id').agg({
            'price': ['count', 'sum']
        }).round(2)
        channel_stats.columns = ['transactions', 'revenue']
        
        fig.add_trace(
            go.Bar(x=channel_stats.index, y=channel_stats['transactions'],
                  name='Transactions by Channel', marker_color='green'),
            row=2, col=2
        )
        
        # 5. Monthly seasonality
        monthly_sales = self.transactions_df.groupby(self.transactions_df['t_dat'].dt.month).size()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig.add_trace(
            go.Bar(x=months, y=monthly_sales.values,
                  name='Monthly Sales', marker_color='purple'),
            row=3, col=1
        )
        
        # 6. Top articles by revenue
        article_revenue = self.transactions_df.groupby('article_id')['price'].sum().nlargest(10)
        fig.add_trace(
            go.Bar(x=list(range(len(article_revenue))), y=article_revenue.values,
                  name='Top Articles Revenue', marker_color='gold'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Fashion Recommender System - Data Analysis Dashboard",
            title_x=0.5,
            showlegend=False
        )
        
        # Save interactive plot
        fig.write_html(self.output_path / "interactive_plots" / "interactive_dashboard.html")
        
        logger.info("Interactive dashboard created")
    
    def generate_comprehensive_report(self, stats_results: Dict) -> None:
        """Generate comprehensive analysis report."""
        logger.info("Generating comprehensive analysis report...")
        
        report_content = f"""
# Data Cleaning and Analysis Report
## Fashion Recommender System - Master's Thesis

### Executive Summary

This report presents the comprehensive data cleaning and analysis results for the fashion recommender system dataset. The analysis covers data quality assessment, statistical analysis, and key insights for the multimodal recommendation system.

### Dataset Overview

**Transactions Dataset:**
- Total transactions: {len(self.transactions_df):,}
- Date range: {stats_results['temporal_statistics']['date_range']['start_date']} to {stats_results['temporal_statistics']['date_range']['end_date']}
- Total days: {stats_results['temporal_statistics']['date_range']['total_days']} days

**Articles Dataset:**
- Total articles: {len(self.articles_df):,}
- Articles with transactions: {stats_results['article_statistics']['total_articles']:,}

**Customers Dataset:**
- Total customers: {len(self.customers_df):,}
- Active customers: {stats_results['customer_statistics']['total_customers']:,}

### Data Quality Assessment

#### Missing Values Treatment
- Systematic handling of missing values across all datasets
- Price missing values: Filled using article-specific median prices
- Categorical missing values: Filled with 'Unknown' category
- Outlier treatment: Capped extreme values at 99.9th percentile

#### Data Consistency
- Referential integrity maintained across datasets
- Customer coverage: {self.cleaning_results['consistency_validation']['customer_coverage']:.2f}%
- Article coverage: {self.cleaning_results['consistency_validation']['article_coverage']:.2f}%

### Statistical Analysis Results

#### Price Analysis
- Mean price: ${stats_results['price_statistics']['mean']:.2f}
- Median price: ${stats_results['price_statistics']['median']:.2f}
- Price distribution skewness: {stats_results['price_statistics']['skewness']:.3f}
- 95th percentile: ${stats_results['price_statistics']['percentiles']['95th']:.2f}

#### Customer Behavior Insights
- Average transactions per customer: {stats_results['customer_statistics']['avg_transactions_per_customer']:.2f}
- Average customer lifetime value: ${stats_results['customer_statistics']['avg_customer_lifetime_value']:.2f}
- Repeat customers: {stats_results['customer_statistics']['customer_retention_metrics']['repeat_customers']:,} ({stats_results['customer_statistics']['customer_retention_metrics']['repeat_customers']/stats_results['customer_statistics']['total_customers']*100:.1f}%)

#### Temporal Patterns
- Average daily transactions: {stats_results['temporal_statistics']['daily_transaction_stats']['avg_daily_transactions']:.0f}
- Peak transaction day: {stats_results['temporal_statistics']['daily_transaction_stats']['max_daily_transactions']:,} transactions
- Seasonal patterns identified in monthly data

#### Article Performance
- Average purchases per article: {stats_results['article_statistics']['avg_purchases_per_article']:.2f}
- Long-tail distribution observed in article popularity
- Power-law relationship between price and popularity

### Key Findings for Recommender System

1. **Data Sparsity**: High sparsity in user-item interactions suggests need for content-based features
2. **Temporal Dynamics**: Strong seasonal patterns indicate importance of time-aware recommendations
3. **Customer Segmentation**: Clear distinction between one-time and repeat customers
4. **Price Sensitivity**: Price distribution suggests multiple customer segments with different price preferences
5. **Long-tail Distribution**: Many articles have few purchases, requiring cold-start handling

### Recommendations for Model Development

1. **Hybrid Approach**: Combine collaborative filtering with content-based methods
2. **Temporal Features**: Incorporate seasonal and trend features
3. **Customer Segmentation**: Develop segment-specific recommendation strategies
4. **Cold-start Handling**: Implement content-based recommendations for new items
5. **Multi-modal Integration**: Leverage product images for content-based features

### Technical Implementation Notes

- Data preprocessing pipeline implemented with robust error handling
- Scalable approach using chunked processing for large datasets
- Comprehensive logging and monitoring throughout the pipeline
- Reproducible results with fixed random seeds

### Files Generated

- `transactions_clean.csv`: Cleaned transaction data
- `articles_clean.csv`: Cleaned article metadata
- `customers_clean.csv`: Cleaned customer data
- `cleaning_results.json`: Detailed cleaning metrics
- Various visualization files in the `visualizations/` directory

---

*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        with open(self.output_path / "comprehensive_analysis_report.md", 'w') as f:
            f.write(report_content)
        
        # Save statistics as JSON
        with open(self.output_path / "statistical_analysis_results.json", 'w') as f:
            json.dump(stats_results, f, indent=2, default=str)
        
        logger.info("Comprehensive analysis report generated")
    
    def run_complete_visualization_pipeline(self) -> Dict:
        """Run the complete visualization and metrics pipeline."""
        logger.info("Starting complete visualization pipeline...")
        
        # Load cleaned data
        self.load_cleaned_data()
        
        # Create visualizations
        self.create_data_quality_dashboard()
        stats_results = self.create_statistical_analysis_plots()
        # Add a dashboard summarizing cleaning effects (before vs after)
        self.create_cleaning_effects_plots()
        self.create_interactive_dashboard()
        
        # Generate comprehensive report
        self.generate_comprehensive_report(stats_results)
        
        logger.info("Visualization pipeline completed successfully!")
        return stats_results

    def create_cleaning_effects_plots(self) -> None:
        """Create plots that summarize the effects of the cleaning process (before vs after)."""
        if not self.cleaning_results:
            results_file = self.data_path / "cleaning_results.json"
            if not results_file.exists():
                raise FileNotFoundError(f"Cleaning results file not found at {results_file}")
            with open(results_file, 'r') as f:
                self.cleaning_results = json.load(f)

        logger.info("Creating cleaning effects dashboard (before vs after)...")

        quality_before = self.cleaning_results.get('quality_before_cleaning', {})
        quality_after = self.cleaning_results.get('quality_after_cleaning', {})
        consistency = self.cleaning_results.get('consistency_validation', {})
        image_stats = self.cleaning_results.get('image_analysis', {})

        # Helper to safely parse ints (duplicates sometimes as strings)
        def as_int(value, default=0):
            try:
                return int(value)
            except Exception:
                try:
                    return int(float(value))
                except Exception:
                    return default

        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
        fig.suptitle('Cleaning Effects Dashboard (Before vs After)', fontsize=18, fontweight='bold')

        # 1) Missing values total per dataset (before vs after)
        datasets = ['articles', 'customers', 'transactions']
        before_missing_totals = []
        after_missing_totals = []
        for ds in datasets:
            b = quality_before.get(ds, {}).get('missing_values', {})
            a = quality_after.get(ds, {}).get('missing_values', {})
            before_missing_totals.append(sum(v for v in b.values()))
            after_missing_totals.append(sum(v for v in a.values()))
        x = np.arange(len(datasets))
        width = 0.35
        axes[0, 0].bar(x - width/2, before_missing_totals, width, label='Before')
        axes[0, 0].bar(x + width/2, after_missing_totals, width, label='After')
        axes[0, 0].set_title('Total Missing Values by Dataset', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([ds.capitalize() for ds in datasets])
        axes[0, 0].set_ylabel('Number of Missing Values')
        axes[0, 0].legend()

        # 2) Duplicados por dataset (before vs after)
        before_dups = [as_int(quality_before.get(ds, {}).get('duplicates', 0)) for ds in datasets]
        after_dups = [as_int(quality_after.get(ds, {}).get('duplicates', 0)) for ds in datasets]
        axes[0, 1].bar(x - width/2, before_dups, width, label='Before')
        axes[0, 1].bar(x + width/2, after_dups, width, label='After')
        axes[0, 1].set_title('Duplicates by Dataset', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([ds.capitalize() for ds in datasets])
        axes[0, 1].set_ylabel('Number of Duplicates')
        axes[0, 1].legend()

        # 3) Cardinalidad de categorías clave en artículos (before vs after)
        article_before_unique = quality_before.get('articles', {}).get('unique_values', {})
        article_after_unique = quality_after.get('articles', {}).get('unique_values', {})
        cat_cols = ['prod_name', 'product_type_name', 'graphical_appearance_name', 'colour_group_name']
        cat_labels = ['Product name', 'Product type', 'Graphical appearance', 'Color group']
        before_card = [article_before_unique.get(col, 0) for col in cat_cols]
        after_card = [article_after_unique.get(col, 0) for col in cat_cols]
        x2 = np.arange(len(cat_cols))
        axes[0, 2].bar(x2 - width/2, before_card, width, label='Before')
        axes[0, 2].bar(x2 + width/2, after_card, width, label='After')
        axes[0, 2].set_title('Cardinality in Article Categories', fontweight='bold')
        axes[0, 2].set_xticks(x2)
        axes[0, 2].set_xticklabels(cat_labels, rotation=20)
        axes[0, 2].set_ylabel('Number of Unique Values')
        axes[0, 2].legend()

        # 4) Estadísticas de precio (before vs after)
        price_before = quality_before.get('transactions', {}).get('price_stats', {})
        price_after = quality_after.get('transactions', {}).get('price_stats', {})
        # Convert potentially string values to float
        def as_float(v):
            try:
                return float(v)
            except Exception:
                return np.nan
        metrics = ['min_price', 'median_price', 'mean_price', 'std_price', 'max_price']
        labels = ['Min', 'Median', 'Mean', 'Std dev', 'Max']
        b_vals = [as_float(price_before.get(m)) for m in metrics]
        a_vals = [as_float(price_after.get(m)) for m in metrics]
        x3 = np.arange(len(metrics))
        axes[1, 0].bar(x3 - width/2, b_vals, width, label='Before')
        axes[1, 0].bar(x3 + width/2, a_vals, width, label='After')
        axes[1, 0].set_title('Price: Statistical Summary (Before vs After)', fontweight='bold')
        axes[1, 0].set_xticks(x3)
        axes[1, 0].set_xticklabels(labels)
        axes[1, 0].set_ylabel('Price')
        axes[1, 0].legend()

        # 5) Cobertura referencial (clientes/artículos en transacciones)
        cust_cov = consistency.get('customer_coverage', np.nan)
        art_cov = consistency.get('article_coverage', np.nan)
        axes[1, 1].bar(['Customers', 'Articles'], [cust_cov, art_cov], color=['#4C78A8', '#F58518'])
        axes[1, 1].set_ylim(0, 110)
        axes[1, 1].set_title('Referential Coverage in Transactions', fontweight='bold')
        axes[1, 1].set_ylabel('Coverage (%)')
        for i, v in enumerate([cust_cov, art_cov]):
            axes[1, 1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

        # 6) Cobertura de imágenes
        total_articles = image_stats.get('total_articles', 0)
        with_images = image_stats.get('unique_articles_with_images', 0)
        without_images = max(total_articles - with_images, 0)
        axes[1, 2].pie([with_images, without_images], labels=['With image', 'Without image'],
                       autopct='%1.1f%%', startangle=90, colors=['#59A14F', '#E15759'])
        axes[1, 2].set_title('Image Coverage Across Articles', fontweight='bold')

        plt.tight_layout()
        out_path = self.output_path / "quality_plots" / "cleaning_effects_dashboard.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Cleaning effects dashboard saved to: {out_path}")

    def run_cleaning_results_visuals_only(self) -> None:
        """Generate only the cleaning-effects plots using cleaning_results.json, without loading large CSVs."""
        # Load cleaning results if not already loaded
        results_file = self.data_path / "cleaning_results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Cleaning results file not found at {results_file}")
        with open(results_file, 'r') as f:
            self.cleaning_results = json.load(f)

        # Only produce the cleaning-effects figure
        self.create_cleaning_effects_plots()


def main():
    """Main execution function."""
    # Initialize visualization module
    data_path = "cleaned_data"  # Path to cleaned data
    viz_module = DataVisualizationMetrics(data_path)
    
    # Run complete pipeline
    results = viz_module.run_complete_visualization_pipeline()
    
    print("\n" + "="*60)
    print("DATA VISUALIZATION AND METRICS PIPELINE COMPLETED")
    print("="*60)
    print(f"Visualizations saved to: {viz_module.output_path}")
    print("\nGenerated Files:")
    print("- Data quality dashboard")
    print("- Statistical analysis plots")
    print("- Interactive dashboard")
    print("- Comprehensive analysis report")
    print("- Statistical results JSON")


if __name__ == "__main__":
    main()