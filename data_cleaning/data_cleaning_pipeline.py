"""
Data Cleaning Pipeline for Multimodal Fashion Recommender System
================================================================

This module implements a comprehensive data cleaning pipeline for a fashion recommendation
system using transaction, customer, and article data along with product images.

Author: Data Science Team
Purpose: Master's Thesis - Multimodal Recommender System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os
from PIL import Image
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class DataCleaningPipeline:
    """
    Comprehensive data cleaning pipeline for fashion recommender system.
    
    This class handles the complete data preprocessing workflow including:
    - Data loading and validation
    - Missing value treatment
    - Outlier detection and handling
    - Data type optimization
    - Feature engineering
    - Data quality assessment
    """
    
    def __init__(self, data_path: str, output_path: str = "cleaned_data"):
        """
        Initialize the data cleaning pipeline.
        
        Args:
            data_path (str): Path to the raw data directory
            output_path (str): Path for cleaned data output
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.articles_df = None
        self.customers_df = None
        self.transactions_df = None
        
        # Quality metrics storage
        self.quality_metrics = {}
        self.cleaning_report = {}
        
        logger.info("Data Cleaning Pipeline initialized")
    
    def load_data(self) -> None:
        """Load all datasets with optimized data types."""
        logger.info("Loading datasets...")
        
        try:
            # Load articles data
            logger.info("Loading articles.csv...")
            self.articles_df = pd.read_csv(
                self.data_path / "articles.csv",
                dtype={'article_id': 'str'}
            )
            
            # Load customers data
            logger.info("Loading customers.csv...")
            self.customers_df = pd.read_csv(
                self.data_path / "customers.csv",
                dtype={'customer_id': 'str'}
            )
            
            # Load transactions data with chunking for large files
            logger.info("Loading transactions_train.csv...")
            chunk_size = 100000
            chunks = []
            
            for chunk in pd.read_csv(
                self.data_path / "transactions_train.csv",
                chunksize=chunk_size,
                dtype={
                    'customer_id': 'str',
                    'article_id': 'str',
                    'price': 'float32',
                    'sales_channel_id': 'int8'
                },
                parse_dates=['t_dat']
            ):
                chunks.append(chunk)
            
            self.transactions_df = pd.concat(chunks, ignore_index=True)
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"  - Articles: {len(self.articles_df):,} rows")
            logger.info(f"  - Customers: {len(self.customers_df):,} rows")
            logger.info(f"  - Transactions: {len(self.transactions_df):,} rows")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_data_quality(self) -> Dict:
        """
        Comprehensive data quality analysis.
        
        Returns:
            Dict: Quality metrics for each dataset
        """
        logger.info("Analyzing data quality...")
        
        quality_report = {}
        
        # Analyze each dataset
        for name, df in [("articles", self.articles_df), 
                        ("customers", self.customers_df), 
                        ("transactions", self.transactions_df)]:
            
            logger.info(f"Analyzing {name} dataset...")
            
            analysis = {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'data_types': df.dtypes.to_dict(),
                'duplicates': df.duplicated().sum(),
                'unique_values': {col: df[col].nunique() for col in df.columns}
            }
            
            # Add specific analysis for each dataset type
            if name == "transactions":
                analysis.update(self._analyze_transactions_specific())
            elif name == "articles":
                analysis.update(self._analyze_articles_specific())
            elif name == "customers":
                analysis.update(self._analyze_customers_specific())
            
            quality_report[name] = analysis
        
        self.quality_metrics = quality_report
        return quality_report
    
    def _analyze_transactions_specific(self) -> Dict:
        """Specific analysis for transactions dataset."""
        df = self.transactions_df
        
        return {
            'date_range': {
                'min_date': df['t_dat'].min(),
                'max_date': df['t_dat'].max(),
                'date_span_days': (df['t_dat'].max() - df['t_dat'].min()).days
            },
            'price_stats': {
                'min_price': df['price'].min(),
                'max_price': df['price'].max(),
                'mean_price': df['price'].mean(),
                'median_price': df['price'].median(),
                'std_price': df['price'].std()
            },
            'sales_channels': df['sales_channel_id'].value_counts().to_dict(),
            'transactions_per_customer': df.groupby('customer_id').size().describe().to_dict(),
            'transactions_per_article': df.groupby('article_id').size().describe().to_dict()
        }
    
    def _analyze_articles_specific(self) -> Dict:
        """Specific analysis for articles dataset."""
        df = self.articles_df
        
        analysis = {}
        
        # Analyze categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'article_id':
                analysis[f'{col}_categories'] = df[col].value_counts().head(10).to_dict()
        
        return analysis
    
    def _analyze_customers_specific(self) -> Dict:
        """Specific analysis for customers dataset."""
        df = self.customers_df
        
        analysis = {}
        
        # Age analysis if age column exists
        if 'age' in df.columns:
            analysis['age_stats'] = {
                'min_age': df['age'].min(),
                'max_age': df['age'].max(),
                'mean_age': df['age'].mean(),
                'median_age': df['age'].median()
            }
        
        return analysis
    
    def clean_transactions_data(self) -> pd.DataFrame:
        """
        Clean transactions dataset.
        
        Returns:
            pd.DataFrame: Cleaned transactions data
        """
        logger.info("Cleaning transactions data...")
        
        df = self.transactions_df.copy()
        initial_rows = len(df)
        
        # Remove rows with missing critical values
        df = df.dropna(subset=['customer_id', 'article_id', 't_dat'])
        
        # Handle missing prices (could indicate free items or data errors)
        missing_prices = df['price'].isnull().sum()
        if missing_prices > 0:
            logger.warning(f"Found {missing_prices} transactions with missing prices")
            # Option 1: Remove transactions with missing prices
            # df = df.dropna(subset=['price'])
            # Option 2: Fill with median price per article
            df['price'] = df.groupby('article_id')['price'].transform(
                lambda x: x.fillna(x.median())
            )
            # If still missing, fill with overall median
            df['price'] = df['price'].fillna(df['price'].median())
        
        # Remove negative prices (data errors)
        negative_prices = (df['price'] < 0).sum()
        if negative_prices > 0:
            logger.warning(f"Removing {negative_prices} transactions with negative prices")
            df = df[df['price'] >= 0]
        
        # Remove extreme outliers in price (beyond 99.9th percentile)
        price_threshold = df['price'].quantile(0.999)
        extreme_prices = (df['price'] > price_threshold).sum()
        if extreme_prices > 0:
            logger.info(f"Capping {extreme_prices} extreme price values at {price_threshold:.2f}")
            df.loc[df['price'] > price_threshold, 'price'] = price_threshold
        
        # Ensure valid date format
        df['t_dat'] = pd.to_datetime(df['t_dat'], errors='coerce')
        df = df.dropna(subset=['t_dat'])
        
        # Add derived features
        df['year'] = df['t_dat'].dt.year
        df['month'] = df['t_dat'].dt.month
        df['day_of_week'] = df['t_dat'].dt.dayofweek
        df['day_of_year'] = df['t_dat'].dt.dayofyear
        
        # Sort by customer and date
        df = df.sort_values(['customer_id', 't_dat']).reset_index(drop=True)
        
        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        removal_percentage = (rows_removed / initial_rows) * 100
        
        logger.info(f"Transactions cleaning completed:")
        logger.info(f"  - Initial rows: {initial_rows:,}")
        logger.info(f"  - Final rows: {final_rows:,}")
        logger.info(f"  - Rows removed: {rows_removed:,} ({removal_percentage:.2f}%)")
        
        self.cleaning_report['transactions'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'rows_removed': rows_removed,
            'removal_percentage': removal_percentage
        }
        
        return df
    
    def clean_articles_data(self) -> pd.DataFrame:
        """
        Clean articles dataset.
        
        Returns:
            pd.DataFrame: Cleaned articles data
        """
        logger.info("Cleaning articles data...")
        
        df = self.articles_df.copy()
        initial_rows = len(df)
        
        # Remove duplicates based on article_id
        df = df.drop_duplicates(subset=['article_id'])
        
        # Handle missing values in categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'article_id':
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    logger.info(f"Filling {missing_count} missing values in {col} with 'Unknown'")
                    df[col] = df[col].fillna('Unknown')
        
        # Handle missing values in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_val = df[col].median()
                logger.info(f"Filling {missing_count} missing values in {col} with median: {median_val}")
                df[col] = df[col].fillna(median_val)
        
        # Standardize categorical values (remove extra spaces, standardize case)
        for col in categorical_cols:
            if col != 'article_id':
                df[col] = df[col].astype(str).str.strip().str.title()
        
        final_rows = len(df)
        rows_removed = initial_rows - final_rows
        
        logger.info(f"Articles cleaning completed:")
        logger.info(f"  - Initial rows: {initial_rows:,}")
        logger.info(f"  - Final rows: {final_rows:,}")
        logger.info(f"  - Duplicates removed: {rows_removed:,}")
        
        self.cleaning_report['articles'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'duplicates_removed': rows_removed
        }
        
        return df
    
    def clean_customers_data(self) -> pd.DataFrame:
        """
        Clean customers dataset.
        
        Returns:
            pd.DataFrame: Cleaned customers data
        """
        logger.info("Cleaning customers data...")
        
        df = self.customers_df.copy()
        initial_rows = len(df)
        
        # Remove duplicates based on customer_id
        df = df.drop_duplicates(subset=['customer_id'])
        
        # Handle age column if it exists
        if 'age' in df.columns:
            # Remove unrealistic ages
            unrealistic_ages = ((df['age'] < 10) | (df['age'] > 100)).sum()
            if unrealistic_ages > 0:
                logger.warning(f"Setting {unrealistic_ages} unrealistic ages to NaN")
                df.loc[(df['age'] < 10) | (df['age'] > 100), 'age'] = np.nan
            
            # Fill missing ages with median
            missing_ages = df['age'].isnull().sum()
            if missing_ages > 0:
                median_age = df['age'].median()
                logger.info(f"Filling {missing_ages} missing ages with median: {median_age}")
                df['age'] = df['age'].fillna(median_age)
        
        # Handle other categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'customer_id':
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    logger.info(f"Filling {missing_count} missing values in {col} with 'Unknown'")
                    df[col] = df[col].fillna('Unknown')
                
                # Standardize categorical values
                df[col] = df[col].astype(str).str.strip().str.title()
        
        final_rows = len(df)
        duplicates_removed = initial_rows - final_rows
        
        logger.info(f"Customers cleaning completed:")
        logger.info(f"  - Initial rows: {initial_rows:,}")
        logger.info(f"  - Final rows: {final_rows:,}")
        logger.info(f"  - Duplicates removed: {duplicates_removed:,}")
        
        self.cleaning_report['customers'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'duplicates_removed': duplicates_removed
        }
        
        return df
    
    def validate_data_consistency(self, 
                                transactions_clean: pd.DataFrame,
                                articles_clean: pd.DataFrame,
                                customers_clean: pd.DataFrame) -> Dict:
        """
        Validate consistency across datasets.
        
        Args:
            transactions_clean: Cleaned transactions data
            articles_clean: Cleaned articles data
            customers_clean: Cleaned customers data
            
        Returns:
            Dict: Validation results
        """
        logger.info("Validating data consistency...")
        
        validation_results = {}
        
        # Check referential integrity
        trans_customers = set(transactions_clean['customer_id'].unique())
        existing_customers = set(customers_clean['customer_id'].unique())
        missing_customers = trans_customers - existing_customers
        
        trans_articles = set(transactions_clean['article_id'].unique())
        existing_articles = set(articles_clean['article_id'].unique())
        missing_articles = trans_articles - existing_articles
        
        validation_results = {
            'missing_customers_in_transactions': len(missing_customers),
            'missing_articles_in_transactions': len(missing_articles),
            'customer_coverage': len(existing_customers & trans_customers) / len(trans_customers) * 100,
            'article_coverage': len(existing_articles & trans_articles) / len(trans_articles) * 100
        }
        
        if missing_customers:
            logger.warning(f"Found {len(missing_customers)} customers in transactions but not in customers dataset")
        
        if missing_articles:
            logger.warning(f"Found {len(missing_articles)} articles in transactions but not in articles dataset")
        
        logger.info(f"Data consistency validation completed:")
        logger.info(f"  - Customer coverage: {validation_results['customer_coverage']:.2f}%")
        logger.info(f"  - Article coverage: {validation_results['article_coverage']:.2f}%")
        
        return validation_results
    
    def analyze_image_availability(self) -> Dict:
        """
        Analyze availability of product images.
        
        Returns:
            Dict: Image availability statistics
        """
        logger.info("Analyzing image availability...")
        
        images_path = self.data_path / "images"
        
        if not images_path.exists():
            logger.warning("Images directory not found")
            return {'images_available': False}
        
        # Get all available images
        available_images = []
        for subdir in images_path.iterdir():
            if subdir.is_dir():
                for img_file in subdir.glob("*.jpg"):
                    article_id = img_file.stem
                    available_images.append(article_id)
        
        available_images_set = set(available_images)
        
        # Compare with articles in dataset
        if self.articles_df is not None:
            total_articles = set(self.articles_df['article_id'].unique())
            articles_with_images = total_articles & available_images_set
            
            image_stats = {
                'images_available': True,
                'total_images': len(available_images),
                'unique_articles_with_images': len(articles_with_images),
                'total_articles': len(total_articles),
                'image_coverage_percentage': len(articles_with_images) / len(total_articles) * 100
            }
        else:
            image_stats = {
                'images_available': True,
                'total_images': len(available_images),
                'unique_articles_with_images': len(available_images_set)
            }
        
        logger.info(f"Image analysis completed:")
        logger.info(f"  - Total images: {image_stats['total_images']:,}")
        logger.info(f"  - Unique articles with images: {image_stats['unique_articles_with_images']:,}")
        if 'image_coverage_percentage' in image_stats:
            logger.info(f"  - Image coverage: {image_stats['image_coverage_percentage']:.2f}%")
        
        return image_stats
    
    def run_complete_pipeline(self) -> Dict:
        """
        Run the complete data cleaning pipeline.
        
        Returns:
            Dict: Complete cleaning and analysis results
        """
        logger.info("Starting complete data cleaning pipeline...")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Analyze data quality (before cleaning)
        quality_before = self.analyze_data_quality()
        
        # Step 3: Clean each dataset
        transactions_clean = self.clean_transactions_data()
        articles_clean = self.clean_articles_data()
        customers_clean = self.clean_customers_data()
        
        # Step 4: Validate consistency
        consistency_results = self.validate_data_consistency(
            transactions_clean, articles_clean, customers_clean
        )
        
        # Step 5: Analyze image availability
        image_stats = self.analyze_image_availability()
        
        # Step 6: Save cleaned data
        logger.info("Saving cleaned datasets...")
        transactions_clean.to_csv(self.output_path / "transactions_clean.csv", index=False)
        articles_clean.to_csv(self.output_path / "articles_clean.csv", index=False)
        customers_clean.to_csv(self.output_path / "customers_clean.csv", index=False)
        
        # Step 7: Update cleaned data in class
        self.transactions_df = transactions_clean
        self.articles_df = articles_clean
        self.customers_df = customers_clean
        
        # Step 8: Analyze quality after cleaning
        quality_after = self.analyze_data_quality()
        
        # Compile final results
        pipeline_results = {
            'quality_before_cleaning': quality_before,
            'quality_after_cleaning': quality_after,
            'cleaning_report': self.cleaning_report,
            'consistency_validation': consistency_results,
            'image_analysis': image_stats,
            'pipeline_completed': True,
            'cleaned_data_path': str(self.output_path)
        }
        
        # Save results to JSON
        with open(self.output_path / "cleaning_results.json", 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            json_results = self._convert_datetime_for_json(pipeline_results)
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info("Data cleaning pipeline completed successfully!")
        logger.info(f"Cleaned data saved to: {self.output_path}")
        
        return pipeline_results
    
    def _convert_datetime_for_json(self, obj):
        """Convert datetime objects to strings for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_datetime_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_datetime_for_json(item) for item in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        else:
            return obj


def main():
    """Main execution function."""
    # Initialize pipeline
    data_path = "../dataset"
    pipeline = DataCleaningPipeline(data_path)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    print("\n" + "="*50)
    print("DATA CLEANING PIPELINE COMPLETED")
    print("="*50)
    print(f"Results saved to: {pipeline.output_path}")
    print("\nKey Statistics:")
    print(f"- Transactions processed: {results['cleaning_report']['transactions']['final_rows']:,}")
    print(f"- Articles processed: {results['cleaning_report']['articles']['final_rows']:,}")
    print(f"- Customers processed: {results['cleaning_report']['customers']['final_rows']:,}")
    
    if results['image_analysis']['images_available']:
        print(f"- Images available: {results['image_analysis']['total_images']:,}")
        if 'image_coverage_percentage' in results['image_analysis']:
            print(f"- Image coverage: {results['image_analysis']['image_coverage_percentage']:.2f}%")


if __name__ == "__main__":
    main()