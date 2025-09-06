"""
Main Pipeline Execution for Fashion Recommender System Data Cleaning
===================================================================

This script orchestrates the complete data cleaning and analysis pipeline
for the multimodal fashion recommender system.

Author: Data Science Team
Purpose: Master's Thesis - Multimodal Recommender System
"""

import sys
import os
from pathlib import Path
import logging
import argparse
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_cleaning_pipeline import DataCleaningPipeline
from data_visualization_metrics import DataVisualizationMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_execution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function for the complete pipeline."""
    
    parser = argparse.ArgumentParser(description='Fashion Recommender Data Cleaning Pipeline')
    parser.add_argument('--data_path', type=str, default='../dataset', 
                       help='Path to raw data directory')
    parser.add_argument('--output_path', type=str, default='cleaned_data',
                       help='Path for cleaned data output')
    parser.add_argument('--viz_output', type=str, default='visualizations',
                       help='Path for visualization outputs')
    parser.add_argument('--skip_cleaning', action='store_true',
                       help='Skip data cleaning step (use existing cleaned data)')
    parser.add_argument('--skip_visualization', action='store_true',
                       help='Skip visualization step')
    parser.add_argument('--viz_cleaning_only', action='store_true',
                       help='Generate only cleaning-effects visuals from cleaning_results.json')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FASHION RECOMMENDER SYSTEM - DATA CLEANING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Visualization path: {args.viz_output}")
    print("="*80)
    
    try:
        # Step 1: Data Cleaning Pipeline
        if not args.skip_cleaning:
            logger.info("Starting data cleaning pipeline...")
            
            cleaning_pipeline = DataCleaningPipeline(
                data_path=args.data_path,
                output_path=args.output_path
            )
            
            cleaning_results = cleaning_pipeline.run_complete_pipeline()
            
            print("\n" + "="*50)
            print("DATA CLEANING COMPLETED")
            print("="*50)
            print("Key Results:")
            print(f"- Transactions: {cleaning_results['cleaning_report']['transactions']['final_rows']:,} rows")
            print(f"- Articles: {cleaning_results['cleaning_report']['articles']['final_rows']:,} rows") 
            print(f"- Customers: {cleaning_results['cleaning_report']['customers']['final_rows']:,} rows")
            
            if cleaning_results['image_analysis']['images_available']:
                print(f"- Images: {cleaning_results['image_analysis']['total_images']:,} available")
                if 'image_coverage_percentage' in cleaning_results['image_analysis']:
                    print(f"- Image coverage: {cleaning_results['image_analysis']['image_coverage_percentage']:.1f}%")
        
        else:
            logger.info("Skipping data cleaning step (using existing cleaned data)")
        
        # Step 2: Visualization and Metrics Pipeline
        if not args.skip_visualization:
            logger.info("Starting visualization and metrics pipeline...")
            
            viz_pipeline = DataVisualizationMetrics(
                data_path=args.output_path,
                output_path=args.viz_output
            )
            
            if args.viz_cleaning_only:
                viz_pipeline.run_cleaning_results_visuals_only()
                viz_results = {}
            else:
                viz_results = viz_pipeline.run_complete_visualization_pipeline()
            
            print("\n" + "="*50)
            print("VISUALIZATION AND METRICS COMPLETED")
            print("="*50)
            print("Generated Outputs:")
            if args.viz_cleaning_only:
                print("- Cleaning effects dashboard (before vs after)")
                print(f"- Files saved to: {args.viz_output}/quality_plots/")
            else:
                print("- Data quality dashboard")
                print("- Statistical analysis plots")
                print("- Interactive dashboard")
                print("- Comprehensive analysis report")
                print(f"- All files saved to: {args.viz_output}/")
        
        else:
            logger.info("Skipping visualization step")
        
        # Final Summary
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nNext Steps for Master's Thesis:")
        print("1. Review the comprehensive analysis report")
        print("2. Examine data quality visualizations")
        print("3. Use insights for model architecture design")
        print("4. Implement multimodal feature extraction")
        print("5. Develop recommendation algorithms")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\nERROR: Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()