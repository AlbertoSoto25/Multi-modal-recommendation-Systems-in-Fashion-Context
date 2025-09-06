# Data Cleaning Pipeline for Fashion Recommender System

## Overview

This repository contains a comprehensive data cleaning and preprocessing pipeline for a multimodal fashion recommender system, developed as part of a Master's thesis. The pipeline handles large-scale e-commerce data including transactions, customer information, product metadata, and associated product images.

## Features

- **Scalable Data Processing**: Handles large datasets through chunked processing
- **Comprehensive Quality Assessment**: Multi-dimensional data quality analysis
- **Statistical Analysis**: In-depth statistical profiling and insights generation
- **Visualization Suite**: Static and interactive visualizations for data exploration
- **Multimodal Support**: Handles both structured data and image assets
- **Academic Standards**: Methodology designed for research reproducibility

## Project Structure

```
limpieza_de_datos/
├── data_cleaning_pipeline.py      # Main data cleaning pipeline
├── data_visualization_metrics.py  # Visualization and metrics module
├── main_pipeline.py               # Pipeline orchestration script
├── METHODOLOGY_DOCUMENTATION.md   # Academic methodology documentation
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── cleaned_data/                  # Output directory for cleaned data
└── visualizations/                # Output directory for visualizations
    ├── quality_plots/
    ├── exploratory_plots/
    ├── statistical_plots/
    └── interactive_plots/
```

## Installation

1. **Clone or download the repository**
   ```bash
   cd /path/to/your/project/limpieza_de_datos
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Data Requirements

The pipeline expects the following data structure:

```
dataset/
├── articles.csv          # Product metadata
├── customers.csv         # Customer information
├── transactions_train.csv # Transaction history
└── images/               # Product images
    ├── 010/
    │   ├── 0108775015.jpg
    │   └── ...
    ├── 011/
    └── ...
```

### Expected Data Formats

**transactions_train.csv**:
- `customer_id`: Unique customer identifier
- `article_id`: Unique product identifier  
- `t_dat`: Transaction date (YYYY-MM-DD)
- `price`: Transaction price
- `sales_channel_id`: Sales channel identifier

**articles.csv**:
- `article_id`: Unique product identifier
- Additional product metadata columns

**customers.csv**:
- `customer_id`: Unique customer identifier
- Additional customer demographic columns

## Usage

### Quick Start

Run the complete pipeline with default settings:

```bash
python main_pipeline.py
```

### Advanced Usage

```bash
python main_pipeline.py \
    --data_path /path/to/your/dataset \
    --output_path /path/to/cleaned/data \
    --viz_output /path/to/visualizations
```

### Pipeline Options

- `--data_path`: Path to raw data directory (default: `../dataset`)
- `--output_path`: Path for cleaned data output (default: `cleaned_data`)
- `--viz_output`: Path for visualization outputs (default: `visualizations`)
- `--skip_cleaning`: Skip data cleaning step (use existing cleaned data)
- `--skip_visualization`: Skip visualization step

### Individual Module Usage

**Data Cleaning Only**:
```python
from data_cleaning_pipeline import DataCleaningPipeline

pipeline = DataCleaningPipeline(data_path="../dataset")
results = pipeline.run_complete_pipeline()
```

**Visualization Only**:
```python
from data_visualization_metrics import DataVisualizationMetrics

viz_module = DataVisualizationMetrics(data_path="cleaned_data")
viz_results = viz_module.run_complete_visualization_pipeline()
```

## Output Files

### Cleaned Data
- `transactions_clean.csv`: Cleaned transaction data
- `articles_clean.csv`: Cleaned product metadata
- `customers_clean.csv`: Cleaned customer data
- `cleaning_results.json`: Detailed cleaning metrics and statistics

### Visualizations
- `data_quality_dashboard.png`: Comprehensive data quality overview
- `price_statistical_analysis.png`: Price distribution and statistical analysis
- `customer_behavior_analysis.png`: Customer behavior patterns
- `temporal_analysis.png`: Time-based patterns and trends
- `article_analysis.png`: Product performance analysis
- `interactive_dashboard.html`: Interactive Plotly dashboard

### Reports
- `comprehensive_analysis_report.md`: Detailed analysis report
- `statistical_analysis_results.json`: Statistical metrics and insights
- Execution logs with timestamps

## Key Features

### Data Cleaning Capabilities

1. **Missing Value Treatment**
   - Intelligent imputation strategies based on data type
   - Article-specific median imputation for prices
   - Category-aware missing value handling

2. **Outlier Detection and Treatment**
   - Statistical outlier detection (99.9th percentile capping)
   - Business logic validation
   - Preservation of legitimate high-value transactions

3. **Data Standardization**
   - Categorical value normalization
   - Date format standardization
   - Data type optimization for memory efficiency

4. **Quality Validation**
   - Referential integrity checks
   - Cross-dataset consistency validation
   - Business rule validation

### Statistical Analysis

1. **Descriptive Statistics**
   - Distribution analysis for all numerical variables
   - Temporal pattern identification
   - Customer behavior profiling

2. **Advanced Analytics**
   - RFM (Recency, Frequency, Monetary) analysis
   - Seasonal decomposition
   - Correlation analysis

3. **Quality Metrics**
   - Completeness, consistency, and accuracy scores
   - Data coverage analysis
   - Pipeline performance metrics

### Visualization Suite

1. **Quality Dashboards**
   - Missing value heatmaps
   - Distribution plots
   - Temporal trend analysis

2. **Statistical Plots**
   - Q-Q plots for normality testing
   - Box plots for group comparisons
   - Scatter plots for relationship analysis

3. **Interactive Visualizations**
   - Plotly-based interactive dashboards
   - Drill-down capabilities
   - Export functionality

## Performance Considerations

### Memory Management
- Chunked processing for large datasets
- Optimized data types to reduce memory footprint
- Explicit garbage collection in long-running processes

### Processing Speed
- Vectorized operations using pandas/NumPy
- Parallel processing where applicable
- Efficient algorithms for statistical computations

### Scalability
- Designed to handle datasets with millions of records
- Configurable chunk sizes for different hardware configurations
- Progress monitoring and logging

## Academic Usage

This pipeline is designed for academic research and includes:

- **Methodology Documentation**: Comprehensive methodology documentation suitable for academic papers
- **Reproducibility**: Fixed random seeds and deterministic operations
- **Statistical Rigor**: Proper statistical testing and validation
- **Citation Ready**: Proper documentation and references for academic use

### For Master's Thesis

The pipeline generates all necessary components for a master's thesis:
1. Methodology documentation
2. Statistical analysis results
3. Quality assessment reports
4. Visualization materials
5. Performance metrics

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce chunk size in `data_cleaning_pipeline.py`
   - Ensure sufficient system memory (recommend 8GB+)

2. **File Path Issues**
   - Use absolute paths when possible
   - Ensure proper directory structure

3. **Missing Dependencies**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

4. **Large Dataset Processing**
   - Monitor system resources during execution
   - Consider processing subsets for initial testing

### Performance Optimization

1. **For Large Datasets**
   - Increase chunk size if memory allows
   - Use SSD storage for faster I/O
   - Close other applications to free memory

2. **For Faster Processing**
   - Use parallel processing where available
   - Optimize data types before processing
   - Cache intermediate results

## Contributing

This pipeline is part of academic research. For improvements or extensions:

1. Follow the existing code structure and documentation standards
2. Include comprehensive testing for new features
3. Update methodology documentation for significant changes
4. Maintain academic rigor in statistical methods

## License

This project is developed for academic purposes as part of a Master's thesis. Please cite appropriately if using in academic work.

## Contact

For questions or issues related to this pipeline, please refer to the methodology documentation or contact the development team.

---

**Note**: This pipeline is designed for academic research and may require adaptation for production environments. Always validate results with domain experts before making business decisions based on the cleaned data.