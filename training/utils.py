"""
Utilidades para el sistema de recomendación multimodal
"""
import os
import torch
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def load_model_from_checkpoint(checkpoint_path: str, config_path: str = None):
    """
    Cargar modelo desde checkpoint
    
    Args:
        checkpoint_path: Ruta al checkpoint
        config_path: Ruta al archivo de configuración (opcional)
    
    Returns:
        model: Modelo cargado
        config: Configuración del modelo
    """
    from model import MultiModalFashionRecommender
    from config import Config
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Cargar configuración
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = Config(**config_dict)
    elif 'config' in checkpoint:
        config = Config(**checkpoint['config'])
    else:
        logger.warning("No se encontró configuración, usando valores por defecto")
        config = Config()
    
    # Crear y cargar modelo
    model = MultiModalFashionRecommender(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Modelo cargado desde {checkpoint_path}")
    logger.info(f"Época: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"Paso: {checkpoint.get('global_step', 'N/A')}")
    logger.info(f"Mejor métrica: {checkpoint.get('best_metric', 'N/A')}")
    
    return model, config

def analyze_training_logs(log_path: str) -> Dict[str, List]:
    """
    Analizar logs de entrenamiento
    
    Args:
        log_path: Ruta al archivo de log
        
    Returns:
        Dictionary con métricas extraídas
    """
    metrics = {
        'steps': [],
        'train_loss': [],
        'eval_metrics': {},
        'timestamps': []
    }
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Extraer timestamp
                if ' - ' in line:
                    timestamp_str = line.split(' - ')[0]
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        metrics['timestamps'].append(timestamp)
                    except:
                        continue
                
                # Extraer métricas de entrenamiento
                if 'train/loss:' in line:
                    try:
                        loss_value = float(line.split('train/loss: ')[1].split()[0])
                        metrics['train_loss'].append(loss_value)
                    except:
                        continue
                
                # Extraer métricas de evaluación
                if 'eval/' in line:
                    try:
                        parts = line.split('eval/')[1].split(': ')
                        if len(parts) == 2:
                            metric_name = parts[0]
                            metric_value = float(parts[1].split()[0])
                            
                            if metric_name not in metrics['eval_metrics']:
                                metrics['eval_metrics'][metric_name] = []
                            metrics['eval_metrics'][metric_name].append(metric_value)
                    except:
                        continue
                
                # Extraer pasos
                if 'train/step:' in line:
                    try:
                        step_value = int(float(line.split('train/step: ')[1].split()[0]))
                        metrics['steps'].append(step_value)
                    except:
                        continue
    
    except FileNotFoundError:
        logger.error(f"Archivo de log no encontrado: {log_path}")
    
    return metrics

def plot_training_metrics(log_path: str, save_path: str = None):
    """
    Visualizar métricas de entrenamiento
    
    Args:
        log_path: Ruta al archivo de log
        save_path: Ruta para guardar la figura (opcional)
    """
    metrics = analyze_training_logs(log_path)
    
    if not metrics['train_loss']:
        logger.warning("No se encontraron métricas de entrenamiento")
        return
    
    # Configurar el plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Métricas de Entrenamiento', fontsize=16)
    
    # Plot 1: Training Loss
    if metrics['train_loss']:
        axes[0, 0].plot(metrics['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
    
    # Plot 2: Evaluation Metrics
    if metrics['eval_metrics']:
        eval_metrics = metrics['eval_metrics']
        
        # Seleccionar métricas principales
        main_metrics = ['precision@10', 'recall@10', 'ndcg@10', 'hit_rate@10']
        available_metrics = [m for m in main_metrics if m in eval_metrics]
        
        if available_metrics:
            for i, metric in enumerate(available_metrics[:4]):
                row = i // 2
                col = 1 if i % 2 else 1
                
                if i < 2:
                    axes[0, 1].plot(eval_metrics[metric], label=metric)
                else:
                    axes[1, 1].plot(eval_metrics[metric], label=metric)
        
        axes[0, 1].set_title('Precision & Recall')
        axes[0, 1].set_xlabel('Evaluation Step')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 1].set_title('NDCG & Hit Rate')
        axes[1, 1].set_xlabel('Evaluation Step')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Plot 3: Learning Rate (si está disponible)
    axes[1, 0].set_title('Métricas Adicionales')
    axes[1, 0].text(0.5, 0.5, 'Ver logs para más detalles', 
                    ha='center', va='center', transform=axes[1, 0].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en {save_path}")
    
    plt.show()

def analyze_dataset_statistics(data_dir: str) -> Dict[str, Any]:
    """
    Analizar estadísticas del dataset
    
    Args:
        data_dir: Directorio con los datos
        
    Returns:
        Diccionario con estadísticas
    """
    stats = {}
    
    # Analizar transacciones
    try:
        transactions_path = os.path.join(data_dir, "transactions_clean.csv")
        logger.info("Analizando transacciones...")
        
        # Leer en chunks para datasets grandes
        chunk_size = 100000
        total_transactions = 0
        unique_customers = set()
        unique_articles = set()
        
        for chunk in pd.read_csv(transactions_path, chunksize=chunk_size):
            total_transactions += len(chunk)
            unique_customers.update(chunk['customer_id'].unique())
            unique_articles.update(chunk['article_id'].astype(str).unique())
        
        stats['transactions'] = {
            'total_transactions': total_transactions,
            'unique_customers': len(unique_customers),
            'unique_articles': len(unique_articles),
            'avg_transactions_per_customer': total_transactions / len(unique_customers)
        }
        
    except Exception as e:
        logger.error(f"Error analizando transacciones: {e}")
    
    # Analizar artículos
    try:
        articles_path = os.path.join(data_dir, "articles_clean.csv")
        articles_df = pd.read_csv(articles_path)
        
        stats['articles'] = {
            'total_articles': len(articles_df),
            'product_types': articles_df['product_type_name'].nunique() if 'product_type_name' in articles_df.columns else 0,
            'departments': articles_df['department_name'].nunique() if 'department_name' in articles_df.columns else 0,
            'color_groups': articles_df['colour_group_name'].nunique() if 'colour_group_name' in articles_df.columns else 0
        }
        
    except Exception as e:
        logger.error(f"Error analizando artículos: {e}")
    
    # Analizar imágenes
    try:
        images_dir = os.path.join(data_dir, "normalized_images")
        if os.path.exists(images_dir):
            # Check if images are in subfolders (old structure) or single directory (new structure)
            jpg_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
            
            if jpg_files:
                # New structure: all images in single directory
                total_images = len(jpg_files)
                folders = 0
            else:
                # Old structure: images in subfolders
                total_images = 0
                for folder in os.listdir(images_dir):
                    folder_path = os.path.join(images_dir, folder)
                    if os.path.isdir(folder_path):
                        total_images += len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
                folders = len([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))])
            
            stats['images'] = {
                'total_images': total_images,
                'folders': folders,
                'structure': 'single_directory' if jpg_files else 'subfolders'
            }
    
    except Exception as e:
        logger.error(f"Error analizando imágenes: {e}")
    
    return stats

def print_dataset_report(data_dir: str):
    """
    Imprimir reporte del dataset
    
    Args:
        data_dir: Directorio con los datos
    """
    stats = analyze_dataset_statistics(data_dir)
    
    print("=" * 60)
    print("REPORTE DEL DATASET")
    print("=" * 60)
    
    if 'transactions' in stats:
        print("\n📊 TRANSACCIONES:")
        print(f"  Total de transacciones: {stats['transactions']['total_transactions']:,}")
        print(f"  Clientes únicos: {stats['transactions']['unique_customers']:,}")
        print(f"  Artículos únicos: {stats['transactions']['unique_articles']:,}")
        print(f"  Transacciones por cliente: {stats['transactions']['avg_transactions_per_customer']:.1f}")
    
    if 'articles' in stats:
        print("\n🛍️  ARTÍCULOS:")
        print(f"  Total de artículos: {stats['articles']['total_articles']:,}")
        print(f"  Tipos de producto: {stats['articles']['product_types']:,}")
        print(f"  Departamentos: {stats['articles']['departments']:,}")
        print(f"  Grupos de color: {stats['articles']['color_groups']:,}")
    
    if 'images' in stats:
        print("\n🖼️  IMÁGENES:")
        print(f"  Total de imágenes: {stats['images']['total_images']:,}")
        print(f"  Carpetas de imágenes: {stats['images']['folders']:,}")
    
    print("\n" + "=" * 60)

def estimate_training_time(config, num_samples: int = 1000000) -> str:
    """
    Estimar tiempo de entrenamiento
    
    Args:
        config: Configuración del modelo
        num_samples: Número estimado de muestras
        
    Returns:
        Estimación de tiempo como string
    """
    # Estimaciones basadas en benchmarks típicos
    samples_per_second = {
        'T4': 50,      # GPU T4
        'P100': 80,    # GPU P100  
        'V100': 120,   # GPU V100
        'A100': 200,   # GPU A100
        'CPU': 5       # CPU only
    }
    
    # Detectar tipo de GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if 'T4' in gpu_name:
            speed = samples_per_second['T4']
        elif 'P100' in gpu_name:
            speed = samples_per_second['P100']
        elif 'V100' in gpu_name:
            speed = samples_per_second['V100']
        elif 'A100' in gpu_name:
            speed = samples_per_second['A100']
        else:
            speed = samples_per_second['T4']  # Conservative estimate
    else:
        speed = samples_per_second['CPU']
    
    # Ajustar por configuración
    speed = speed * (config.data.batch_size / 32)  # Normalizar por batch size
    speed = speed / config.training.gradient_accumulation_steps  # Ajustar por accumulation
    
    # Calcular tiempo
    samples_per_epoch = num_samples
    seconds_per_epoch = samples_per_epoch / speed
    total_seconds = seconds_per_epoch * config.training.max_epochs
    
    # Convertir a formato legible
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    return f"Estimación: {hours}h {minutes}m por {config.training.max_epochs} épocas"

def check_system_requirements():
    """Verificar requisitos del sistema"""
    print("🔍 VERIFICACIÓN DEL SISTEMA")
    print("=" * 40)
    
    # Python version
    import sys
    print(f"Python: {sys.version}")
    
    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA: No disponible")
    
    # RAM
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB disponible")
    
    # Disk space
    disk = psutil.disk_usage('/')
    print(f"Disco: {disk.free / 1024**3:.1f} GB libres de {disk.total / 1024**3:.1f} GB")
    
    print("=" * 40)
