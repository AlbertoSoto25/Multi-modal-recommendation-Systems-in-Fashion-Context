#!/usr/bin/env python3
"""
Análisis Exploratorio de Datos para Sistema Recomendador Multimodal
Trabajo Fin de Máster - H&M Personalized Fashion Recommendations

Este script realiza un análisis exhaustivo de los datos incluyendo:
- Análisis de estructura de datos
- Métricas descriptivas
- Análisis temporal
- Patrones de comportamiento
- Visualizaciones comprehensivas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Configuración
warnings.filterwarnings('ignore')
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Configurar directorio de trabajo
DATA_DIR = Path("dataset")
IMAGES_DIR = DATA_DIR / "images"

class FashionRecommenderEDA:
    """Clase principal para el análisis exploratorio de datos"""
    
    def __init__(self):
        self.articles = None
        self.customers = None
        self.transactions = None
        self.image_stats = None
        
    def load_data(self):
        """Carga todos los datasets"""
        print("🔄 Cargando datasets...")
        
        # Cargar articles.csv
        print("📦 Cargando metadatos de artículos...")
        self.articles = pd.read_csv(DATA_DIR / "articles.csv")
        print(f"   ✅ Articles: {self.articles.shape[0]:,} artículos, {self.articles.shape[1]} columnas")
        
        # Cargar customers.csv
        print("👥 Cargando metadatos de clientes...")
        self.customers = pd.read_csv(DATA_DIR / "customers.csv")
        print(f"   ✅ Customers: {self.customers.shape[0]:,} clientes, {self.customers.shape[1]} columnas")
        
        # Cargar transactions_train.csv (sample para análisis rápido)
        print("💳 Cargando muestra de transacciones para análisis...")
        # Usar una muestra para análisis exploratorio inicial
        sample_size = 5000000  # 5M rows sample
        self.transactions = pd.read_csv(DATA_DIR / "transactions_train.csv", nrows=sample_size)
        print(f"   📊 Muestra cargada para análisis exploratorio")
        print(f"   ✅ Transactions: {self.transactions.shape[0]:,} transacciones, {self.transactions.shape[1]} columnas")
        
    def analyze_data_quality(self):
        """Análisis de calidad de datos"""
        print("\n" + "="*60)
        print("📊 ANÁLISIS DE CALIDAD DE DATOS")
        print("="*60)
        
        datasets = {
            'Articles': self.articles,
            'Customers': self.customers, 
            'Transactions': self.transactions
        }
        
        for name, df in datasets.items():
            print(f"\n🔍 {name.upper()}:")
            print(f"   Forma: {df.shape}")
            print(f"   Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Valores faltantes
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            missing_df = pd.DataFrame({
                'Missing Count': missing,
                'Missing %': missing_pct
            }).sort_values('Missing %', ascending=False)
            
            print("   Valores faltantes:")
            for col, row in missing_df[missing_df['Missing Count'] > 0].iterrows():
                print(f"     {col}: {row['Missing Count']:,} ({row['Missing %']:.2f}%)")
            
            # Tipos de datos
            print("   Tipos de datos:")
            type_counts = df.dtypes.value_counts()
            for dtype, count in type_counts.items():
                print(f"     {dtype}: {count} columnas")
                
    def analyze_articles(self):
        """Análisis detallado de artículos"""
        print("\n" + "="*60)
        print("👕 ANÁLISIS DE ARTÍCULOS")
        print("="*60)
        
        # Información básica
        print(f"\n📈 ESTADÍSTICAS BÁSICAS:")
        print(f"   Total de artículos: {len(self.articles):,}")
        print(f"   Artículos únicos: {self.articles['article_id'].nunique():,}")
        
        # Análisis por categorías
        categorical_cols = self.articles.select_dtypes(include=['object']).columns
        
        print(f"\n🏷️ ANÁLISIS POR CATEGORÍAS:")
        for col in categorical_cols[:8]:  # Top 8 categorías más importantes
            if col != 'detail_desc':  # Evitar descripciones muy largas
                unique_count = self.articles[col].nunique()
                print(f"   {col}: {unique_count:,} valores únicos")
                
                # Top 5 valores más frecuentes
                top_values = self.articles[col].value_counts().head()
                print(f"     Top 5:")
                for val, count in top_values.items():
                    pct = (count / len(self.articles)) * 100
                    print(f"       {val}: {count:,} ({pct:.1f}%)")
                print()
        
        # Análisis de precios
        if 'price' in self.articles.columns:
            print(f"💰 ANÁLISIS DE PRECIOS:")
            price_stats = self.articles['price'].describe()
            print(f"   Estadísticas de precios:")
            for stat, value in price_stats.items():
                print(f"     {stat.capitalize()}: {value:.2f}")
                
    def analyze_customers(self):
        """Análisis detallado de clientes"""
        print("\n" + "="*60)
        print("👥 ANÁLISIS DE CLIENTES")
        print("="*60)
        
        print(f"\n📈 ESTADÍSTICAS BÁSICAS:")
        print(f"   Total de clientes: {len(self.customers):,}")
        print(f"   Clientes únicos: {self.customers['customer_id'].nunique():,}")
        
        # Análisis demográfico
        if 'age' in self.customers.columns:
            print(f"\n🎂 ANÁLISIS DE EDAD:")
            age_stats = self.customers['age'].describe()
            for stat, value in age_stats.items():
                print(f"     {stat.capitalize()}: {value:.1f}")
                
            # Distribución por grupos de edad
            self.customers['age_group'] = pd.cut(self.customers['age'], 
                                               bins=[0, 25, 35, 45, 55, 65, 100],
                                               labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
            age_dist = self.customers['age_group'].value_counts().sort_index()
            print(f"     Distribución por grupos:")
            for group, count in age_dist.items():
                pct = (count / len(self.customers)) * 100
                print(f"       {group}: {count:,} ({pct:.1f}%)")
        
        # Análisis por otras variables categóricas
        categorical_cols = self.customers.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.customers[col].nunique() < 20:  # Solo mostrar categorías con pocos valores
                print(f"\n🏷️ {col.upper()}:")
                dist = self.customers[col].value_counts()
                for val, count in dist.items():
                    pct = (count / len(self.customers)) * 100
                    print(f"     {val}: {count:,} ({pct:.1f}%)")
                    
    def analyze_transactions(self):
        """Análisis detallado de transacciones"""
        print("\n" + "="*60)
        print("💳 ANÁLISIS DE TRANSACCIONES")
        print("="*60)
        
        # Convertir fecha
        self.transactions['t_dat'] = pd.to_datetime(self.transactions['t_dat'])
        
        print(f"\n📈 ESTADÍSTICAS BÁSICAS:")
        print(f"   Total de transacciones: {len(self.transactions):,}")
        print(f"   Clientes únicos: {self.transactions['customer_id'].nunique():,}")
        print(f"   Artículos únicos: {self.transactions['article_id'].nunique():,}")
        
        # Período de tiempo
        date_range = self.transactions['t_dat']
        print(f"   Período: {date_range.min().date()} a {date_range.max().date()}")
        print(f"   Duración: {(date_range.max() - date_range.min()).days} días")
        
        # Análisis de precio
        if 'price' in self.transactions.columns:
            print(f"\n💰 ANÁLISIS DE PRECIOS:")
            price_stats = self.transactions['price'].describe()
            for stat, value in price_stats.items():
                print(f"     {stat.capitalize()}: {value:.2f}")
                
            # Revenue total
            total_revenue = self.transactions['price'].sum()
            print(f"     Revenue total: {total_revenue:,.2f}")
            
        # Análisis temporal
        print(f"\n📅 ANÁLISIS TEMPORAL:")
        
        # Transacciones por mes
        monthly_trans = self.transactions.groupby(self.transactions['t_dat'].dt.to_period('M')).size()
        print(f"     Transacciones por mes (últimos 6):")
        for period, count in monthly_trans.tail(6).items():
            print(f"       {period}: {count:,}")
            
        # Día de la semana
        weekday_trans = self.transactions.groupby(self.transactions['t_dat'].dt.day_name()).size()
        print(f"     Transacciones por día de la semana:")
        for day, count in weekday_trans.items():
            pct = (count / len(self.transactions)) * 100
            print(f"       {day}: {count:,} ({pct:.1f}%)")
            
    def analyze_customer_behavior(self):
        """Análisis de comportamiento de clientes"""
        print("\n" + "="*60)
        print("🛍️ ANÁLISIS DE COMPORTAMIENTO DE CLIENTES")
        print("="*60)
        
        # Análisis por cliente
        customer_stats = self.transactions.groupby('customer_id').agg({
            'article_id': 'count',  # Número de compras
            'price': ['sum', 'mean'],  # Gasto total y promedio
            't_dat': ['min', 'max']  # Primera y última compra
        }).round(2)
        
        customer_stats.columns = ['num_purchases', 'total_spent', 'avg_price', 'first_purchase', 'last_purchase']
        
        # Calcular días activos
        customer_stats['days_active'] = (customer_stats['last_purchase'] - customer_stats['first_purchase']).dt.days
        
        print(f"\n📊 ESTADÍSTICAS POR CLIENTE:")
        print("   Número de compras por cliente:")
        purchases_stats = customer_stats['num_purchases'].describe()
        for stat, value in purchases_stats.items():
            print(f"     {stat.capitalize()}: {value:.1f}")
            
        print("   Gasto total por cliente:")
        spending_stats = customer_stats['total_spent'].describe()
        for stat, value in spending_stats.items():
            print(f"     {stat.capitalize()}: {value:.2f}")
            
        # Segmentación de clientes
        print(f"\n🎯 SEGMENTACIÓN DE CLIENTES:")
        
        # Por frecuencia de compra
        customer_stats['frequency_segment'] = pd.cut(customer_stats['num_purchases'],
                                                   bins=[0, 1, 5, 20, float('inf')],
                                                   labels=['One-time', 'Occasional', 'Regular', 'Frequent'])
        
        freq_dist = customer_stats['frequency_segment'].value_counts()
        print("   Por frecuencia de compra:")
        for segment, count in freq_dist.items():
            pct = (count / len(customer_stats)) * 100
            print(f"     {segment}: {count:,} ({pct:.1f}%)")
            
        # Por valor monetario
        customer_stats['value_segment'] = pd.qcut(customer_stats['total_spent'],
                                                q=4,
                                                labels=['Low', 'Medium', 'High', 'Premium'])
        
        value_dist = customer_stats['value_segment'].value_counts()
        print("   Por valor monetario:")
        for segment, count in value_dist.items():
            pct = (count / len(customer_stats)) * 100
            print(f"     {segment}: {count:,} ({pct:.1f}%)")
            
        return customer_stats
        
    def analyze_product_popularity(self):
        """Análisis de popularidad de productos"""
        print("\n" + "="*60)
        print("🔥 ANÁLISIS DE POPULARIDAD DE PRODUCTOS")
        print("="*60)
        
        # Análisis por artículo
        product_stats = self.transactions.groupby('article_id').agg({
            'customer_id': 'nunique',  # Clientes únicos
            'article_id': 'count',     # Número de ventas
            'price': ['sum', 'mean']   # Revenue y precio promedio
        }).round(2)
        
        product_stats.columns = ['unique_customers', 'total_sales', 'total_revenue', 'avg_price']
        
        print(f"\n📊 ESTADÍSTICAS POR PRODUCTO:")
        print("   Ventas por producto:")
        sales_stats = product_stats['total_sales'].describe()
        for stat, value in sales_stats.items():
            print(f"     {stat.capitalize()}: {value:.1f}")
            
        # Top productos
        print(f"\n🏆 TOP 10 PRODUCTOS MÁS VENDIDOS:")
        top_products = product_stats.sort_values('total_sales', ascending=False).head(10)
        for i, (article_id, row) in enumerate(top_products.iterrows(), 1):
            print(f"   {i:2d}. Artículo {article_id}: {row['total_sales']:,} ventas, {row['unique_customers']:,} clientes")
            
        print(f"\n💰 TOP 10 PRODUCTOS POR REVENUE:")
        top_revenue = product_stats.sort_values('total_revenue', ascending=False).head(10)
        for i, (article_id, row) in enumerate(top_revenue.iterrows(), 1):
            print(f"   {i:2d}. Artículo {article_id}: €{row['total_revenue']:,.2f}, {row['total_sales']:,} ventas")
            
        return product_stats
        
    def analyze_images(self):
        """Análisis de disponibilidad de imágenes"""
        print("\n" + "="*60)
        print("🖼️ ANÁLISIS DE IMÁGENES")
        print("="*60)
        
        # Contar imágenes por directorio
        image_count = 0
        dir_stats = {}
        
        print("🔍 Explorando directorios de imágenes...")
        
        for subdir in IMAGES_DIR.iterdir():
            if subdir.is_dir():
                images_in_dir = len(list(subdir.glob("*.jpg")))
                dir_stats[subdir.name] = images_in_dir
                image_count += images_in_dir
                
        print(f"\n📊 ESTADÍSTICAS DE IMÁGENES:")
        print(f"   Total de imágenes: {image_count:,}")
        print(f"   Directorios: {len(dir_stats)}")
        print(f"   Promedio por directorio: {image_count / len(dir_stats):.1f}")
        
        # Artículos con/sin imagen (muestra para estimación)
        articles_with_images = set()
        sample_articles = self.articles['article_id'].sample(min(10000, len(self.articles)))  # Muestra de 10k artículos
        for article_id in sample_articles:
            article_str = str(article_id).zfill(10)
            subdir = article_str[:3]
            image_path = IMAGES_DIR / subdir / f"{article_str}.jpg"
            if image_path.exists():
                articles_with_images.add(article_id)
                
        # Extrapolar resultado a toda la población
        sample_coverage = len(articles_with_images) / len(sample_articles) * 100
        estimated_articles_with_images = int(len(self.articles) * sample_coverage / 100)
                
        print(f"\n📈 COBERTURA DE IMÁGENES (ESTIMACIÓN):")
        print(f"   Muestra analizada: {len(sample_articles):,} artículos")
        print(f"   Artículos con imagen (muestra): {len(articles_with_images):,} ({sample_coverage:.1f}%)")
        print(f"   Estimación total con imagen: {estimated_articles_with_images:,}")
        print(f"   Estimación sin imagen: {len(self.articles) - estimated_articles_with_images:,}")
        
        self.image_stats = {
            'total_images': image_count,
            'articles_with_images': estimated_articles_with_images,
            'coverage_pct': sample_coverage
        }
        
        return dir_stats
        
    def create_visualizations(self):
        """Crear visualizaciones comprehensivas"""
        print("\n" + "="*60)
        print("📈 CREANDO VISUALIZACIONES")
        print("="*60)
        
        # Configurar subplot
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Transacciones por Mes',
                'Distribución de Precios',
                'Top 15 Categorías de Productos',
                'Distribución de Edad de Clientes',
                'Transacciones por Día de la Semana',
                'Segmentación de Clientes por Frecuencia',
                'Revenue Mensual',
                'Cobertura de Imágenes'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # 1. Transacciones por mes
        monthly_trans = self.transactions.groupby(self.transactions['t_dat'].dt.to_period('M')).size()
        fig.add_trace(
            go.Scatter(x=[str(p) for p in monthly_trans.index], y=monthly_trans.values,
                      mode='lines+markers', name='Transacciones'),
            row=1, col=1
        )
        
        # 2. Distribución de precios
        fig.add_trace(
            go.Histogram(x=self.transactions['price'], nbinsx=50, name='Precios'),
            row=1, col=2
        )
        
        # 3. Top categorías (usando product_group_name si existe)
        if 'product_group_name' in self.articles.columns:
            top_categories = self.articles['product_group_name'].value_counts().head(15)
            fig.add_trace(
                go.Bar(x=top_categories.values, y=top_categories.index, 
                      orientation='h', name='Categorías'),
                row=2, col=1
            )
        
        # 4. Distribución de edad
        if 'age' in self.customers.columns:
            fig.add_trace(
                go.Histogram(x=self.customers['age'], nbinsx=30, name='Edad'),
                row=2, col=2
            )
        
        # 5. Transacciones por día de la semana
        weekday_trans = self.transactions.groupby(self.transactions['t_dat'].dt.day_name()).size()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_ordered = weekday_trans.reindex(days_order)
        
        fig.add_trace(
            go.Bar(x=days_order, y=weekday_ordered.values, name='Día semana'),
            row=3, col=1
        )
        
        # 6. Segmentación por frecuencia
        customer_stats = self.transactions.groupby('customer_id').size()
        freq_segments = pd.cut(customer_stats, bins=[0, 1, 5, 20, float('inf')],
                              labels=['One-time', 'Occasional', 'Regular', 'Frequent'])
        freq_dist = freq_segments.value_counts()
        
        fig.add_trace(
            go.Pie(labels=freq_dist.index, values=freq_dist.values, name='Segmentación'),
            row=3, col=2
        )
        
        # 7. Revenue mensual
        monthly_revenue = self.transactions.groupby(self.transactions['t_dat'].dt.to_period('M'))['price'].sum()
        fig.add_trace(
            go.Scatter(x=[str(p) for p in monthly_revenue.index], y=monthly_revenue.values,
                      mode='lines+markers', name='Revenue'),
            row=4, col=1
        )
        
        # 8. Cobertura de imágenes
        if self.image_stats:
            coverage_data = ['Con imagen', 'Sin imagen']
            coverage_values = [self.image_stats['articles_with_images'], 
                             len(self.articles) - self.image_stats['articles_with_images']]
            
            fig.add_trace(
                go.Pie(labels=coverage_data, values=coverage_values, name='Imágenes'),
                row=4, col=2
            )
        
        # Actualizar layout
        fig.update_layout(
            height=1600,
            title_text="Dashboard de Análisis Exploratorio - Sistema Recomendador H&M",
            showlegend=False
        )
        
        # Guardar visualización
        fig.write_html("fashion_recommender_eda_dashboard.html")
        print("   ✅ Dashboard guardado como 'fashion_recommender_eda_dashboard.html'")
        
        # Crear visualizaciones adicionales con matplotlib
        self._create_matplotlib_plots()
        
    def _create_matplotlib_plots(self):
        """Crear plots adicionales con matplotlib"""
        
        # 1. Heatmap de correlaciones (si hay datos numéricos suficientes)
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Evolución temporal detallada
        plt.subplot(2, 3, 1)
        daily_trans = self.transactions.groupby('t_dat').size()
        daily_trans.plot(kind='line', alpha=0.7)
        plt.title('Evolución Diaria de Transacciones')
        plt.xticks(rotation=45)
        plt.ylabel('Número de Transacciones')
        
        # Subplot 2: Distribución logarítmica de precios
        plt.subplot(2, 3, 2)
        plt.hist(np.log1p(self.transactions['price']), bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribución Logarítmica de Precios')
        plt.xlabel('Log(Precio + 1)')
        plt.ylabel('Frecuencia')
        
        # Subplot 3: Top productos más vendidos
        plt.subplot(2, 3, 3)
        product_sales = self.transactions['article_id'].value_counts().head(20)
        product_sales.plot(kind='bar')
        plt.title('Top 20 Productos Más Vendidos')
        plt.xlabel('Article ID')
        plt.ylabel('Ventas')
        plt.xticks(rotation=90)
        
        # Subplot 4: Distribución de compras por cliente
        plt.subplot(2, 3, 4)
        customer_purchases = self.transactions.groupby('customer_id').size()
        plt.hist(customer_purchases, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribución de Compras por Cliente')
        plt.xlabel('Número de Compras')
        plt.ylabel('Número de Clientes')
        plt.yscale('log')
        
        # Subplot 5: Análisis estacional
        plt.subplot(2, 3, 5)
        seasonal_trans = self.transactions.groupby(self.transactions['t_dat'].dt.month).size()
        months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        plt.bar(range(1, 13), seasonal_trans.reindex(range(1, 13), fill_value=0))
        plt.title('Estacionalidad de Compras')
        plt.xlabel('Mes')
        plt.ylabel('Transacciones')
        plt.xticks(range(1, 13), months, rotation=45)
        
        # Subplot 6: Distribución de revenue por transacción
        plt.subplot(2, 3, 6)
        plt.boxplot(self.transactions['price'])
        plt.title('Distribución de Precios (Boxplot)')
        plt.ylabel('Precio')
        
        plt.tight_layout()
        plt.savefig('fashion_recommender_detailed_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✅ Gráficos detallados guardados como 'fashion_recommender_detailed_analysis.png'")
        
    def generate_summary_report(self):
        """Generar reporte resumen con insights clave"""
        print("\n" + "="*80)
        print("📋 REPORTE RESUMEN - INSIGHTS CLAVE PARA SISTEMA RECOMENDADOR")
        print("="*80)
        
        # Métricas clave del dataset
        print(f"\n🎯 MÉTRICAS CLAVE DEL DATASET:")
        print(f"   • Total de transacciones: {len(self.transactions):,}")
        print(f"   • Clientes únicos: {self.transactions['customer_id'].nunique():,}")
        print(f"   • Productos únicos: {self.transactions['article_id'].nunique():,}")
        print(f"   • Revenue total: €{self.transactions['price'].sum():,.2f}")
        print(f"   • Período de datos: {(self.transactions['t_dat'].max() - self.transactions['t_dat'].min()).days} días")
        
        if self.image_stats:
            print(f"   • Cobertura de imágenes: {self.image_stats['coverage_pct']:.1f}%")
        
        # Insights de comportamiento
        print(f"\n🔍 INSIGHTS DE COMPORTAMIENTO:")
        
        # Análisis de clientes
        customer_purchases = self.transactions.groupby('customer_id').size()
        repeat_customers = (customer_purchases > 1).sum()
        repeat_rate = repeat_customers / len(customer_purchases) * 100
        
        print(f"   • Tasa de clientes recurrentes: {repeat_rate:.1f}%")
        print(f"   • Promedio de compras por cliente: {customer_purchases.mean():.1f}")
        print(f"   • Ticket promedio: €{self.transactions['price'].mean():.2f}")
        
        # Análisis de productos
        product_sales = self.transactions['article_id'].value_counts()
        products_80_percent = (product_sales.cumsum() / product_sales.sum() <= 0.8).sum()
        
        print(f"   • Productos que generan 80% de ventas: {products_80_percent:,} ({products_80_percent/len(product_sales)*100:.1f}%)")
        print(f"   • Productos con una sola venta: {(product_sales == 1).sum():,}")
        
        # Recomendaciones para el sistema recomendador
        print(f"\n💡 RECOMENDACIONES PARA EL SISTEMA RECOMENDADOR:")
        print(f"   🎯 ESTRATEGIAS DE RECOMENDACIÓN:")
        print(f"      • Implementar filtrado colaborativo para clientes recurrentes ({repeat_rate:.1f}%)")
        print(f"      • Usar filtrado basado en contenido para productos con pocas interacciones")
        print(f"      • Aprovechar datos multimodales (imágenes) para {self.image_stats['coverage_pct']:.1f}% de productos")
        print(f"      • Considerar estacionalidad en recomendaciones")
        
        print(f"\n   📊 CONSIDERACIONES TÉCNICAS:")
        print(f"      • Manejar sparsity: muchos productos tienen pocas ventas")
        print(f"      • Implementar cold-start para nuevos productos/usuarios")
        print(f"      • Usar técnicas de ensemble para combinar diferentes enfoques")
        print(f"      • Optimizar para diversidad vs precisión")
        
        print(f"\n   🔧 FEATURES RECOMENDADAS:")
        print(f"      • Historial de compras del usuario")
        print(f"      • Similitud visual entre productos (CNN)")
        print(f"      • Categorías y atributos de productos")
        print(f"      • Patrones temporales y estacionales")
        print(f"      • Popularidad y tendencias de productos")
        
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("🚀 INICIANDO ANÁLISIS EXPLORATORIO COMPLETO")
        print("="*80)
        
        # Cargar datos
        self.load_data()
        
        # Análisis de calidad
        self.analyze_data_quality()
        
        # Análisis específicos
        self.analyze_articles()
        self.analyze_customers()
        self.analyze_transactions()
        self.analyze_customer_behavior()
        self.analyze_product_popularity()
        self.analyze_images()
        
        # Visualizaciones
        self.create_visualizations()
        
        # Reporte final
        self.generate_summary_report()
        
        print(f"\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
        print(f"   Archivos generados:")
        print(f"   • fashion_recommender_eda_dashboard.html")
        print(f"   • fashion_recommender_detailed_analysis.png")

if __name__ == "__main__":
    # Ejecutar análisis completo
    eda = FashionRecommenderEDA()
    eda.run_complete_analysis()