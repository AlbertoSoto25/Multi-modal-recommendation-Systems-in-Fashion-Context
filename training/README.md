# Sistema Recomendador de Moda Multimodal

Este proyecto implementa un sistema recomendador de moda híbrido que combina elementos visuales y textuales usando Vision Transformer, BERT y mecanismos de atención cross-modal.

## 🏗️ Arquitectura del Modelo

### Componentes Principales

1. **Vision Transformer (ViT)**: Procesa las imágenes de los artículos de moda
2. **BERT**: Procesa las descripciones textuales de los productos
3. **Atención Cross-Modal**: Fusiona información visual y textual
4. **Aprendizaje Contrastivo**: Optimiza las representaciones para recomendación

### Flujo del Modelo

```
Imagen → ViT → Embeddings Visuales
                                    ↘
                                      Cross-Modal Attention → Fusión → Embedding Final
                                    ↗
Texto → BERT → Embeddings Textuales
```

## 📊 Dataset: H&M Personalized Fashion Recommendations

### Archivos de Datos
- `transactions_clean.csv`: Historial de compras de clientes
- `articles_clean.csv`: Metadatos detallados de artículos
- `customers_clean.csv`: Información de clientes
- `normalized_images/`: Imágenes de productos organizadas por carpetas

### Características del Dataset
- **Modalidad Visual**: Imágenes de productos de moda
- **Modalidad Textual**: Descripciones, categorías, colores, etc.
- **Datos Temporales**: Transacciones con timestamps
- **Escala**: Miles de productos y millones de transacciones

## 🚀 Configuración para Google Colab

### Paso 1: Configuración Inicial

```python
# Ejecutar en la primera celda
!git clone <repository-url>
%cd /content/model/training
!python colab_setup.py
```

### Paso 2: Subir Datos

1. Subir los archivos de datos a `/content/cleaned-data/`
2. Asegurarse de que la estructura de carpetas sea correcta:
   ```
   /content/cleaned-data/
   ├── transactions_clean.csv
   ├── articles_clean.csv  
   ├── customers_clean.csv
   └── normalized_images/
       ├── 010/
       ├── 011/
       └── ...
   ```

### Paso 3: Entrenamiento

```python
# Entrenamiento básico
!python main.py

# Entrenamiento con configuración personalizada
!python main.py --batch_size 16 --max_epochs 3 --learning_rate 3e-5

# Entrenamiento sin Wandb
!python main.py --no_wandb
```

## ⚙️ Configuración del Modelo

### Parámetros Principales

```python
# Configuración del modelo
model_config = {
    "vit_model_name": "google/vit-base-patch16-224",
    "bert_model_name": "bert-base-uncased", 
    "cross_modal_num_layers": 4,
    "cross_modal_num_heads": 12,
    "temperature": 0.07
}

# Configuración de datos
data_config = {
    "batch_size": 32,
    "streaming_buffer_size": 1000,
    "num_negative_samples": 5,
    "image_size": 224,
    "max_text_length": 128
}

# Configuración de entrenamiento
training_config = {
    "learning_rate": 5e-5,
    "max_epochs": 5,
    "patience": 3,
    "gradient_accumulation_steps": 4
}
```

### Optimizaciones para Colab

- **Carga en Streaming**: Los datos se cargan en chunks para evitar OOM
- **Batch Size Adaptativo**: Se ajusta automáticamente según la GPU disponible
- **Gradient Accumulation**: Para simular batch sizes más grandes
- **Mixed Precision**: Reduce uso de memoria y acelera entrenamiento
- **Checkpointing**: Guarda progreso regularmente

## 📈 Métricas de Evaluación

### Métricas Implementadas

1. **Precision@K**: Precisión en las top-K recomendaciones
2. **Recall@K**: Recall en las top-K recomendaciones  
3. **NDCG@K**: Normalized Discounted Cumulative Gain
4. **Hit Rate@K**: Porcentaje de usuarios con al menos un hit
5. **MRR**: Mean Reciprocal Rank
6. **MAP**: Mean Average Precision
7. **Coverage**: Cobertura del catálogo
8. **Diversity**: Diversidad intra-lista

### Valores K Evaluados
- K = [5, 10, 20, 50]

## 🔧 Uso del Sistema

### Entrenamiento Completo

```python
from config import Config
from data_loader import create_data_loaders
from trainer import FashionRecommenderTrainer

# Configuración
config = Config()

# Crear data loaders
train_loader, test_loader = create_data_loaders(config)

# Entrenar modelo
trainer = FashionRecommenderTrainer(config, train_loader, test_loader)
trainer.train()
```

### Inferencia

```python
# Cargar modelo entrenado
model = MultiModalFashionRecommender(config)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Obtener embeddings de artículos
embeddings = model(images, input_ids, attention_mask)

# Calcular similitudes para recomendación
similarities = torch.mm(user_embedding, item_embeddings.t())
recommendations = torch.argsort(similarities, descending=True)
```

## 📁 Estructura del Proyecto

```
training/
├── __init__.py              # Inicialización del paquete
├── config.py               # Configuraciones del sistema
├── model.py                # Arquitectura del modelo multimodal
├── data_loader.py          # Carga de datos en streaming
├── trainer.py              # Loop de entrenamiento
├── metrics.py              # Métricas de evaluación
├── main.py                 # Script principal de entrenamiento
├── colab_setup.py          # Configuración para Google Colab
├── requirements.txt        # Dependencias
└── README.md              # Documentación
```

## 🎯 Características Clave

### Streaming de Datos
- Carga datos en chunks para manejar datasets grandes
- Buffer configurable para balance memoria/velocidad
- Compatible con limitaciones de memoria de Colab

### Atención Cross-Modal
- Fusión bidireccional entre modalidades visual y textual
- Múltiples capas de atención para capturar interacciones complejas
- Normalización por capas y conexiones residuales

### Aprendizaje Contrastivo
- Muestreo negativo inteligente
- Temperatura ajustable para control de la distribución
- Optimización para ranking de recomendaciones

### Optimizaciones para Producción
- Mixed precision training
- Gradient accumulation
- Checkpointing automático
- Early stopping
- Logging con Wandb

## 🔍 Monitoreo del Entrenamiento

### Wandb Dashboard
- Pérdida de entrenamiento en tiempo real
- Métricas de evaluación
- Uso de memoria y GPU
- Hiperparámetros

### Logs Locales
- Archivo `training.log` con información detallada
- Métricas por época y paso
- Información de checkpoints

## ⚡ Consejos de Optimización

### Para T4 GPU (< 12GB)
```python
config.data.batch_size = 16
config.training.gradient_accumulation_steps = 8
config.data.streaming_buffer_size = 500
```

### Para P100/V100 (12-16GB)
```python
config.data.batch_size = 24
config.training.gradient_accumulation_steps = 6
config.data.streaming_buffer_size = 750
```

### Para A100 (> 16GB)
```python
config.data.batch_size = 32
config.training.gradient_accumulation_steps = 4
config.data.streaming_buffer_size = 1000
```

## 🐛 Solución de Problemas

### Out of Memory (OOM)
1. Reducir `batch_size`
2. Aumentar `gradient_accumulation_steps`
3. Reducir `streaming_buffer_size`
4. Activar `mixed_precision`

### Entrenamiento Lento
1. Aumentar `batch_size` si hay memoria disponible
2. Usar `num_workers > 0` si no está en Colab
3. Verificar que se está usando GPU

### Datos No Encontrados
1. Verificar rutas en `config.data.data_dir`
2. Asegurar estructura correcta de carpetas
3. Comprobar permisos de archivos

## 📚 Referencias

- Vision Transformer: "An Image is Worth 16x16 Words"
- BERT: "Bidirectional Encoder Representations from Transformers"  
- Cross-Modal Attention: "Attention is All You Need"
- Contrastive Learning: "A Simple Framework for Contrastive Learning"

## 🤝 Contribución

Para contribuir al proyecto:
1. Fork el repositorio
2. Crear una branch para tu feature
3. Hacer commit de los cambios
4. Push a la branch
5. Crear un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.
