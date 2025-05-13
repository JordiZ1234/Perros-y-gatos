# 1. LIBRERIES NECESSÀRIES
import tensorflow as tf  # Importa TensorFlow, la biblioteca principal para tareas de machine learning.
from tensorflow.keras import layers, models  # Importa herramientas de Keras para definir las capas y el modelo.
import matplotlib.pyplot as plt  # Permite generar gráficos para visualizar resultados.
import numpy as np  # Biblioteca para manejo de arrays y operaciones matemáticas.
import zipfile  # Nos ayuda a trabajar con archivos Zip.
import os  # Permite interactuar con el sistema operativo (por ejemplo, rutas de archivos).
import pathlib  # Facilita el manejo de rutas de archivos de forma intuitiva.
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Herramienta para preprocesar y enriquecer imágenes.

# Explicación: Se cargan las librerías necesarias para entrenar el modelo, preprocesar imágenes, gestionar archivos y visualizar resultados.

# 2. DESCÀRREGA I PREPARACIÓ DEL DATASET
url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'  # URL del dataset comprimido de gatos y gossos (gatos y perros).
zip_path = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=url)  # Descarga el archivo ZIP y guarda la ruta.

# Abre el archivo ZIP y extrae todo su contenido en el directorio donde se descargó.
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(zip_path))

# Define las rutas de los directorios donde se encuentran los datos extraídos.
base_dir = os.path.join(pathlib.Path(zip_path).parent, 'cats_and_dogs_filtered')  # Directorio base del dataset.
train_dir = os.path.join(base_dir, 'train')  # Directorio con las imágenes para entrenamiento.
validation_dir = os.path.join(base_dir, 'validation')  # Directorio con las imágenes para validación.

# Explicación: Se descarga y descomprime el dataset, y se especifican las rutas de las carpetas de entrenamiento y validación.

# 3. PREPROCESSAT DE DADES
# Se crea un objeto ImageDataGenerator para normalizar las imágenes (los valores de píxel se escalan de 0 a 1).
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)

# Se preparan los datos de entrenamiento leyendo las imágenes de la carpeta 'train'
train_generator = train_datagen.flow_from_directory(
    train_dir,                  # Directorio de imágenes de entrenamiento.
    target_size=(100, 100),     # Redimensiona las imágenes a 100x100 píxeles.
    batch_size=10,              # Procesa las imágenes en lotes de 10.
    class_mode='binary'         # Se usa para clasificación binaria (dos clases: gato o perro).
)

# Se preparan los datos de validación leyendo las imágenes de la carpeta 'validation'
validation_generator = val_datagen.flow_from_directory(
    validation_dir,             # Directorio de imágenes de validación.
    target_size=(100, 100),     # Redimensiona las imágenes a 100x100 píxeles.
    batch_size=10,              # Procesa las imágenes en lotes de 10.
    class_mode='binary'         # Clasificación binaria.
)

# Explicación: Se generan los lotes de datos a partir de las imágenes, redimensionándolas y normalizándolas para que el modelo pueda procesarlas correctamente.

# 4. DEFINICIÓ DEL MODEL LLEUGER
# Se define un modelo secuencial, es decir, las capas se apilan una detrás de otra de forma lineal.
model = models.Sequential([
    # Primera capa convolucional: 8 filtros de 3x3, función de activación ReLU.
    # 'input_shape' define las dimensiones de las imágenes de entrada (100x100 píxeles con 3 canales de color).
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(100, 100, 3)),  
    layers.MaxPooling2D(2, 2),  # Capa de pooling para reducir la dimensión espacial de los datos.

    # Segunda capa convolucional: 16 filtros que permiten aprender características más complejas.
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),  # Otra capa de pooling para seguir reduciendo la dimensión.

    layers.Flatten(),  # Aplana la salida 2D en un vector 1D para conectarla con las capas densas.
    layers.Dense(32, activation='relu'),  # Capa densa con 32 neuronas para aprender combinaciones de características.
    layers.Dense(1, activation='sigmoid')  # Capa de salida con 1 neurona y activación sigmoide para clasificar en dos clases.
])

# Compilamos el modelo definiendo la función de pérdida, el optimizador y la métrica a evaluar.
model.compile(
    loss='binary_crossentropy',  # Pérdida adecuada para problemas de clasificación binaria.
    optimizer='adam',            # Optimizador 'adam', que ajusta los pesos de manera eficiente.
    metrics=['accuracy']         # Métrica para evaluar la precisión del modelo durante el entrenamiento y validación.
)

# Explicación: Se ha definido una red neuronal convolucional simple y ligera para clasificar imágenes entre dos clases (gato y perro).

# 5. ENTRENAMENT DEL MODEL
history = model.fit(
    train_generator,            # Utiliza los datos de entrenamiento.
    epochs=5,                   # Número de veces que el modelo verá todo el conjunto de datos; se puede aumentar para mejorar los resultados.
    validation_data=validation_generator  # Datos de validación para evaluar el rendimiento en cada época.
)

# Explicación: Entrena el modelo durante 5 épocas, ajustando internamente sus parámetros y evaluando la precisión con el conjunto de validación.

# 6. EXPORTACIÓ DEL MODEL EN FORMAT LLEUGER
# Guardamos la estructura del modelo en un archivo JSON.
model_json = model.to_json()  # Convierte la arquitectura del modelo a formato JSON.
with open("model_gats_gossos.json", "w") as json_file:
    json_file.write(model_json)  # Escribe la arquitectura en el archivo JSON.

# Guardamos los pesos del modelo en un archivo H5.
model.save_weights("model_gats_gossos.weights.h5")  # Guarda los pesos en formato H5.

# Explicación: Se separa la estructura y los pesos del modelo para crear un archivo compacto y fácil de compartir (por ejemplo, en GitHub).

# 7. DESCÀRREGA DELS FITXERS PER PUJAR A GITHUB
from google.colab import files  # Importa la función 'files' de Google Colab para descargar archivos.
files.download("model_gats_gossos.json")  # Descarga el archivo JSON con la arquitectura del modelo.
files.download("model_gats_gossos.weights.h5")  # Descarga el archivo H5 con los pesos del modelo.

# Explicación: Este bloque permite bajar los archivos resultantes desde Google Colab para poder subirlos a GitHub o usarlos localmente.

# (Opcional) GRÀFICA D'EVOLUCIÓ DE LA PRECISIÓ DURANT L'ENTRENAMENT
acc = history.history['accuracy']         # Extrae la precisión del entrenamiento registrada en cada época.
val_acc = history.history['val_accuracy']    # Extrae la precisión de la validación registrada en cada época.
epochs_range = range(len(acc))               # Crea un rango que representa el número de épocas.

plt.figure(figsize=(8, 6))                   # Configura el tamaño del gráfico.
plt.plot(epochs_range, acc, 'r', label='Entrenament')       # Dibuja la curva de precisión para el entrenamiento (en rojo).
plt.plot(epochs_range, val_acc, 'b', label='Validació')     # Dibuja la curva de precisión para la validación (en azul).
plt.title('Evolució de la Precisió')         # Añade un título al gráfico.
plt.legend()                                  # Muestra una leyenda para identificar cada curva.
plt.show()                                    # Muestra el gráfico en pantalla.

# Explicación: Se genera y muestra una gráfica que permite ver cómo evoluciona la precisión del modelo a lo largo del entrenamiento y validar visualmente su comportamiento.
