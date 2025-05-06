import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Configuración ---
img_path = "20250105_180738.jpg"  # <-- Cambia esto si usas otra imagen
img_size = (224, 224)
threshold = 0.5  # Puedes ajustar este valor según tu análisis

# --- Cargar modelo ---
model = tf.keras.models.load_model('saved_model/posture_model/5')

# --- Cargar y preprocesar imagen ---
img = Image.open(img_path).convert('RGB').resize(img_size)
img_array = np.array(img)
img_array = preprocess_input(img_array)  # <-- ¡importante!
img_array = np.expand_dims(img_array, axis=0)

# --- Hacer predicción ---
pred = model.predict(img_array)
score = pred[0][0]
label = "safe" if score > threshold else "dangerous"

# --- Mostrar resultados ---
print(f"Score bruto del modelo: {score:.4f}")
print(f"Clasificación con umbral {threshold}: {label}")

# --- Mostrar imagen ---
plt.imshow(img)
plt.title(f"Predicción: {label} (score: {score:.2f})")
plt.axis('off')
plt.show()
