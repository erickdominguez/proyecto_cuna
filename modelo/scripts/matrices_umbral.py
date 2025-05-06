import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


img_size = (224, 224)
batch_size = 16

#EVALUAR MODELO ACTUAL
model = tf.keras.models.load_model('saved_model/posture_model/5')


test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "padded_dataset_split/test",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False  
)


test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


y_true = []
y_pred_scores = []

for images, labels in test_ds:
    preds = model.predict(images).flatten()
    y_pred_scores.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred_scores = np.array(y_pred_scores)

thresholds = np.arange(0.0, 1.01, 0.05)
fn_counts = []
tn_counts = []

for thresh in thresholds:
    y_pred_labels = (y_pred_scores > thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred_labels)
    
    tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
    
    fn_counts.append(fn)
    tn_counts.append(tn)

plt.plot(thresholds, fn_counts, label='Falsos Negativos (FN)', color='red')
plt.plot(thresholds, tn_counts, label='Verdaderos Negativos (TN)', color='green')
plt.xlabel('Umbral de decisión')
plt.ylabel('Cantidad')
plt.title('FN y TN vs Umbral de decisión')
plt.legend()
plt.grid(True)
plt.show()
