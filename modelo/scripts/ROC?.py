import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


img_size = (224, 224)
batch_size = 16


model = tf.keras.models.load_model('saved_model/posture_model/4')


test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "padded_dataset_split/test",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False  
)

class_names = test_ds.class_names 
print("Clases detectadas:", class_names)

test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

y_true = []
y_pred_labels = []
y_pred_scores = []

for images, labels in test_ds:
    preds = model.predict(images).flatten()
    y_pred_scores.extend(preds)
    y_pred_labels.extend((preds > 0.5).astype(int))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred_labels = np.array(y_pred_labels)
y_pred_scores = np.array(y_pred_scores)

fpr, tpr, thresholds = roc_curve(y_true, y_pred_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
