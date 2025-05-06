from sklearn.metrics import classification_report, f1_score
import numpy as np
import tensorflow as tf
img_size = (224, 224)
batch_size = 16
#COLOCAR EL MODELO ACTUAL A EVALUAR
model = tf.keras.models.load_model('saved_model/posture_model/5')

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "padded_dataset_split/val",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)


val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


y_pred_probs = model.predict(val_ds)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)

print("Reporte de clasificación (safe vs dangerous):\n")
print(classification_report(y_true, y_pred, target_names=["dangerous", "safe"]))

f1 = f1_score(y_true, y_pred)
print(f"\nF1 Score global: {f1:.4f}")
