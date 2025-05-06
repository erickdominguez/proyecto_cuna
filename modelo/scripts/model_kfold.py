import os
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

img_size = (224, 224)
batch_size = 16
num_folds = 5

# Cargar imágenes y etiquetas manualmente
def load_images_from_directory(directory, img_size):
    class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    images = []
    labels = []

    for label_index, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue  # Saltar archivos no imagen
            img = Image.open(fpath).convert('RGB').resize(img_size)
            img = np.array(img) / 255.0
            images.append(img)
            labels.append(label_index)

    return np.array(images), np.array(labels)

X, y = load_images_from_directory("padded_dataset_split/train", img_size)

# Validación cruzada K-Fold
kf = KFold(n_splits=num_folds, shuffle=True, random_state=123)

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = models.Sequential([
        tf.keras.applications.MobileNetV2(input_shape=img_size + (3,),
                                          include_top=False,
                                          weights='imagenet',
                                          pooling='avg'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_val, y_val))

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")
