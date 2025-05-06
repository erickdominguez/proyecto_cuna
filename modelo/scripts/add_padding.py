import os
import tensorflow as tf
import numpy as np
from PIL import Image

TARGET_SIZE = 224

input_root = "dataset_split"
output_root = "padded_dataset_split"

def pad_and_resize_image(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0  

    h, w, _ = image_np.shape
    scale = min(TARGET_SIZE / h, TARGET_SIZE / w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = tf.image.resize(image_np, [new_h, new_w])

    pad_height = TARGET_SIZE - new_h
    pad_width = TARGET_SIZE - new_w
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded = tf.pad(resized, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0)

    padded_image = tf.image.convert_image_dtype(padded, dtype=tf.uint8)
    img = Image.fromarray(padded_image.numpy())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)

def process_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                in_path = os.path.join(root, fname)
                rel_path = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, rel_path)
                pad_and_resize_image(in_path, out_path)

for split in ["train", "val", "test"]:
    input_dir = os.path.join(input_root, split)
    output_dir = os.path.join(output_root, split)
    process_directory(input_dir, output_dir)

