import requests
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_and_pad_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert("RGB")
    old_size = img.size

    ratio = min(target_size[0] / old_size[1], target_size[1] / old_size[0])
    new_size = (int(old_size[0] * ratio), int(old_size[1] * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    new_img = Image.new("RGB", target_size, (0, 0, 0))
    paste_position = (
        (target_size[0] - new_size[0]) // 2,
        (target_size[1] - new_size[1]) // 2,
    )
    new_img.paste(img, paste_position)

    img_array = np.array(new_img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

img_array = load_and_pad_image("image.jpg")
data = {"instances": img_array.tolist()}
response = requests.post("http://localhost:8501/v1/models/posture_model_v5:predict", json=data)

try:
    prediction = response.json()["predictions"][0][0]
    print(f"Score: {prediction:.4f}")
    print("SAFE" if prediction >= 0.5 else " DANGEROUS")
except Exception as e:
    print(" Error", response.text)
