import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import numpy as np

# Rutas
input_dir = 'dataset_split/train/safe'
output_dir = 'dataset_augmented/train/safe'  
os.makedirs(output_dir, exist_ok=True)


datagen = ImageDataGenerator(
    brightness_range=[1, 1.4],  
    fill_mode='nearest'
)
num_to_generate = 82
generated_count = 0

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_dir, filename)
        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        aug_iter = datagen.flow(x, batch_size=1)

        for i in range(5): 
            if generated_count >= num_to_generate:
                break

            aug_img = next(aug_iter)[0].astype(np.uint8)
            aug_img_pil = array_to_img(aug_img)
            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(output_dir, f'{base_name}_aug_{generated_count}.jpg')
            aug_img_pil.save(save_path)
            generated_count += 1

    if generated_count >= num_to_generate:
        break

print(f'Listo, done, terminado')
