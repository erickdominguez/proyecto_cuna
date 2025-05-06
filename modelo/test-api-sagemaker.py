import boto3
import json
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_and_pad_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert("RGB")
    old_size = img.size

    ratio = min(target_size[0]/old_size[1], target_size[1]/old_size[0])
    new_size = (int(old_size[0]*ratio), int(old_size[1]*ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    new_img = Image.new("RGB", target_size, (0, 0, 0))
    paste_position = ((target_size[0] - new_size[0]) // 2,
                      (target_size[1] - new_size[1]) // 2)
    new_img.paste(img, paste_position)

    img_array = np.array(new_img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict_from_sagemaker(img_path, endpoint_name):
    runtime = boto3.client('sagemaker-runtime')

    img_array = load_and_pad_image(img_path)
    payload = json.dumps({"instances": img_array.tolist()})

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=payload
    )

    result = json.loads(response['Body'].read().decode())
    prediction_score = result['predictions'][0][0]

    print(f"Prediction score: {prediction_score:.4f}")
    if prediction_score < 0.5:
        print("DANGEROUS posture")
    else:
        print("SAFE posture")

#DESCOMENTAR PARA PROBAR
# predict_from_sagemaker("IMAGEN AQUI", endpoint_name="posture-v5")
