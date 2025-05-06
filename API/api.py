from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import boto3
import json
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()
ENDPOINT_NAME = "posture-v5"

# Función para preprocesar imagen
def preprocess_image(file: UploadFile, target_size=(224, 224)):
    img = Image.open(BytesIO(file.file.read())).convert("RGB")
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

# Endpoint FastAPI
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_array = preprocess_image(file)

        # SageMaker runtime client
        runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
        payload = json.dumps({"instances": img_array.tolist()})

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=payload
        )

        result = json.loads(response['Body'].read().decode())
        score = result['predictions'][0][0]

        label = "safe" if score >= 0.5 else "dangerous"

        return JSONResponse({
            "score": round(score, 4),
            "prediction": label
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
