# from fastapi import FastAPI, File, UploadFile
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf


# app = FastAPI()
# MODEL = tf.keras.models.load_model("saved_model/v1")
# CLASS_NAME = ["Early Blight", "Late Blight", "Healthy"]
# @app.get("/ping")
# async def ping():
#     return {"message": "Server is alive"}


# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image


# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):  
#     image = read_file_as_image(await file.read())
#     img_batch =np.expand_dims(image, 0)
#     prediction = MODEL.predict(img_batch)
#     print(prediction)
#     pass

    
    

        






# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer
from tensorflow.keras import Input, Model
# from keras.applications import MobileNetV3Large
# from tensorflow.keras.applications import MobileNetV3Large



app = FastAPI()

# Load exported SavedModel using TFSMLayer (Keras 3+)
input_layer = Input(shape=(256, 256, 3), name="input_image")
output_layer = TFSMLayer("saved_model/v1", call_endpoint="serving_default")(input_layer)
MODEL = Model(inputs=input_layer, outputs=output_layer)

# Classes your model predicts
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))  # Match model input
    return np.array(image)  # Don't normalize if model was trained without it

@app.get("/ping")
async def ping():
    return {"message": "Server is alive"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

    predictions = MODEL.predict(img_batch)

# Extract the actual prediction array
    output_tensor = list(predictions.values())[0]  # Could be tf.Tensor or np.ndarray

# Handle both tf.Tensor and np.ndarray safely
    if hasattr(output_tensor, "numpy"):
        output = output_tensor.numpy()[0]
    else:
        output = output_tensor[0]

    predicted_class = CLASS_NAMES[np.argmax(output)]
    confidence = float(np.max(output))

    return {
        "class": predicted_class,
        "confidence": confidence
    }



import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # use PORT from env, fallback to 8000
    uvicorn.run("apis.main:app", host="0.0.0.0", port=port)



    # predictions = MODEL.predict(img_batch)
    # print("🧪 Predictions raw output:", predictions)

    # predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    # confidence = float(np.max(predictions[0]))

    # return {
    #     "class": predicted_class,
    #     "confidence": confidence
    # }

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
