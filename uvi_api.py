from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import io

# Load the saved model
model = load_model('dog_emotions_model.h5')

# Define the emotions that the model can classify
emotions = ['happy', 'sad', 'angry']

app = FastAPI()

def predict_emotion(image):
    # Preprocess the image
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0

    # Make a prediction
    prediction = model.predict(image_array)
    predicted_class = emotions[np.argmax(prediction)]

    return predicted_class

@app.post("/predict_emotion/")
async def predict_emotion_endpoint(file: UploadFile = File(...)):
    # Load the input image
    image = load_img(io.BytesIO(file.file.read()), target_size=(224, 224))
    # Predict the emotion
    emotion = predict_emotion(image)
    print(emotion)
    # Return the result
    return JSONResponse({"predicted_emotion": emotion})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)