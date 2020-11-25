#USAGE: curl -X POST -F image=@IMAGE 'http://localhost:5000/predict'
#EXAMPLE: curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import flask
import io

app = flask.Flask(__name__)

def load_model():
    model_path = "bee_model_v0"
    model = keras.models.load_model(model_path)
    return model

def classify(img_path, saved_model, class_names):
    # Preprocess image
    IMG_SIZE = 150
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img.astype("float") / 255.0
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    
    prediction = saved_model.predict(img_batch)
    return class_names[np.argmax(prediction)]

@app.route("/predict", methods=["GET"])
def predict():
    class_names = ['Blueberry Bee', 'Bumblebee', 'Carpenter Bee', 'Honey Bee', 'Mason Bee', 'Mining Bee', 'Western Honey Bee']
    model = load_model()
    try:
        return flask.jsonify(classify('bees/honeybees/7565602-bee-apis-mellifera-european-or-western-honey-bee-isolated-on-white-wingspan-18mm.jpg', model, class_names))
        #return flask.jsonify(classify('bees/masonbees/large0 2.jpg', model, class_names))
    except Exception as e:
        print(str(e))

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	app.run()
