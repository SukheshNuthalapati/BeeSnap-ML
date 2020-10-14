#USAGE: curl -X POST -F image=@IMAGE 'http://localhost:5000/predict'
#EXAMPLE: curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import pickle

app = flask.Flask(__name__)
model = None

def load_model():
	global model
	model = pickle.load(open('model.pickle', 'rb'))

def prepare_image(image, target):
	if image.mode != "RGB":
		image = image.convert("RGB")
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)
	return image

@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			image = prepare_image(image, target=(224, 224))
				
			prediction_array = []
			prediction_array = model.predict(np.array([image]))
			class_names = ['sunflower', 'rose', 'tulip', 'daisy', 'iris', 'azalea', 'buttercup', 'daffodil', 'fouroclock', 'magnolia', 'marigold', 'peonies', 'petunia', 'violet']
			predicted_label = np.argmax(prediction_array)
			print(class_names[predicted_label])
	return flask.jsonify(class_names[predicted_label])

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()
