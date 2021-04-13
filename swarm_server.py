import os
import numpy as np
import pandas as pd
import pickle
from dateutil.parser import parse
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.arima.model import ARIMAResults
import flask
import io

app = flask.Flask(__name__)

def load_model():
	model_path = 'hive_model_arima.pkl'
	model = ARIMAResults.load(model_path)
	return model

def preprocess(dataframe):
	dataframe = dataframe.iloc[14:-100]
	dataframe = dataframe[['date', 'hive_weight', 'hive_temperature', 'hive_humidity', 'ambient_temperature', 'ambient_humidity', 'ambient_rain']]
	data_missing = dataframe.isna()
	data_missing_count = data_missing.sum()
	data_missing_count / len(dataframe)

	columns_to_drop = []
	for column in dataframe.columns:
	    drop_percentage = dataframe[column].isna().sum() / len(dataframe[column])
	    if drop_percentage > 0.25:
	      columns_to_drop.append(column)

	dataframe = dataframe.drop(columns = columns_to_drop)
	dataframe = dataframe.fillna(method='bfill').fillna(method='ffill')

	values = dataframe.hive_weight.to_numpy()
	exog = dataframe.drop(columns = ['date', 'hive_weight']).to_numpy()

	return values, exog


def forecast(data, model, steps):
	df = pd.read_csv(data)
	values, exog = preprocess(df)
	new_model = model.apply(values)
	fc = new_model.forecast(steps, alpha = 0.05)
	return fc.tolist()

# model = load_model()
# print(model.summary())
# df = pd.read_csv('https://raw.githubusercontent.com/SukheshNuthalapati/BeeSnap-ML/master/HiveTool/LadnhausHainsHiveTool%20-%20Sheet1.csv')
# values, exog = preprocess(df)
# print(values)
# new_model = model.apply(values)

@app.route("/predict", methods = ["GET"])
def predict():
	model = load_model()
	try:
		return flask.jsonify(forecast('https://raw.githubusercontent.com/SukheshNuthalapati/BeeSnap-ML/master/HiveTool/LadnhausHainsHiveTool%20-%20Sheet1.csv', model, 5))
	except Exception as e:
		print(str(e))

if __name__ == "__main__":
	print(("* Loading ARIMA model and Flask starting server..."
		"please wait until server has fully started"))
	app.run()
