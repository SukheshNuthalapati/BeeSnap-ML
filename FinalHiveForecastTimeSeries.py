#!/usr/bin/env python
# coding: utf-8

# In[83]:


# Import libraries
from dateutil.parser import parse 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[84]:


# Retrieve database
ladnaus = 'https://raw.githubusercontent.com/SukheshNuthalapati/BeeSnap-ML/master/HiveTool/LadnhausHainsHiveTool%20-%20Sheet1.csv'

df = pd.read_csv(ladnaus)

# Clean database to only observe needed variables
# Our independent variable is date (we are conducting a time series analysis)
# Our dependent (endogenous) variable is hive_weight
# Our exogenous variables are hive_temperature, hive_humidity, ambient_temperature, ambient_humidity, ambient_rain
df = df.iloc[14:-100]
df = df[['date', 'hive_weight', 'hive_temperature', 'hive_humidity', 'ambient_temperature', 'ambient_humidity', 'ambient_rain']]
df


# In[85]:


# Analyze how much data is missing from exogenous variables
data_missing = df.isna()
data_missing_count = data_missing.sum()
data_missing_count / len(df)

# If we are missing more than 25% of the data for hive_humidity and ambient_humidity, we should drop those variables as they may impact our results
columns_to_drop = []
for column in df.columns:
    drop_percentage = df[column].isna().sum() / len(df[column])
    if drop_percentage > 0.25:
      columns_to_drop.append(column)

df = df.drop(columns = columns_to_drop)

# To deal with any missing data, we can perform a backwards fill and then a forward fill on missing data
df = df.fillna(method='bfill').fillna(method='ffill')

values = df.hive_weight.to_numpy()

exog = df.drop(columns = ['date', 'hive_weight']).to_numpy()

df


# In[86]:


# We can start by running our initial ARIMA model without using exogenous variables
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

model = ARIMA(df.hive_weight, order=(p_initial,d_initial,q_initial))
arima = model.fit()

forecasts = arima.fittedvalues.to_numpy()
forecasts[forecasts < 0] = 0
print("Predicted: ", forecasts)
print("Actual: ", values)

plt.plot(forecasts, color = 'red')
plt.plot(values, color = 'blue')
plt.xlabel("Time (in days from 2013 to 2017)")
plt.ylabel("Hive Weight (in pounds/div)")

plt.show()


# In[87]:


# Now we can test a (1, 0, 2) ARIMA model using exogenous variables

model = ARIMA(df.hive_weight, exog = exog, order=(1,0,2))
arimax = model.fit()

forecasts = arimax.fittedvalues

forecasts = forecasts.to_numpy()
forecasts[forecasts < 0] = 0

plt.plot(forecasts, color = 'red')
plt.plot(values, color = 'blue')
plt.xlabel("Time (in days from 2013 to 2017)")
plt.ylabel("Hive Weight (in pounds/div)")

plt.show()


# In[88]:


# Finally, we can test our forecasting on a SARIMAX model, which may be more effective since we know that our data is seasonal

from statsmodels.tsa.statespace.sarimax import SARIMAX

#0, 1, 2 SARIMAX Model
model= SARIMAX(df.hive_weight, exog = exog, order=(1,0,2), enforce_invertibility=False, enforce_stationarity=False)
sarimax = model.fit(disp=0)

forecasts = sarimax.fittedvalues.to_numpy()
forecasts[forecasts < 0] = 0

plt.plot(forecasts, color = 'red')
plt.plot(values, color = 'blue')
plt.xlabel("Time (in days from 2013 to 2017)")
plt.ylabel("Hive Weight (in pounds/div)")

plt.show()


# In[92]:


# To look at next 50 forecast predictions

length = len(df.hive_weight)
amt = 5
diff = length - amt
# Create Training and Test
train = df.hive_weight[:diff]
exo_train = exog[:diff]
test = df.hive_weight[diff:]
exo_test = exog[diff:]
print(train.to_numpy())


# In[93]:


# Build Model
model = SARIMAX(train, order=(1, 0, 2))  
sarimax_test = model.fit()  

model2 = ARIMA(train, order = (1, 0, 2))
arima = model2.fit()

# Forecast
forecasts = arima.forecast(amt, alpha = 0.05)  # 95% conf

# Forecast
print(forecasts)

# Actual
print(test.to_numpy())


# In[91]:


from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import pickle

sarimax.save('hive_model.pkl')
loaded = SARIMAXResults.load('hive_model.pkl')

arima.save('hive_model_arima.pkl')
loaded_arima = ARIMAResults.load('hive_model_arima.pkl')
#pickle.dump('hive_model.pkl', open('hive_model.pkl', 'wb'))
print(loaded_arima.summary())


# In[ ]:





# In[ ]:




