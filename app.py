from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import sklearn.externals 
from datetime import datetime

app = Flask(__name__)

df= pd.read_csv('Data/feed.csv',parse_dates=True)
df['Date'] = pd.to_datetime(df['Date'])
df=df.set_index('Date')
#lr = joblib.load("Model/forecast.pickle")
scaler = joblib.load("Data/scaler.save") 
assert isinstance(scaler, MinMaxScaler)
scaler.clip = False # add this line
model=load_model("Models/mymodel.h5")

@app.route("/")
def home():
    # test= df
    # scaled_test = scaler.transform(test)
    # #print(scaled_test)
    # #print(test)
    # n_input = 12
    # n_features = 1
    # first_eval_batch = scaled_test[-n_input:]
    # current_batch = first_eval_batch.reshape((1, n_input, n_features))
    # #print(current_batch)
    # result = model.predict(current_batch)[0]
    # #print(result)
    # prediction = scaler.inverse_transform([result])
    # print(prediction)
    
    return render_template('index.html')

def predict(number_of_days):
    test= df
    scaled_test = scaler.transform(test)
    
    n_input = 12
    n_features = 1
    first_eval_batch = scaled_test[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    
    #result = model.predict(current_batch)[0]
    
    test_predictions = []


    for i in range(number_of_days):
        
    
        # get the prediction value for the first batch
        current_pred = model.predict(current_batch)[0]
    
        # append the prediction into the array
        test_predictions.append(current_pred) 
    
        # use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
    
    prediction = scaler.inverse_transform(test_predictions)
    print(prediction[len(prediction)-3])
    return(prediction)

@app.route('/', methods = ['POST'])
def main():
                 
    if request.method == 'POST':
       Date= request.form['Date']
       
       #print(Date)
       #print(type(Date))
       
       
       # dates in string format
       str_d1 = '2022/11/27'
       str_d2 = Date

       # convert string to date object
       d1 = datetime.strptime(str_d1, "%Y/%m/%d")
       d2 = datetime.strptime(str_d2, "%Y/%m/%d")

       # difference between dates in timedelta
       delta = d2 - d1
       print(f'Difference is {delta.days} days')
              
       days=0
       days=delta.days
       
       print(days)
       print(type(days))
       
       predicted_AQI=predict(days)
       pq1=predicted_AQI[len(predicted_AQI)-1][0]
       pq2=predicted_AQI[len(predicted_AQI)-2][0]
       pq3=predicted_AQI[len(predicted_AQI)-3][0]
              
    
            
    
   
    return render_template("index.html",pq1=np.round(pq1,3),pq2=np.round(pq2,3),pq3=np.round(pq3,3) )

if __name__ == "__main__":
    app.run(debug = True)
