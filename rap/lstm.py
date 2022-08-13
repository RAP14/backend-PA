import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

from django.http import JsonResponse
from rap import utils

#contoh

def get_prediction(nama_coin, day):
    df = utils.get_coin(nama_coin)

    price = pd.DataFrame(df,columns=['last_price'])
    data = price.values
    data = data.astype('float32')

    temp_data = []
    for i in range (len(data)-100, len(data)):
        temp_data.append(data[i])

    scaler = MinMaxScaler(feature_range=(0, 1))
    preprocessing_data = scaler.fit_transform(temp_data)

    train_size = int(len(temp_data) * 0.7)
    test_size = len(temp_data) - train_size

    train_data = preprocessing_data[0:train_size, :]
    test_data = preprocessing_data[train_size:len(temp_data) , :]

    timesteps = 1
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for x in range (timesteps, len(train_data)):
        x_train.append(train_data[x-timesteps:x,0])
        y_train.append(train_data[x, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    for x in range (timesteps, len(test_data)):
        x_test.append(test_data[x-timesteps:x,0])
        y_test.append(test_data[x, 0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = np.reshape(x_train,(x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test,(x_test.shape[0], 1, x_test.shape[1]))

    model = Sequential()
    model.add(LSTM(200, return_sequences=True, input_shape=(1, timesteps)))
    model.add(LSTM(200, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))
    model.fit(x_train, y_train, batch_size=1, epochs=50, verbose=1)

    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)
    y_train = scaler.inverse_transform([y_train])
    y_test = scaler.inverse_transform([y_test])

    # result_plot = price[len(price)-len(y_test[0]):len(price)]
    # result_plot['prediction'] = prediction

    score = math.sqrt(mean_squared_error(y_test[0], prediction[:,0]))

    ts = pd.DataFrame(df,columns=['timestamp'])
    timestamp = np.array(ts)
    datetime = pd.to_datetime(ts['timestamp'])
    n_future = day
    forecast_period = pd.date_range(list(datetime)[len(datetime)-1], periods=n_future+1, freq='1H').tolist()
    forecast_period.pop(0)

    forecast = model.predict(x_test[-n_future:])
    forecast = scaler.inverse_transform(forecast)

    response = {}
    prediction_price = []

    for x in range (0, len(data)):
        temp_1 = {}
        temp_1["price"] = data[x][0].tolist()
        temp_1["timestamp"] = timestamp[x].tolist()
        prediction_price.append(temp_1)

    for x in range (0, len(forecast)):
        temp_2 = {}
        temp_2["price"] = forecast[len(forecast)-1-x][0].tolist()
        temp_2["timestamp"] = forecast_period[x].strftime('%Y-%m-%d %H:%M:%S')
        prediction_price.append(temp_2)
        
    response["prediction_prices"] = prediction_price
    response["prediction_score"] = score

    print("Price : ", len(data))
    print("Data : ", len(temp_data))
    print("Hari : ", n_future)
    print("Prediction : ", forecast[n_future-1])
    print("MSE : ", score)

    return JsonResponse(response)