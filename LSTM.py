# LSTM Neural Network
# non-descriptive machine learning method

import datetime
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# Function to run LSTM neural network model
def run_lstm(ticker):
    # Import and clean the data
    # Use pandas_datareader to pull from yahoo finance API
    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2021, 1, 1)
    data = web.DataReader(ticker, 'yahoo', start_date, end_date)
    data = data.dropna()

    # Scaled down data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))
    num_days = 90

    # Initiate the arrays for x and y axis
    x_train = []
    y_train = []

    # Append scaled data to the arrays
    for x in range(num_days, len(scaled_data)):
        x_train.append(scaled_data[x - num_days:x, 0])
        y_train.append(scaled_data[x, 0])

    # Reformat arrays for use by numpy
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # LSTM Model
    model = Sequential()  # New Sequential object from tensorFlow
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=1, batch_size=1)

    # Create testing data set
    test_start_date = end_date
    test_end_date = datetime.datetime.now()
    test_data = web.DataReader(ticker, 'yahoo', test_start_date, test_end_date)
    actual_prices = test_data['Adj Close'].values
    total_dataset = pd.concat((data['Adj Close'], test_data['Adj Close']), axis=0)

    # Create datasets to input into the model
    model_input = total_dataset[len(total_dataset) - len(test_data) - num_days:].values
    model_input = scaler.transform(model_input.reshape(-1, 1))

    # Append model values to x_test array
    x_test = []
    for x in range(num_days, len(model_input)):
        x_test.append(model_input[x - num_days:x, 0])

    # Reshape x_test array and run prediction model
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    prediction = scaler.inverse_transform(model.predict(x_test))

    # Data visualized as a scatter plot comparing performance of the prediction model and actual data
    def scatter_plot():

        print("Plot created")
        plt.plot(actual_prices, color='black', label=f'{ticker} price')
        plt.plot(prediction, color='blue', label=f'Predicted {ticker} price')
        plt.title(f'{ticker} price')
        plt.xlabel('Days')
        plt.ylabel('$USD')
        plt.legend()
        plt.show()

    scatter_plot()
