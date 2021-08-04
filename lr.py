# Logistic Regression
# Descriptive machine learning method

import datetime
import pandas_datareader as web
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from prettytable import PrettyTable
from sklearn.linear_model import LinearRegression


# Function to run the linear regression model
def run_linear_regression(ticker):
    # Import the data using pandas_datareader from yahoo finance API
    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2021, 1, 1)
    data = web.DataReader(ticker, 'yahoo', start_date, end_date)

    # Clean the data
    data = data.drop(['Adj Close'], axis=1).dropna()

    # Create x and y arrays from data
    # y (Closing price) is the target data we are trying to predict
    x = data.drop(['Close'], axis=1)
    y = data['Close']

    # Train, test, split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    # Create the linear regression model
    model = LinearRegression()
    model = model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    # Prepare actual performance data
    test_start_date = end_date
    test_end_date = datetime.datetime.now()
    test_data = web.DataReader(ticker, 'yahoo', test_start_date, test_end_date)
    actual_prices = test_data['Adj Close'].values
    actual_prices = actual_prices[:len(prediction)]

    # Plot the data
    # 1st data visualization method
    def scatter_plot():
        plt.plot(actual_prices, color='black', label=f'{ticker} price')
        plt.plot(prediction, color='blue', label=f'Predicted {ticker} price')
        plt.title(f'{ticker} price')
        plt.xlabel('Days')
        plt.ylabel('$USD')
        plt.legend()
        plt.grid()
        plt.show()

    # Plot the difference between the prediction and actual data over time
    # 2nd data visualization method
    def price_comparison():

        # Create an empty difference array
        difference = []

        # Append data to the difference array
        for i in range(0, len(prediction)):
            val = prediction[i] - actual_prices[i]
            difference.append(val)

        # Show the plot
        plt.plot(difference, color='green')
        plt.title(f'Difference between predicted value of {ticker} and actual value over time')
        plt.xlabel('Days')
        plt.ylabel('$USD')
        plt.grid()
        plt.show()

    # Show the predicted data vs. actual data and the difference in a table
    # 3rd data visualization method
    def table_data():

        # Create an empty array for the difference and a table from PrettyTable
        difference = []
        table = PrettyTable(['Trading day', 'Actual', 'Predicted', 'Difference'])

        # Append data to difference array and add data to each row of the table
        for i in range(0, len(prediction)):
            val = prediction[i] - actual_prices[i]
            difference.append(val)
            table.add_row([i + 1, actual_prices[i], prediction[i], difference[i]])

        print(table)

    # Command line interface
    print('Prediction complete \n')
    print('Data can be visualized in 3 different ways from Jan 1, 2021 to 100 trading days later.')
    print('1. A scatter plot showing both actual performance and predicted performance')
    print('2. A scatter plot showing the difference between actual performance and predicted performance')
    print('3. A table showing actual data, predicted data, and the difference between the two \n')

    visualization = input('Enter 1, for option 1.  2 for option 2.  3 for option 3: ')

    if visualization == '1':
        print("\nOption 1 selected")
        scatter_plot()
    elif visualization == '2':
        print("\nOption 2 selected")
        price_comparison()
    elif visualization == '3':
        print("\nOption 3 selected\n")
        table_data()
    else:
        print("\nInvalid input, program closing")
        exit()









