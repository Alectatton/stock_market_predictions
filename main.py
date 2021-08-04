import LSTM
import lr


print("Welcome to this stock prediction model\n")
ticker = input("Please input the ticker symbol of the stock you want to predict (Ex: 'AAPL' for Apple): \n")
print("The program uses two different machine learning models to predict future values, "
      "a Long Short Term Memory neural network for the non-descriptive method,"
      "and a Linear Regression method for the descriptive method")

model_type = input("Input 'LSTM' for the neural network, and 'LR' for the Linear regression model (All caps): ")

if model_type == 'LSTM':
    print("\nCreating LSTM neural network model\n")
    LSTM.run_lstm(ticker)
elif model_type == 'LR':
    print("\nRunning linear regression model\n")
    lr.run_linear_regression(ticker)
else:
    print("\nInvalid input, closing program")
    exit()


