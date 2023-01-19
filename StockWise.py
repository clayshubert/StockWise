import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tabulate import tabulate



def get_stock_data(stock_symbol):
    # Download historical stock prices
    stock_data = yf.download(stock_symbol)
    # Convert the data into a format that can be easily analyzed
    print("Data collected from Yahoo Finance.")
    return pd.DataFrame(stock_data).dropna()

"""
def predict_stock_price(df):
    # Create new features based on the data
    df['close_lag'] = df['Close'].shift(1)
    df['close_lag_2'] = df['Close'].shift(2)
    df['volume_lag'] = df['Volume'].shift(1)
    # Define the predictors and the target variable
    X = df[['close_lag', 'close_lag_2', 'volume_lag']]
    y = df['Close']
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a pipeline with an imputer to handle missing values
    imputer = SimpleImputer()
    model = LinearRegression()
    pipeline = Pipeline([("imputer", imputer), ("regression", model)])
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    print("\n[----------------Linear Regression Predictor Results----------------]\n")
    # Use the model to make predictions for the next day
    next_day_prediction = pipeline.predict(X_test.tail(1))
    y_pred = pipeline.predict(X_test)
    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    table = [["Prediction for next day", "{:.4f}".format(next_day_prediction[0])],
             ["Mean Squared Error", "{:.4f}".format(mse)]]
    print(tabulate(table, tablefmt="double_outline"))
    threshold = 1
    if mse < threshold:
        print("Verdict: Prediction is accurate")
    else:
        print("Verdict: Prediction is not accurate")
"""

def calculate_moving_average(df, window=50):
    df['sma_' + str(window)] = df['Close'].rolling(window=window).mean()
    return df

def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, window=20):
    df['middleband'] = df['Close'].rolling(window=window).mean()
    df['std'] = df['Close'].rolling(window=window).std()
    df['upperband'] = df['middleband'] + (df['std'] * 2)
    df['lowerband'] = df['middleband'] - (df['std'] * 2)
    return df

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    df['short_ma'] = df['Close'].rolling(window=short_window).mean()
    df['long_ma'] = df['Close'].rolling(window=long_window).mean()
    df['macd'] = df['short_ma'] - df['long_ma']
    df['signal'] = df['macd'].rolling(window=signal_window).mean()
    df['histogram'] = df['macd'] - df['signal']
    return df

def calculate_fibonacci_retracement(df, period=100):
    high = df['High'].rolling(window=period).max()
    low = df['Low'].rolling(window=period).min()
    retracement_levels = pd.DataFrame({'0.236': high * 0.236,
                                      '0.382': high * 0.382,
                                      '0.500': high * 0.500,
                                      '0.618': high * 0.618,
                                      '0.764': high * 0.764},
                                      index=df.index)
    df = pd.concat([df, retracement_levels], axis=1)
    return df

def get_current_price(stock_symbol):
    stock_info = yf.Ticker(stock_symbol).info
    current_price = stock_info['regularMarketPrice']
    print("Current Stock Price: {:.2f}".format(current_price))

def get_analyst_rec(stock_symbol):
    print("\nAnalyst Reccomendations:\n")
    df = yf.Ticker(stock_symbol).recommendations_summary.drop(["Action", "From Grade"], axis=1)
    df = df.sort_values(by='Date', ascending=False)
    df = df.head(3)
    print(tabulate(df,headers= ['Date', 'Firm', 'Grade'], tablefmt="double_outline"))
    df = yf.Ticker(stock_symbol).analyst_price_target
    df = df.drop('currentPrice', axis=0)
    table = [["Number of Analysts", df.loc['numberOfAnalystOpinions'].values[0]],
             ["Target Low", df.loc['targetLowPrice'].values[0]],
             ["Target Mean", df.loc['targetMeanPrice'].values[0]],
             ["Target Hight", df.loc['targetHighPrice'].values[0]]]
    print(tabulate(table, headers= ["Measure","Value"], tablefmt="double_outline")) 
   

def main():
    # Define the stock symbol
    print("Provide the stock ticker (ex: Apple = AAPL): ")
    stock_symbol = input()
    df = get_stock_data(stock_symbol)
    #predict_stock_price(df)

    df = calculate_moving_average(df)
    df = calculate_rsi(df)
    df = calculate_bollinger_bands(df)
    df = calculate_macd(df)

    print("\n[----------------General Information----------------]\n")
    get_current_price(stock_symbol)
    print(tabulate(yf.Ticker(stock_symbol).calendar.head(1), tablefmt="plain"))
    get_analyst_rec(stock_symbol)

    print("\n[----------------Technical Indicators----------------]\n")
    indicators = [["Moving Average 50 days:", "{:.2f}".format(df['sma_50'].tail(1).values[0])],
                  ["MACD:", "{:.4f}".format(df['macd'].tail(1).values[0])],
                  ["MACD Signal:", "{:.4f}".format(df['signal'].tail(1).values[0])],
                  ["MACD Histogram:", "{:.4f}".format(df['histogram'].tail(1).values[0])],
                  ["Relative Strength Index", "{:.2f}".format(df['rsi'].tail(1).values[0])],
                  ["Bollinger Bands", "Upper: {:.2f}, Middle: {:.2f}, Lower: {:.2f}".format(
                      df['upperband'].tail(1).values[0], df['middleband'].tail(1).values[0], df['lowerband'].tail(1).values[0])]]
    print(tabulate(indicators, headers=["Indicator", "Value"], tablefmt="double_outline"))

    df = calculate_fibonacci_retracement(df)
    print("\n[Fibonacci Retracement Levels]\n")
    table = [["23.6%", df['0.236'].tail(1).values[0]],
            ["38.2%", df['0.382'].tail(1).values[0]],
            ["50.0%", df['0.500'].tail(1).values[0]],
            ["61.8%", df['0.618'].tail(1).values[0]]]
    print(tabulate(table, headers=["Level","Value"], tablefmt="double_outline"))

if __name__ == "__main__":
    main()
