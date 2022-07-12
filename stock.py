from this import d
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

plt.style.use("dark_background")
df = pd.read_csv("/Users/jenishdhaduk/Desktop/PPSU/datascience_project/dataset/AAPL.csv")
df

#evaluation
x = df[['High']]
y = df['Low']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
model = LinearRegression()

model.fit(x_train, y_train)

print("Accuracy : ",model.coef_)

print(model.intercept_)

pd.DataFrame(model.coef_, x.columns, columns = ['Coeff'])

predictions = model.predict(x_test)

# plt.scatter(y_test, predictions)

plt.hist(y_test - predictions)

metrics.mean_absolute_error(y_test, predictions)

metrics.mean_squared_error(y_test, predictions)

np.sqrt(metrics.mean_squared_error(y_test, predictions))

plt.figure(figsize=(12, 5))
plt.plot(df['Adj Close'], label='Stock')
plt.title('Stock Adj Close Price History')
plt.xlabel("May 27,2014 - May 25,2020 ")
plt.ylabel("Adj Close Price USD ($)")
plt.legend(loc="upper left")
plt.show()

sma30 = pd.DataFrame()
sma30['Adj Close'] = df['Adj Close'].rolling(window=30).mean()
sma30

sma100 = pd.DataFrame()
sma100['Adj Close'] = df['Adj Close'].rolling(window=100).mean()
sma100

plt.figure(figsize=(12,5))
plt.plot(df['Adj Close'], label='Stock')
plt.plot(sma30['Adj Close'], label='SMA30')
plt.plot(sma100['Adj Close'], label='SMA100')
plt.title("Stock Adj. Close Price History")
plt.xlabel('May 27,2014 - May 25,2020')
plt.ylabel('Adj. Close Price USD($)')
plt.legend(loc='upper left')
plt.show()


data = pd.DataFrame()
data['df'] = df['Adj Close']
data['SMA30'] = sma30['Adj Close']
data['SMA100'] = sma100['Adj Close']
data

def buySell(data):
  sigPriceBuy = []
  sigPriceSell = []
  flag = -1
  for i in range(len(data)):
    if data ['SMA30'][i] > data['SMA100'][i]:
      if flag != 1:
        sigPriceBuy.append(data['df'][i])
        sigPriceSell.append(np.nan)
        flag = 1
      else:
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(np.nan)
    elif data['SMA30'][i] < data['SMA100'][i]:
      if flag != 0:
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(data['df'][i])
        flag = 0
      else:
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(np.nan)
    else:
      sigPriceBuy.append(np.nan)
      sigPriceSell.append(np.nan)
  return(sigPriceBuy, sigPriceSell)

buySell = buySell(data)
data['Buy Signal Price'] = buySell[0]
data['Sell Signal Price'] = buySell[1]
data

plt.style.use('classic')
plt.figure(figsize=(12,5))
plt.plot(data['df'], label='STOCK', alpha=0.35)
plt.plot(data['SMA30'], label='SMA30', alpha=0.35)
plt.plot(data['SMA100'],label='SMA100', alpha=0.35)
plt.scatter(data.index, data['Buy Signal Price'], label ='Buy', marker='^',color='green')
plt.scatter(data.index, data['Sell Signal Price'],label='Sell', marker='v', color='red')
plt.title('Stock Adj. Close Price History Buy and Sell Signals')
plt.xlabel("May 27,2014 - May 25,2020")
plt.ylabel("Adj Close Price USD($)")
plt.legend(loc='upper left')
plt.show()

