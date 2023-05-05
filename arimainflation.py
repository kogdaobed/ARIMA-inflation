import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima


data = pd.read_csv('inflation_data.csv', index_col='DATE', parse_dates=True) #загружаю данные

plt.plot(data)
plt.title('Inflation Time Series')
plt.show() #график временного ряда



model = auto_arima(data, seasonal=False, trace=True)
print(model) #определила порядок AR(1) I(1) MA(1) с минимальным AIC(3069.417)

model = ARIMA(data, order=(1,1,1))
model_fit = model.fit() #обучила модель

forecast = model_fit.forecast(steps=3)
print(forecast) #прогноз на 3 месяца


plt.plot(forecast, label='forecast')
plt.legend(loc='best')
plt.show() #график прогноза
