import time
import os
import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pmdarima
import arch
from arch.__future__ import reindexing


# evaluate an ARIMA model using a walk-forward validation
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
# first approach, no returns

counter = -1
ids = {"USDJPY", "USDCAD", "USDCHF", "EURUSD", "USDGBP"}

for id_ in ids:
    counter += 1
    start_time = time.time()
    df = pd.read_csv(
        f"./data/{id_}_2010-01-01_2021-01-01.csv",
        names=["id", "date", "xr"],
    )

    if id_ == "USDGBP":
        id_ = "GBPUSD"

    # treat the data to use correct dates and replace NaNs with ffill
    df.date = pd.to_datetime(df.date)
    df = df.set_index('date').asfreq('1D').reset_index()
    #df.date = pd.to_numeric(df.date)
    c = df.columns.difference(['xr'])
    df[c] = df[c].ffill()
    df['xr'] = df['xr'].ffill()
    del df['id']
    df = df.set_index('date').asfreq('D')

    if id_ == "GBPUSD":
        df["xr"] = 1/df["xr"]   # only for usdgbp to gbpusd

    train_size = round(0.7*len(df))

    train0 = df[:train_size]
    test0 = df[train_size:]

    df = df.values

    train = df[:train_size]
    test = df[train_size:]

    order = (9,9,9)
    if id_ == "USDCAD":
        order = (0,1,1)
        garch_order = (2,3)
    elif id_ == "USDJPY":
        order = (1,0,1)
        garch_order = (1,3)
    elif id_ == "USDCHF":
        order = (1,0,1)
        garch_order = (2,0)
    elif id_ == "GBPUSD":
        order = (8,1,1)
        garch_order = (1,1)
    elif id_ == "EURUSD":
        order = (0,1,1)
        garch_order = (1,3)


    history = [x for x in train]
    arima_predictions = list()
    arima_garch_predictions = []
    residuals = []
    # walk-forward validation
    for t in range(len(test)):
        # ARIMA
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()

        yhat = output[0]
        arima_predictions.append(yhat)

        # GARCH
        # fit a GARCH model on the residuals of the ARIMA model
        residual = model_fit.resid
        residuals.append(residual[0])
        residual *= 100
        garch = arch.arch_model(residual, p=garch_order[0], q=garch_order[1])
        garch_fitted = garch.fit()

        garch_forecast = garch_fitted.forecast(horizon=1)
        predicted_et = garch_forecast.mean['h.1'].iloc[-1]
        # Combine both models' output: yt = mu + et
        predicted_et /= 100
        arima_garch_prediction = yhat + predicted_et
        arima_garch_predictions.append(arima_garch_prediction)

        obs = test[t]
        history.append(obs)

    # ARIMA
    # evaluate forecasts
    mse_arima = mean_squared_error(test, arima_predictions)
    mae_arima = mean_absolute_error(test, arima_predictions)

    fc_series_arima = pd.Series(arima_predictions, index=test0.index)
    plt.plot(train0, label="Training",color="navy")
    plt.plot(test0, label="Actual",color="orange")
    plt.plot(fc_series_arima, label="Forecast",color="red")
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("xr")
    plt.title(f"{id_} ARIMA {order} Forecast")

    arima_figures_directory = "./figures/arima/"
    if not os.path.exists(arima_figures_directory):
        os.makedirs(arima_figures_directory)

    plt.savefig(f"./figures/arima/{id_}_arima_forecast.png")
    plt.close()

    results_directory = "./results/"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    with open(f"./results/errors_arima.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if counter == 0:
            writer.writerow(["id", "(p,d,q)", "mse", "mae", "time (s)"])
        writer.writerow([id_, order, round(mse_arima, 3), round(mae_arima, 3), round(time.time() - start_time, 3)])


    # GARCH

    # evaluate forecasts
    mse = mean_squared_error(test, arima_garch_predictions)
    mae = mean_absolute_error(test, arima_garch_predictions)
    print('MSE: '+str(mse))
    print('MAE: '+str(mse))

    fc_series_arima_garch = pd.Series(arima_garch_predictions, index=test0.index)
    plt.plot(train0, label="Training",color="navy")
    plt.plot(test0, label="Actual",color="orange")
    plt.plot(fc_series_arima_garch, label="Forecast",color="red")
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("xr")
    plt.title(f"{id_} ARIMA-GARCH{garch_order} Forecast")

    arima_garch_figures_directory = "./figures/arima-garch/"
    if not os.path.exists(arima_garch_figures_directory):
        os.makedirs(arima_garch_figures_directory)

    plt.savefig(f"./figures/arima-garch/{id_}_arima-garch_forecast.png")
    plt.close()

    with open(f"./results/arima-garch_{id_}.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if counter == 0:
            writer.writerow(["index", "mae"])
        for index in range(len(fc_series_arima_garch)):
            writer.writerow([index + 1, np.abs(fc_series_arima_garch[index] - test0["xr"].to_list()[index])])

    with open(f"./results/errors_arima-garch.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if counter == 0:
            writer.writerow(["id", "(P,Q)", "mse", "mae", "time (s)"])
        writer.writerow([id_, garch_order, round(mse, 3), round(mae, 3), round(time.time() - start_time, 3)])
