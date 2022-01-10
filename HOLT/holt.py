import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import Holt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def holt(y,y_to_train,y_to_test,smoothing_level,smoothing_slope, predict_date):

    fit1 = Holt(y_to_train).fit(smoothing_level, smoothing_slope, optimized=False)
    fcast1 = fit1.forecast(predict_date).rename("Holt's linear trend")
    mse1 = mean_squared_error(y_to_test, fcast1)
    mae1 = mean_absolute_error(y_to_test, fcast1)

    fit2 = Holt(y_to_train, exponential=True).fit(smoothing_level, smoothing_slope, optimized=False)
    fcast2 = fit2.forecast(predict_date).rename("Exponential trend")
    mse2 = mean_squared_error(y_to_test, fcast2)
    mae2 = mean_absolute_error(y_to_test, fcast2)

    return (mse1, mse2, mae1, mae2, fit1, fit2, fcast1, fcast2, smoothing_level, smoothing_slope)

def plot_custom(y, fit1, fit2, fcast1, fcast2, id_, smoothing_level, smoothing_slope):
    y.plot(color='green', legend=True)
    fcast1.plot(color='blue', legend=True)
    fcast2.plot(color='red', legend=True)
    fit1.fittedvalues.plot(color='blue', legend=False)
    fit2.fittedvalues.plot(color='red', legend=False)

    graphs_directory = "./graphs/"
    if not os.path.exists(graphs_directory):
        os.makedirs(graphs_directory)

    plt.savefig(f"./graphs/{id_}_alpha_{smoothing_level}_beta_{smoothing_slope}.png")
    plt.close()

counter = -1
ids = {"USDCAD", "USDJPY", "EURUSD", "USDGBP", "USDCHF"}

for id_ in ids:
    start_time = time.time()
    counter += 1
    df = pd.read_csv(
        f"./data/{id_}_2010-01-01_2021-01-01.csv",
        names=["id", "date", "xr"],
    )

    if id_ == "USDGBP":
        id_ = "GBPUSD"

    # treat the data to use correct dates and replace NaNs with ffill
    df.date = pd.to_datetime(df.date)
    df = df.set_index('date').asfreq('1D').reset_index()
    c = df.columns.difference(['xr'])
    df[c] = df[c].ffill()
    df['xr'] = df['xr'].ffill()
    del df['id']
    df = df.set_index('date').asfreq('D')

    if id_ == "GBPUSD":
        df["xr"] = 1/df["xr"]   # only for usdgbp to gbpusd

    train_size = round(0.7*len(df))
    train_df = df[:train_size]
    train = 100*(df[:train_size].to_numpy())
    test = df[train_size:]

    results = []

    for i in np.arange(0.0,1.0,0.1):
        for j in np.arange(0.0,1.0,0.1):
            #mse, _ = holt(df, train_df,test,i,j,len(test))
            results.append(holt(df, train_df,test,i,j,len(test)))

    min_tuple_mse_1 = min(results, key = lambda t: t[0])
    min_tuple_mse_2 = min(results, key = lambda t: t[1])
    min_tuple_mae_1 = min(results, key = lambda t: t[2])
    min_tuple_mae_2 = min(results, key = lambda t: t[3])

    if [min_tuple_mse_2[:4], min_tuple_mae_1[:4], min_tuple_mae_2[:4]].count(min_tuple_mse_1[:4]) == 3:
        min_tuple_mse = min_tuple_mse_1
        plot_custom(df, min_tuple_mse_1[4], min_tuple_mse_1[5], min_tuple_mse_1[6], min_tuple_mse_1[7], id_, min_tuple_mse_1[-2], min_tuple_mse_1[-1])
    else:
        #sort only by mse
        min_tuple_mse = min(results, key = lambda t: (t[0], t[1]))
        plot_custom(df, min_tuple_mse[4], min_tuple_mse[5], min_tuple_mse[6], min_tuple_mse[7], id_, min_tuple_mse[-2], min_tuple_mse[-1])


    results_directory = "./results/"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    with open(f"./results/errors.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if counter == 0:
            writer.writerow(["id", "Trend", "mse", "mae", "time (s)"])
        end = round(time.time() - start_time, 3)
        writer.writerow([id_, "Linear", min_tuple_mse[0], min_tuple_mse[2], end])
        writer.writerow([id_, "Exponential", min_tuple_mse[1], min_tuple_mse[3], end])

    with open(f"./results/holt_{id_}.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if counter == 0:
            writer.writerow(["index", "mae"])
        forecast = min_tuple_mse[7] # exponential
        if id_ == "EURUSD":
            forecast = min_tuple_mse[6] # linear

        for index in range(len(test["xr"])):
            writer.writerow([index + 1, np.abs(forecast[index] - test["xr"].to_list()[index])])
