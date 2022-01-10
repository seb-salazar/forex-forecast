import time
import os
import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

counter = -1
ids = {"USDCAD", "USDCHF", "USDJPY", "EURUSD", "USDGBP"}

for id_ in ids:
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
    train = df[:train_size]
    test = df[train_size:]
    entire_data = df
    test_df = df[train_size:]

    from numpy import mean
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # prepare situation
    X = test_df["xr"]
    windows = [3,]
    for window in windows:
        counter += 1
        start_time = time.time()
        history = [X[i] for i in range(window)]
        test = [X[i] for i in range(window, len(X))]
        predictions = list()
        # walk forward over time steps in test
        for t in range(len(test)):
        	length = len(history)
        	yhat = mean([history[i] for i in range(length-window,length)])
        	obs = test[t]
        	predictions.append(yhat)
        	history.append(obs)

        mse = mean_squared_error(test, predictions)
        mae = mean_absolute_error(test, predictions)


        slicer = window

        actual_series = pd.Series(df["xr"], index=df.index)
        predictions_series = pd.Series(predictions, index=test_df[slicer:].index)

        # plot
        plt.plot(actual_series, label="Actual")
        plt.plot(predictions_series, color='red', label="Training")
        plt.legend(loc='upper left')
        plt.xlabel("Date")
        plt.ylabel("xr")

        figures_directory = "./figures/"
        if not os.path.exists(figures_directory):
            os.makedirs(figures_directory)

        plt.savefig(f"./figures/{window}_{id_}_ma.png")
        plt.close()

        # zoom plot
        plt.plot(pd.Series(test[:365], index=df[:365].index), label="Actual")
        plt.plot(pd.Series(predictions[:365], index=df[:365].index), color='red', label="Training")
        plt.legend(loc='upper left')
        plt.xlabel("Date")
        plt.ylabel("xr")

        plt.savefig(f"./figures/{window}_{id_}_ma_zoom.png")
        plt.close()

        results_directory = "./results/"
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        with open(f"./results/ma_{id_}.csv", "a", newline='') as f:
            writer = csv.writer(f)
            if counter == 0:
                writer.writerow(["index", "mae"])
            for index in range(len(predictions)):
                writer.writerow([index + 1, np.abs(predictions[index] - test[index])])

        with open(f"./results/errors.csv", "a", newline='') as f:
            writer = csv.writer(f)
            if counter == 0:
                writer.writerow(["id", "ma", "mse", "mae", "time (s)"])
            writer.writerow([id_, window, mse, mae, round(time.time() - start_time, 3)])
