import csv
import os
import time

from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import tensorflow as tf


ids = {"USDCAD", "USDCHF", "USDJPY", "EURUSD", "USDGBP"}
outer_counter = -1
for lr in [0.001, 0.005, 0.01, 0.05, 0.09, 0.1, 0.5]:
    for EPOCHS in [30, 300, 3000, 5000]:
        counter = -1
        for id_ in ids:
            counter += 1
            outer_counter += 1
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

            print(df.head())
            if id_ == "GBPUSD":
                df["xr"] = 1/df["xr"]   # only for usdgbp to gbpusd
            #df["xr"].plot()
            #plt.show()

            BATCH_SIZE = 64


            train_size = round(0.7*len(df))
            train_df = df[:train_size]
            test_df = df[train_size:]

            train = df["xr"][:train_size].to_numpy()
            test = df["xr"][train_size:].to_numpy()

            # define the keras model
            model = Sequential()
            model.add(Dense(256, input_dim=1, activation='relu'))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(1, activation='linear'))
            # compile the keras model
            model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])
            # fit the keras model on the dataset
            model.fit(train, train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
            # make class predictions with the model
            predictions = model.predict(test)

            whole_df = pd.Series(df["xr"].to_numpy(), index=df.index)
            predictions0 = pd.Series(predictions.flatten(), index=test_df.index)

            plt.plot(whole_df)
            plt.plot(predictions0)

            plt.xlabel("Date [Daily]")
            plt.ylabel("xr")
            plt.legend(["Actual xr", "Predicted xr"])
            plt.title(f"ANN {id_} LR {lr} EPOCHS {EPOCHS}")
            #plt.show()
            figures_directory = f"./figures/{lr}/{EPOCHS}/"
            if not os.path.exists(figures_directory):
                os.makedirs(figures_directory)

            plt.savefig(f"./figures/{lr}/{EPOCHS}/{id_}.png")
            plt.close()

            # report performance
            mse = mean_squared_error(test, predictions)
            mae = mean_absolute_error(test, predictions)

            results_directory = f"./results/{lr}/{EPOCHS}/"
            if not os.path.exists(results_directory):
                os.makedirs(results_directory)

            finish_time = time.time()

            with open(f"./results/{id_}_final_summary.csv", "a", newline='') as f:
                writer = csv.writer(f)
                if outer_counter == 0:
                    writer.writerow(["id", "lr", "epochs", "mse", "mae", "time (s)"])
                writer.writerow([id_, lr, EPOCHS, mse, mae, round(finish_time - start_time, 3)])


            with open(f"./results/{lr}/{EPOCHS}/errors.csv", "a", newline='') as f:
                writer = csv.writer(f)
                if counter == 0:
                    writer.writerow(["id", "lr", "mse", "mae", "time (s)"])
                writer.writerow([id_, lr, mse, mae, round(finish_time - start_time, 3)])
