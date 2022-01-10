'''
Script used for LSTM network training and prediction - Sebastian Salazar

Create a virtual environment, activate it and then run
pip install -r requirements.txt to install the required dependencies

Then, run the training sesssion by first changing the parameters on the config.py file
and then typing

python3 main.py

on the command line
'''

from evaluate_ts import evaluate_ts
from tensorflow.contrib import rnn
from tools import fetch_csv_exchange_rate, fetch_exchange_rate, format_dataset
from config import (
    test_id,
    time_dimension,
    learning_rate,
    n_epochs,
    n_embeddings,
)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
import time
import csv
import pandas as pd

# to log on Tensor board
import os
tf_logdir = f"./results/{test_id}/logs/tf/exchange_rate_lstm"
os.makedirs(tf_logdir, exist_ok=1)

tf.set_random_seed(101)


optimizer = tf.train.AdagradOptimizer

symbols = ["EURUSD", "USDCAD", "USDJPY", "USDGBP", "USDCHF",]
symbols_dict = {"USDCAD":"XUDLCDD", "USDJPY":"XUDLJYD", "USDGBP":"XUDLGBD", "USDCHF":"XUDLSFD", "EURUSD":"EURUSD"}

# Optional:
# to save the config file for reference for difference test for a same currency
# with open(f'../results/{test_id}/config.txt', "w") as file_:
#     file_.write(
#         f"""
#         test_id = {test_id}

#         time_dimension = {time_dimension}
#         train_size = "0.7*dataset"
#         test_size = "0.3*dataset"

#         learning_rate = {learning_rate}
#         n_epochs = {n_epochs}
#         n_embeddings = {n_embeddings}
#         """
#     )
counter = 0

for symbol in symbols:

    # for the MCS, after knowing which params are the "best"
    # if symbol == "USDCAD":
    #     learning_rate = 0.001
    #     n_epochs = 5000
    # elif symbol == "USDJPY":
    #     learning_rate = 0.09
    #     n_epochs = 5000
    # elif symbol == "GBPUSD":
    #     learning_rate = 0.001
    #     n_epochs = 3000
    # elif symbol == "USDCHF":
    #     learning_rate = 0.5
    #     n_epochs = 3000
    # elif symbol == "EURUSD":
    #     learning_rate = 0.001
    #     n_epochs = 30

    fetch_csv_exchange_rate(test_id, symbols_dict[symbol], datetime.date(2010, 1, 1), datetime.date(2021, 1, 1))

    tf.reset_default_graph()

    start = time.time()

    df = pd.read_csv(
        f"./data/{symbol}_2010-01-01_2021-01-01.csv",
        names=["id", "date", "xr"],
    )

    if symbol == "USDGBP":
        symbol = "GBPUSD"

    # treat the data to use correct dates and replace NaNs with ffill
    df.date = pd.to_datetime(df.date)
    df = df.set_index('date').asfreq('1D').reset_index()
    c = df.columns.difference(['xr'])
    df[c] = df[c].ffill()
    df['xr'] = df['xr'].ffill()
    del df['id']
    df = df.set_index('date').asfreq('D')

    if symbol == "GBPUSD":
        df["xr"] = 1/df["xr"]   # only for usdgbp to gbpusd

    exchange_rate_values = df["xr"].tolist()

    train_size = round(0.7*len(df))
    test_size = len(df) - train_size

    minibatch_exch_x, minibatch_exch_y = format_dataset(exchange_rate_values, time_dimension)
    train_X = minibatch_exch_x[:train_size, :].astype(np.float32)
    train_y = minibatch_exch_y[:train_size].reshape((-1, 1)).astype(np.float32)
    test_X = minibatch_exch_x[train_size:, :].astype(np.float32)
    test_y = minibatch_exch_y[train_size:].reshape((-1, 1)).astype(np.float32)
    train_X_ts = train_X[:, :, np.newaxis]
    test_X_ts = test_X[:, :, np.newaxis]

    X_tf = tf.placeholder("float", shape=(None, time_dimension, 1), name="X")
    y_tf = tf.placeholder("float", shape=(None, 1), name="y")

    def RNN(x, weights, biases):
        with tf.name_scope("LSTM"):
            x_ = tf.unstack(x, time_dimension, 1)
            lstm_cell = rnn.BasicLSTMCell(n_embeddings)
            outputs, _ = rnn.static_rnn(lstm_cell, x_, dtype=tf.float32)
            return tf.add(biases, tf.matmul(outputs[-1], weights))

    weights = tf.Variable(tf.truncated_normal([n_embeddings, 1], mean=0.0, stddev=10.0), name="weights")
    biases = tf.Variable(tf.zeros([1]), name="bias")
    y_pred = RNN(X_tf, weights, biases)
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.square(y_tf - y_pred))
        train_op = optimizer(learning_rate).minimize(cost)
        tf.summary.scalar("MSE", cost)
        with tf.name_scope("mae"):
            mae_cost = tf.reduce_mean(tf.abs(y_tf - y_pred))
            tf.summary.scalar("mae", mae_cost)

    # Exactly as before, this is the main loop.
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(tf_logdir, sess.graph)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        # For each epoch, the whole training set is feeded into the tensorflow graph
        for i in range(n_epochs):
            summary, train_cost, _ = sess.run([merged, cost, train_op], feed_dict={X_tf:
                                                train_X_ts, y_tf: train_y})
            writer.add_summary(summary, i)
            if i%100 == 0:
                print("Training iteration", i, "MSE", train_cost)
        # After the training, let's check the performance on the test set
        test_cost, y_pr = sess.run([cost, y_pred], feed_dict={X_tf: test_X_ts, y_tf:
                                    test_y})
        print("Test dataset:", test_cost)

        end = time.time()

        # Evaluate the results
        evaluation = evaluate_ts(test_X, test_y, y_pr, symbol, time_dimension, test_id)
        mae = float("{0:.3f}".format(evaluation[0]))
        mse = float("{0:.3f}".format(evaluation[1]))

        tables_directory = f'./results/{test_id}/tables/'
        if not os.path.exists(tables_directory):
            os.makedirs(tables_directory)

        with open(f"./results/{test_id}/lstm_{symbol}.csv", "a", newline='') as f:
            writer = csv.writer(f)
            if counter == 0:
                writer.writerow(["index", "mae"])
            for index in range(len(test_y)):
                writer.writerow([index + 1, np.abs(y_pr[index][0] - test_y[index][0])])

        with open(f'./results/{test_id}/tables/results.csv', 'a') as f:
            if counter == 0:
                f.write("symbol, time_dimension, train_size, test_size, learning_rate, n_epochs, n_embeddings, mae, mse, elapsed_time}"+'\n')

            f.write(f'{symbol},{time_dimension},{train_size},{test_size},{learning_rate},{n_epochs},{n_embeddings},{mae},{mse},{round(end-start, 2)}'+'\n')

        counter += 1

# launch logs of training behaviour with
# tensorboard --logdir=../results/{test_id}/logs/tf/exchange_rate_lstm
# at
# localhost:6006
