import time
import os
import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
    c = df.columns.difference(['xr'])
    df[c] = df[c].ffill()
    df['xr'] = df['xr'].ffill()
    del df['id']
    df = df.set_index('date').asfreq('D')

    print(df.head())
    if id_ == "GBPUSD":
        df["xr"] = 1/df["xr"]   # only for usdgbp to gbpusd

    train_size = round(0.7*len(df))
    train = df[:train_size]
    test = df[train_size:]

    # test for white noise
    from pandas.plotting import autocorrelation_plot

    autocorrelation_plot(df["xr"])

    figures_autocorrelation_directory = "./figures/autocorrelation/"
    if not os.path.exists(figures_autocorrelation_directory):
        os.makedirs(figures_autocorrelation_directory)

    plt.title(f"{id_} Autocorrelation")
    plt.savefig(f"./figures/autocorrelation/{id_}_autocorrelation.png")
    plt.close()

    # test for stationary
    from statsmodels.tsa.stattools import adfuller
    adfullerreport = adfuller(df["xr"])
    adfullerreportdata = pd.DataFrame(
        adfullerreport[0:4],
        columns = ["Values"],
        index=[
            "ADF F% statistics",
            "P-value",
            "No. of lags used",
            "No. of observations"
        ]
    )
    fuller_report = adfullerreportdata
    print(fuller_report)

    results_directory = "./results/adfuller/"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    fuller_report.to_csv(f"./results/adfuller/{id_}_adfuller.csv", index=True)

    train0 = train
    test0 = test

    # if the series need diferencing
    if id_ != "USDCHF":
        train = train.diff()[1:]
        test = test.diff()[1:]
        df = df.diff()[1:]

    # check autocorrelation
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(df["xr"])
    plt.xlabel("Lag")
    plt.ylabel("ACF")

    figures_acf_directory = "./figures/acf/"
    if not os.path.exists(figures_acf_directory):
        os.makedirs(figures_acf_directory)

    plt.title(f"{id_} ACF")
    plt.savefig(f"./figures/acf/{id_}_acf.png")
    plt.close()

    # check partial autocorrelation

    from statsmodels.graphics.tsaplots import plot_pacf
    plot_pacf(df["xr"])
    plt.xlabel("Lag")
    plt.ylabel("PACF")

    figures_pacf_directory = "./figures/pacf/"
    if not os.path.exists(figures_pacf_directory):
        os.makedirs(figures_pacf_directory)

    plt.title(f"{id_} PACF")
    plt.savefig(f"./figures/pacf/{id_}_pacf.png")
    plt.close()

    # build the ARIMA model

    # hyperparameter optimization
    # se elijen los mejores parametros basados en el mas bajo AIC

    from statsmodels.tsa.arima.model import ARIMA as arima
    from numpy import cumsum

    order = (9,9,9)
    if id_ == "USDCAD":
        order = (0,1,1)
    elif id_ == "USDJPY":
        order = (1,0,1)
    elif id_ == "USDCHF":
        order = (1,0,1)
    elif id_ == "GBPUSD":
        order = (8,1,1)
    elif id_ == "EURUSD":
        order = (0,1,1)

    from statsmodels.tsa.arima_model import ARIMA

    # entrena el modelo arima
    arima_model = ARIMA(train0["xr"], order=order) # A ser variado por el de AIC mas bajo
    arima_fitted = arima_model.fit()
    summary = arima_fitted.summary()
    print(summary)

    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, str(summary), {'fontsize': 16}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()

    results_summary_directory = "./results/summary/"
    if not os.path.exists(results_summary_directory):
        os.makedirs(results_summary_directory)

    plt.savefig(f"./results/summary/{id_}_summary.png", bbox_inches='tight')
    plt.close()

    # plot residuals
    residuals = pd.DataFrame(arima_fitted.resid, columns=['residuals'])
    residuals.plot(kind='kde')

    residuals_figures_directory = "./figures/residuals/"
    if not os.path.exists(residuals_figures_directory):
        os.makedirs(residuals_figures_directory)

    plt.title(f"{id_} ARIMA {order} Residuals")
    plt.savefig(f"./figures/residuals/{id_}_residuals.png")
    plt.close()

    # predict
    fc, se, conf = arima_fitted.forecast(len(test0), alpha=0.05)
    fc_series = pd.Series(fc, index=test0.index)

    lower_series = pd.Series(conf[:, 0], index=test0.index)
    upper_series = pd.Series(conf[:, 1], index=test0.index)
    plt.plot(train0, label="Training",color="navy")
    plt.plot(test0, label="Actual",color="orange")
    plt.plot(fc_series, label="Forecast",color="red")
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='gray')
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("xr")
    plt.title(f"{id_} ARIMA {order} Forecast")

    forecast_figures_directory = "./figures/forecast/"
    if not os.path.exists(forecast_figures_directory):
        os.makedirs(forecast_figures_directory)

    plt.savefig(f"./figures/forecast/{id_}_forecast.png")
    plt.close()


    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # report performance
    mse = mean_squared_error(test0, fc)
    print('MSE: '+str(mse))
    mae = mean_absolute_error(test0, fc)
    print('MAE: '+str(mae))

    with open(f"./results/arima_{id_}.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if counter == 0:
            writer.writerow(["index", "mae"])
        for index in range(len(fc)):
            writer.writerow([index + 1, np.abs(fc[index] - test0["xr"].to_list()[index])])

    with open(f"./results/errors.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if counter == 0:
            writer.writerow(["id", "(p,d,q)", "mse", "mae", "time (s)"])
        writer.writerow([id_, order, round(mse, 3), round(mae, 3), round(time.time() - start_time, 3)])
