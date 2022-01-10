import os
import numpy as np
import datetime
import pickle
import quandl
import csv
from datetime import datetime

quandl.ApiConfig.api_key = "RpFtW9CK1zzVDzbNwGhb"

def matrix_to_array(m):
    return np.asarray(m).reshape(-1)

def format_dataset(values, temporal_features):
    feat_splits = [values[i:i + temporal_features] for i in range(len(values) - temporal_features)]
    feats = np.vstack(feat_splits)
    labels = np.array(values[temporal_features:])
    return feats, labels

# an example of the data used
#  http://www.quandl.com/api/v3/datasets/BOE/XUDLJYS/data.csv Spot exchange rate
#  http://www.quandl.com/api/v3/datasets/BOE/XUDLCDD/data.csv
#  BOE: Bank of england

def date_obj_to_str(date_obj):
    return date_obj.strftime('%Y-%m-%d')

def save_pickle(something, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as fh:
        pickle.dump(something, fh, pickle.DEFAULT_PROTOCOL)

def load_pickle(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)

def fetch_exchange_rate(symbol,
                      from_date,
                      to_date,
                      cache_path="./tmp/exchange_rates/"):
    assert(from_date <= to_date)
    filename = "{}_{}_{}.pk".format(symbol, str(from_date), str(to_date))
    price_filepath = os.path.join(cache_path, filename)
    try:
        prices = load_pickle(price_filepath)
        print("loaded from", price_filepath)
    except IOError:
        if symbol[:3] == "EUR":
            source_bank_path = "ECB/"
        else:
            source_bank_path = "BOE/"

        historic = quandl.get(source_bank_path + symbol,
        start_date=date_obj_to_str(from_date),
        end_date=date_obj_to_str(to_date))
        prices = historic["Value"].tolist()
        if symbol == "XUDLGBD":
            prices = list(map(lambda x: round(1/x, 4), prices))
        save_pickle(prices, price_filepath)
        print("saved into", price_filepath)

    return prices


def fetch_csv_exchange_rate(
                        test_id,
                        symbol,
                        from_date,
                        to_date,) -> int:
    assert(from_date <= to_date)

    csvs_directory = "./results/{}/csvs/".format(test_id)
    if not os.path.exists(csvs_directory):
        os.makedirs(csvs_directory)

    filename = "./results/{}/csvs/{}_{}_{}.csv".format(test_id, symbol, str(from_date), str(to_date))
    if symbol[:3] == "EUR":
        source_bank_path = "ECB/"
    else:
        source_bank_path = "BOE/"
    historic = quandl.get(source_bank_path + symbol,
    start_date=date_obj_to_str(from_date),
    end_date=date_obj_to_str(to_date), returns="numpy")
    #historic[:, 0] = datetime.fromtimestamp(historic[:, 0])
    print(historic)
    num_rows = len(historic)

    historic_lists = [list(x) for x in historic]
    for count, item in enumerate(historic_lists):
        item.insert(0, count + 1)

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for row in historic_lists:
            writer.writerow(row)

    return num_rows
