import os
import numpy as np
from matplotlib import pylab as plt
from tools import matrix_to_array

def evaluate_ts(features, y_true, y_pred, symbol, time_dimension, test_id):
    print("Evaluation of the predictions:")
    mse = np.mean(np.square(y_true - y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    print("MSE:", mse)
    print("mae:", mae)

    print("Benchmark: if prediction == last feature")
    print("MSE:", np.mean(np.square(features[:, -1] - y_true)))
    print("mae:", np.mean(np.abs(features[:, -1] - y_true)))

    plt.plot(matrix_to_array(y_true), 'b')
    plt.plot(matrix_to_array(y_pred), 'r--')
    plt.xlabel("Days")
    plt.ylabel("Predicted and true values")
    plt.title("Predicted (Red) VS Real (Blue)")
    #plt.show()
    predictions_directory = f'./results/{test_id}/graphs/predictions/'
    if not os.path.exists(predictions_directory):
        os.makedirs(predictions_directory)

    plt.savefig(f'./results/{test_id}/graphs/predictions/{symbol}_{time_dimension}_prediction.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    error = np.abs(matrix_to_array(y_pred) - matrix_to_array(y_true))
    plt.plot(error, 'r')
    fit = np.polyfit(range(len(error)), error, deg=1)
    plt.plot(fit[0] * range(len(error)) + fit[1], '--')
    plt.xlabel("Days")
    plt.ylabel("Prediction error L1 norm")
    plt.title("Prediction error (absolute) and trendline")
    #plt.show()
    errors_directory = f'./results/{test_id}/graphs/errors/'
    if not os.path.exists(errors_directory):
        os.makedirs(errors_directory)

    plt.savefig(f'./results/{test_id}/graphs/errors/{symbol}_{time_dimension}_error.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    return (mae, mse)
