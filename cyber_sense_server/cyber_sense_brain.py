from datetime import datetime
import numpy as np
import tensorflow as tf
from keras.models import load_model
from pandas import DataFrame
from pandas import concat
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

# Feature and TimeStep setup
time_step = 60
number_of_features = 13
normalizer_model = "dnn-models/normalizer.save"
dnn_model_file = "dnn-models/lstm_60t_120b"


# Prepare the test cross_validation_data for time step wise distribution
def prepare_timestep_data(data, lookback=1, dropnan=True):
    number_of_var = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # X sequence (t-n, ... t-1)
    for i in range(lookback, 0, -1):
        cols.append(df.shift(i))
        names += [('X%d(t-%d)' % (j + 1, i)) for j in range(number_of_var)]

    # put it all together
    aggregated_data = concat(cols, axis=1)
    aggregated_data.columns = names
    # drop rows with NaN values
    if dropnan:
        aggregated_data.dropna(inplace=True)

    return aggregated_data


def normalize_test_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(data)
    return normalized


def reshape_test_data(test_x):
    # split into train and test sets
    values = test_x.values
    # split into input and outputs
    number_of_observation = time_step * number_of_features
    test_x = values[:, :number_of_observation]
    # reshape input to be 3D [samples, timesteps, features]
    test_x = test_x.reshape((test_x.shape[0], time_step, number_of_features))

    return test_x


def get_time():
    d = datetime.today()
    time_date = d.strftime("%d_%B_%Y_%H_%M_%S")
    return time_date


def make_prediction(model, test_x, normalizer_file):
    # Make a prediction
    scaler = joblib.load(normalizer_file)
    yhat = model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], number_of_features * time_step))
    inv_yhat = np.concatenate((yhat, test_x[:, -number_of_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    return inv_yhat


def predict_cybersickness(minute_wise_data, model):
    t = datetime.now()
    start_time = t.strftime('%m-%d-%Y-%H-%M-%S')
    normalized_data = normalize_test_data(minute_wise_data)
    test_data = prepare_timestep_data(data=normalized_data, lookback=time_step)
    test_data.to_csv("results/test_data-" + start_time + "-simulation.csv")
    test_data = reshape_test_data(test_data)
    predicted_y = make_prediction(model, test_data, normalizer_model)

    with open("results/predicted.csv", "ab") as f:
        np.savetxt(f, predicted_y, delimiter=",", fmt='%.2f',
                   header='Predicted_Sickness')

    avg_sick_score = "{:.2f}".format(np.average(predicted_y))  # Format for 2 decimal point
    print("Predicted Average Sickness Score: ", avg_sick_score)

    return str(avg_sick_score)


def load_model_from_file():
    # Tensorflow Config
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    # Load model from file
    loaded_model = load_model(dnn_model_file + '.h5')
    print("Loaded DNN Model from disk..")
    print(loaded_model.summary())
    return loaded_model
