from datetime import datetime
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.models import load_model
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.regularizers import l2


def load_data_from_csv(file_name):
    dataset = read_csv(file_name, header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')
    return values


def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(data)
    joblib.dump(scaler, scaler_filename)  # Save the Scaler to normalize the test data later

    return normalized


def fit_test_data_to_normalizer(data, file_name):
    scaler = joblib.load(file_name)

    return scaler.fit_transform(data)


def save_network(model, file_name):
    # serialize weights to HDF5
    model.save(file_name + ".h5")
    print("Saved model to disk")


def load_model_from_file(file_name):
    loaded_model = load_model(file_name + '.h5')
    print("Loaded Model from disk")
    print(loaded_model.summary())
    return loaded_model


def get_time():
    d = datetime.today()
    time_date = d.strftime("%d_%B_%Y_%H_%M_%S")
    return time_date


# Prepare the individual_raw_data for DL Regression
def prepare_data_for_dl(data, lookback=1, output=1, drop_na=True):
    number_of_var = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    # X sequence (t-n, ... t-1)
    for i in range(lookback, 0, -1):
        cols.append(df.shift(i))
        names += [('X%d(t-%d)' % (j + 1, i)) for j in range(number_of_var)]

    # target sequence (t, t+1, ... t+n)
    for i in range(0, output):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('Y%d(t)' % (j + 1)) for j in range(number_of_var)]
        else:
            names += [('Y%d(t+%d)' % (j + 1, i)) for j in range(number_of_var)]

    # aggregate everything together
    agg = concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if drop_na:
        agg.dropna(inplace=True)

    # Ignore X1 as it is actually the target
    agg = agg[agg.columns[~agg.columns.to_series().str.contains(pat='X1\(')]]

    return agg


def split_data_x_data_y(data, time_step, number_of_feature):
    values = data.values
    number_of_observation = time_step * number_of_feature
    print("Number of Observation: ", number_of_observation)

    # Split into validation and train set
    train = values[:]
    print("Total train cross_validation_data: ", len(train))

    # Split into input and outputs
    train_x, train_y = train[:, :number_of_observation], train[:, -(number_of_features + 1)]

    # Reshape input to be 3D [samples, time steps, features]
    train_x = train_x.reshape((train_x.shape[0], time_step, number_of_features))
    print("Train X shape: ", train_x.shape)
    print("Train Y shape: ", train_y.shape)

    return train_x, train_y


def simple_lstm(train_X, train_y, test_x, test_y):
    model = Sequential()
    model.add(LSTM(60, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(30))
    model.add(Dense(8))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0001))
    model.summary()
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=120, verbose=1,
                        shuffle=False, validation_split=0.1)

    return history, model


def cnn_lstm(train_X, train_y):
    model = Sequential()
    model.add(Conv1D(filters=30, kernel_size=8, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(32, recurrent_dropout=0.20, return_sequences=True))
    model.add(LSTM(16, recurrent_dropout=0.20))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.0001, clipnorm=1.))
    model.summary()
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=120, validation_split=0.2,
                        verbose=1, shuffle=False)

    return history, model


def plot_network_history(history, file):
    # plot history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b-o', label='Training loss')
    plt.plot(epochs, val_loss, 'r-*', label='Validation Loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(min(epochs), max(epochs) + 1, 5.0))
    plt.legend()
    plt.savefig(file)
    plt.show()


def plot_predictions(actual, predicted):
    plt.figure()
    plt.plot(actual, 'b-o', label='Actual')
    plt.plot(predicted, 'r-*', label='Predicted')
    plt.title('Actual and Predicted SR_t score')
    plt.xlabel('Number of Observations')
    plt.ylabel('SR_t score')
    plt.legend()
    plt.savefig(predicted_vs_actual_fig_file)
    plt.show()


def evaluate_model(model, test_X, test_y):
    # Evaluate the model
    test_loss = model.evaluate(test_X, test_y, verbose=1)
    print("Test Loss", test_loss)


def calculate_rmse_of_predictions(model, test_x, test_y):
    # get the predictions
    scaler = joblib.load(scaler_filename)
    yhat = model.predict(test_x)
    print("Shape test X: ", test_x.shape)
    test_x = test_x.reshape((test_x.shape[0], number_of_features * time_step))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_x[:, -number_of_features:]), axis=1)

    print("Shape yhat: ", yhat.shape)
    print("Shape test X (reshape): ", test_x.shape)
    print("Shape Concatenate: ", inv_yhat.shape)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    print("Predicted Y: ", inv_yhat)
    np.savetxt(predicted_csv_file, inv_yhat, delimiter=",", fmt='%.3f', header='predicted_fms')

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_x[:, -number_of_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    print("Actual Y: ")
    np.savetxt(actual_csv_file, inv_y, delimiter=",", fmt='%.3f', header='actual_fms')
    print(inv_y)

    # calculate RMSE
    rsqr = r2_score(inv_y, inv_yhat)
    mae = mean_absolute_error(inv_y, inv_yhat)
    mdae = median_absolute_error(inv_y, inv_yhat)
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    plot_predictions(actual=inv_y, predicted=inv_yhat)

    print('Test RMSE: %.3f' % rmse)
    print("TEST R square: %.3f" % rsqr)
    print("TEST MAE: %.3f" % mae)
    print("TEST Median AE: %.3f" % mdae)


def train_data():
    # Load and process train data
    dl_train_data = load_data_from_csv(train_data_file)
    dl_train_data = normalize_data(dl_train_data)
    processed_train_data = prepare_data_for_dl(dl_train_data, time_step, 1)

    # Load test data
    dl_test_data = load_data_from_csv(test_data_file)
    dl_test_data = fit_test_data_to_normalizer(dl_test_data, scaler_filename)
    processed_test_data = prepare_data_for_dl(dl_test_data, time_step, 1)

    # Splitting Train data
    train_X, train_y = split_data_x_data_y(processed_train_data, time_step, number_of_features)

    # Splitting Test data
    test_x, test_y = split_data_x_data_y(processed_test_data, time_step, number_of_features)

    # # Run Neural Nets and Save it
    history, model = simple_lstm(train_X, train_y, test_x, test_y)
    save_network(model, model_name)

    # # Plot learning history
    plot_network_history(history, learning_history_fig_file)

    # Load Model and make predictions from the test data
    loaded_model = load_model_from_file(model_name)
    calculate_rmse_of_predictions(loaded_model, test_x, test_y)


############################################ Run the Code ######################################
time_step = 60
number_of_features = 13
train_data_file = 'dnn_data/cross_validation_data/validation1/train_data.csv'
test_data_file = 'dnn_data/cross_validation_data/validation1/test_data.csv'

# Results and Model
model_name = "dnn_data/saved_model/validation1/lstm_60t_120b"
scaler_filename = "dnn_data/normalizer/validation1/normalizer.save"
learning_history_fig_file = 'dnn_data/results/validation1/train_vs_validation_loss.png'
predicted_vs_actual_fig_file = 'dnn_data/results/validation1/predicted_vs_actual.png'
predicted_csv_file = 'dnn_data/results/validation1/predicted.csv'
actual_csv_file = 'dnn_data/results/validation1/actual.csv'

# Run File
train_data()
