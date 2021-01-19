import glob
import os
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def get_hrbr_from_file(file_name):
    data = pd.read_csv(file_name, delimiter=',', names=["Date", "Time", "AM/PM", "HR", "BR"])
    data.drop([0], axis=0, inplace=True)
    data.Time = data['Time'] + ' ' + data['AM/PM']
    data.Date = data.Date + ' ' + data.Time
    data.drop(['AM/PM'], axis=1, inplace=True)
    data.drop(['Time'], axis=1, inplace=True)
    data.Date = [datetime.strptime(x, '%m/%d/%Y %H:%M:%S %p') for x in data.Date]
    data.Date = pd.to_datetime(data.Date)
    data.HR = pd.to_numeric(data.HR)
    data.BR = pd.to_numeric(data.BR)
    data.set_index('Date', inplace=True)

    return data


def get_hrv_from_file(file_name):
    """Get the heart rate variability from file"""
    data = pd.read_csv(file_name, delimiter=',', names=["Date", "Time", "AM/PM", "HRV"])
    data.drop([0], axis=0, inplace=True)
    data.Time = data['Time'] + ' ' + data['AM/PM']
    data.Date = data.Date + ' ' + data.Time
    data.drop(['AM/PM'], axis=1, inplace=True)
    data.drop(['Time'], axis=1, inplace=True)
    data.Date = [datetime.strptime(x, '%m/%d/%Y %H:%M:%S %p') for x in data.Date]
    data.Date = pd.to_datetime(data.Date)
    data.HRV = pd.to_numeric(data.HRV)
    data.set_index('Date', inplace=True)

    return data


def filter_data_within_date_range(data, start, end):
    """Trunc Data on date start and end date"""
    ind = (data.index >= start) & (data.index <= end)

    return data[ind]


def trunc_data_on_event(data, event_list):
    val = pd.DataFrame()
    total = len(event_list)
    i = 0
    j = 1
    while j < total:
        start = event_list[i]
        end = event_list[j]
        val = val.append(filter_data_within_date_range(data, start, end))
        i += 2
        j += 2

    return val


def get_event_list_from_file(file_name):
    lines_list = open(file_name).read().splitlines()
    return lines_list


def get_gsr_from_file(file_name, sample_rate, start):
    data = pd.read_csv(file_name, delimiter=',', names=["ETA", "GSR"])
    data.drop([0], axis=0, inplace=True)
    data.ETA = data.ETA.str[1:]  # Removing the leading ' from the first column
    data.GSR = pd.to_numeric(data.GSR)
    ret_val = data.groupby(np.arange(len(data)) // sample_rate).mean()
    period = len(ret_val.index)
    time = pd.date_range(start, periods=period, freq='S')
    ret_val.index = time

    return ret_val


def add_verbal_feedback_with_gsr(file, gsr_data, thirty_min_gsr):
    feedback = pd.read_csv(file, delimiter=',', names=["Feedback"])
    total = len(feedback)
    gsr_data.loc[0:, 'Feedback'] = feedback.iloc[0]['Feedback']
    gsr_data.loc[0:, 'Epoch'] = "0"
    thirty_min_gsr.loc[0:300, 'Feedback'] = feedback.iloc[1]['Feedback']
    thirty_min_gsr.loc[0:300, 'Epoch'] = "1"
    thirty_min_gsr.loc[301:357, 'Feedback'] = feedback.iloc[2]['Feedback']
    thirty_min_gsr.loc[301:357, 'Epoch'] = "2"

    start = 358
    end = 427
    i = 3
    loop = 3
    for i in range(3, total):
        thirty_min_gsr.loc[start:end, 'Feedback'] = feedback.iloc[i]['Feedback']
        thirty_min_gsr.loc[start:end, 'Epoch'] = str(loop)
        start = end + 1
        end += 69
        loop += 1

    thirty_min_gsr.loc[start:, 'Feedback'] = feedback.iloc[i]['Feedback']
    thirty_min_gsr.loc[start:, 'Epoch'] = "14"
    gsr_data = gsr_data.append(thirty_min_gsr)

    return gsr_data


def add_hrv_data(src, dest):
    if len(src) != len(dest):
        raise Exception("Src and Destination length did not matched.")
    src["HRV"] = dest["HRV"]

    return src


def get_mouse_click(file):
    data = pd.read_csv(file, delimiter=',', names=["Time", "Click"])
    return data


def remove_outliers_z_score(data, features):
    tmp_data = data[features]
    z = np.abs(stats.zscore(tmp_data))
    threshold = 3  # z score value for which we will consider it an outlier
    outliers = np.where(z > 3)  # Outliers where z value is greater than 3 std
    dataset_without_outliers = data[(z < threshold).all(axis=1)]

    return dataset_without_outliers


def process_individual_data(files, indi_count):
    event_log = []
    hr_data = []
    hrv_data = []
    mouse_click = []
    gsr_data = []
    thirty_min_gsr = []
    gsr_sample_rate = 10
    feed_back_file = ''
    for file in files:
        if 'event_log' in file:
            event_log = get_event_list_from_file(file)
        if 'hr_br' in file:
            hr_data = get_hrbr_from_file(file)
        elif 'hrv' in file:
            hrv_data = get_hrv_from_file(file)
        elif 'gsr' in file:
            if '5Min' in file:
                gsr_data = get_gsr_from_file(file, gsr_sample_rate, event_log[0])
            elif '30Min' in file:
                thirty_min_gsr = get_gsr_from_file(file, gsr_sample_rate, event_log[2])
        elif 'verbal_feedback' in file:
            feed_back_file = file

    # Get filtered cross_validation_data on the EVENT LOG
    gsr_data = add_verbal_feedback_with_gsr(feed_back_file, gsr_data, thirty_min_gsr)
    filtered_hr_data = trunc_data_on_event(hr_data, event_log)
    filtered_hrv_data = trunc_data_on_event(hrv_data, event_log)

    # Add cross_validation_data
    aggregated_data = add_hrv_data(filtered_hr_data, filtered_hrv_data)
    # print("HR+HRV cross_validation_data count: ", len(aggregated_data))
    # aggregated_data.set_index('Date', inplace=True)  # Add datetime as index
    data = aggregated_data.merge(gsr_data, left_index=True, right_index=True, how='inner')
    data['Time'] = data.index.astype(str)

    # process_mouse_click(cross_validation_data, mouse_click)
    calculate_min_max(data, window=3)
    calculate_rolling_avg(data, window=3)
    calculate_percentage_of_change(data)

    # # Evenly  Distribution of all different types of SEVERITY
    # resting_data = even_distribution(data, percentage=0.3, epoch='0')  # Take only 30% of zero feedback
    # vr_resting_data = even_distribution(data, percentage=0.5, epoch='1')  # Take only 30% of zero feedback

    data = data.loc[(data['Epoch'] != "0")]  # Ignoring Resting baseline data
    data = data.loc[(data['Epoch'] != "1")]
    #
    # data = pd.concat([pd.DataFrame(resting_data), data])
    # data = pd.concat([pd.DataFrame(vr_resting_data), data])
    # data.sort_index(inplace=True)  # Re-organize the cross_validation_data
    data['Individual'] = indi_count
    print("Individual cross_validation_data collected: ", len(data))

    return data


def calculate_rolling_avg(data, window=3):
    data['HR'] = data['HR'].rolling(window=window).mean()
    data['HRV'] = data['HRV'].rolling(window=window).mean()
    data['BR'] = data['BR'].rolling(window=window).mean()
    data['GSR'] = data['GSR'].rolling(window=window).mean()


def calculate_min_max(data, window=3):
    # Calculate min
    data['HR_MIN'] = data['HR'].rolling(window=window).min()
    data['HRV_MIN'] = data['HRV'].rolling(window=window).min()
    data['BR_MIN'] = data['BR'].rolling(window=window).min()
    data['GSR_MIN'] = data['GSR'].rolling(window=window).min()

    # Calculate Max
    data['HR_MAX'] = data['HR'].rolling(window=window).max()
    data['HRV_MAX'] = data['HRV'].rolling(window=window).max()
    data['BR_MAX'] = data['BR'].rolling(window=window).max()
    data['GSR_MAX'] = data['GSR'].rolling(window=window).max()


def calculate_percentage_of_change(data):
    # Resting cross_validation_data
    resting_data = data.loc[data['Epoch'] == '0']
    # Calculate difference from Resting cross_validation_data
    data['PC_HR'] = (data['HR'] - resting_data['HR'].mean()) / resting_data['HR'].mean()
    data['PC_HRV'] = (data['HRV'] - resting_data['HRV'].mean()) / resting_data[
        'HRV'].mean()
    data['PC_BR'] = (data['BR'] - resting_data['BR'].mean()) / resting_data['BR'].mean()
    data['PC_GSR'] = (data['GSR'] - resting_data['GSR'].mean()) / resting_data[
        'GSR'].mean()

    # calculate percentage of change for each cross_validation_data
    data['PC_HR'] = data['PC_HR'] * 100.0
    data['PC_HRV'] = data['PC_HRV'] * 100.0
    data['PC_BR'] = data['PC_BR'] * 100.0
    data['PC_GSR'] = data['PC_GSR'] * 100.0

    return data


def process_mouse_click(data, mouse_click):
    individual_data = pd.DataFrame()

    for index, row in mouse_click.iterrows():
        current_data = data[data['Time'].str.contains(row['Time'])]
        if not current_data.empty:
            previous_nth_data = data.loc[current_data.index.values[0] - 5:current_data.index.values[0]]

            if row['Click'] == '0':  # Left click means sickness cross_validation_data
                delta_hr, delta_br, delta_hrv, delta_gsr = process_left_click(current_data, previous_nth_data)
            else:
                delta_hr, delta_br, delta_hrv, delta_gsr = process_right_click(current_data, previous_nth_data)

            delta_feedback = previous_nth_data['Feedback'] * (
                    delta_hr.values - delta_hrv.values - delta_br.values + delta_gsr.values)
            previous_nth_data.loc[:, 'Feedback'] = previous_nth_data.loc[:, 'Feedback'] + delta_feedback.values
            previous_nth_data = previous_nth_data.dropna()
            previous_nth_data.loc[previous_nth_data['Feedback'] > 10, 'Feedback'] = 10
            individual_data = individual_data.append(previous_nth_data, ignore_index=True)

    individual_data.to_csv('indiv_mouse_click_only.csv', mode='a', header=False)

    return data


def calculate_corr(data, x, y):
    r, p = stats.pearsonr(data[x], data[y])
    print("Pearson Correlation of ", x, "vs", y)
    print("R value: ", r, "p value: ", p)


def remove_outliers_z_score(data, features):
    tmp_data = data[features]
    z = np.abs(stats.zscore(tmp_data))
    threshold = 3  # z score value for which we will consider it an outlier
    outliers = np.where(z > 3)  # Outliers where z value is greater than 3 std
    dataset_without_outliers = data[(z < threshold).all(axis=1)]

    return dataset_without_outliers


def process_right_click(current_data, previous_data):
    worst_hr = previous_data.loc[previous_data['HR'].idxmax()]  # Worst HR is max
    worst_br = previous_data.loc[previous_data['BR'].idxmin()]
    worst_hrv = previous_data.loc[previous_data['HRV'].idxmin()]
    worst_gsr = previous_data.loc[previous_data['GSR'].idxmax()]

    # How much each signal is deviated from Worst
    delta_hr = (previous_data['HR'] - worst_hr['HR']) / worst_hr['HR']
    delta_br = (previous_data['BR'] - worst_br['BR']) / worst_br['BR']
    delta_hrv = (previous_data['HRV'] - worst_hrv['HRV']) / worst_hrv['HRV']
    delta_gsr = (previous_data['GSR'] - worst_gsr['GSR']) / worst_gsr['GSR']

    return delta_hr, delta_br, delta_hrv, delta_gsr


def process_left_click(current_data, previous_data):
    stable_hr = previous_data.loc[previous_data['HR'].idxmin()]  # Stable HR is min
    stable_br = previous_data.loc[previous_data['BR'].idxmax()]
    stable_hrv = previous_data.loc[previous_data['HRV'].idxmax()]
    stable_gsr = previous_data.loc[previous_data['GSR'].idxmin()]

    # how much each signal is deviated from stable
    delta_hr = (previous_data['HR'] - stable_hr['HR']) / stable_hr['HR']
    delta_br = (previous_data['BR'] - stable_br['BR']) / stable_br['BR']
    delta_hrv = (previous_data['HRV'] - stable_hrv['HRV']) / stable_hrv['HRV']
    delta_gsr = (previous_data['GSR'] - stable_gsr['GSR']) / stable_gsr['GSR']

    return delta_hr, delta_br, delta_hrv, delta_gsr


def process_data_from_file(files, folders, shuffle=False):
    indi_count = 0
    data = pd.DataFrame()
    if shuffle:
        print("NOT Implememted yet")
    else:
        for f in folders:
            individual_file_list = [file for file in files if file.startswith(f)]
            file_name = f.replace('cross_validation_data\\', '')
            print("Processing Individual: ", indi_count, file_name)
            ind_data = process_individual_data(individual_file_list, indi_count)
            data = data.append(ind_data)
            indi_count = indi_count + 1
    return data


def even_distribution(data, percentage, epoch):
    resting_epoch = data.loc[(data['Epoch'] == epoch)]
    print("Resting Len: ", len(resting_epoch))
    resting_epoch = resting_epoch.sample(frac=percentage, replace=False, random_state=1)
    resting_epoch.sort_index(inplace=True)

    return resting_epoch


def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    return scaled


def standardize_data(data):
    scaler = StandardScaler()
    standardized = scaler.fit_transform(data)
    return pd.DataFrame(standardized)


def plot_model(data):
    pyplot.subplot(211)
    pyplot.hist(data['Feedback'], label='Feedback')
    pyplot.legend()
    pyplot.show()


def correlation_matrix(df, features):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Feature Correlation')
    labels = features
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])
    plt.show()


############# RUN THE FILE ###################
features = ['HR',
            'PC_HR',
            'HR_MIN',
            'HR_MAX',
            'HRV',
            'PC_HRV',
            'HRV_MIN',
            'HRV_MAX',
            # 'BR',
            'PC_BR',
            # 'BR_MIN',
            # 'BR_MAX',
            'GSR',
            'PC_GSR',
            'GSR_MIN',
            'GSR_MAX',
            ]

d = datetime.today()
time_date = d.strftime("%d_%B_%Y_%H_%M_%S")
file_name = 'dnn_data/agg_data/raw_data_' + time_date + '.csv'
data_dir = 'raw_data'
collum_name = ['Epoch'] + features + ['Feedback'] + ['Individual']

file_list = [os.path.join(r, n) for r, _, f in os.walk(data_dir) for n in f]
sub_folders = glob.glob("raw_data\\*")
dataset = process_data_from_file(file_list, sub_folders, shuffle=False)
dataset = dataset.dropna()
dataset = dataset[collum_name]
# Outliers removal using z-score analysis.
dataset = remove_outliers_z_score(dataset, ['Feedback'] + features)

# Saving all raw cross_validation_data
# dataset = dataset[['Feedback'] + features]
dataset = dataset[collum_name]  # Save all column
dataset.to_csv(file_name)

#
# # Cross Validation Save
# test_list1 = [2, 21, 15, 20, 7]
# test_list2 = [5, 22, 13, 14, 16]
# test_list3 = [0, 12, 18, 11, 8]
# test_list4 = [10, 19, 7, 3, 6]
# test_list5 = [9, 1, 17, 2, 11]
#
test_list1 = dataset.loc[
    (dataset['Individual'] == 9) | (dataset['Individual'] == 1) | (dataset['Individual'] == 17) | (
                dataset['Individual'] == 2) | (dataset['Individual'] == 11)]
train_list1 = dataset.loc[
    (dataset['Individual'] != 9) & (dataset['Individual'] != 1) & (dataset['Individual'] != 17) & (
                dataset['Individual'] != 2) & (dataset['Individual'] != 11)]
test_list1.to_csv('dnn_data/cross_validation_data/full_test_data_set1-' + time_date + '.csv')
train_list1.to_csv('dnn_data/cross_validation_data/full_train_data_set1-' + time_date + '.csv')

