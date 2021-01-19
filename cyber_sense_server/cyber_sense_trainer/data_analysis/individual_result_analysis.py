from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

actual = [1.22,
          5.41,
          2.00,
          2.00,
          2.00,
          2.00,
          2.00,
          2.00,
          2.00,
          2.00,
          2.40,
          3.00,
          3.00,
          3.00,
          3.00,
          3.00]

predicted = [8.50,
             5.69,
             7.81,
             2.10,
             .14,
             4.72,
             4.93,
             5.25,
             5.19,
             7.94,
             5.58,
             4.04,
             6.06,
             3.24,
             4.01,
             5.87]

# calculate RMSE
rsqr = r2_score(actual, predicted)
mae = mean_absolute_error(actual, predicted)
mdae = median_absolute_error(actual, predicted)
rmse = sqrt(mean_squared_error(actual, predicted))

print('Test RMSE: %.3f' % rmse)
print("TEST R square: %.3f" % rsqr)
print("TEST MAE: %.3f" % mae)
print("TEST Median AE: %.3f" % mdae)


def plot_predictions(actual, predicted):
    plt.figure()
    plt.plot(actual, 'b-o', label='Actual')
    plt.plot(predicted, 'r-o', label='Predicted')
    plt.title('Individual 6 Actual vs Predicted SR_t score')
    plt.xlabel('Number of Observations')
    plt.ylabel('SR_t score')
    plt.legend()
    plt.savefig("individual_6_fig.png")
    plt.show()


plot_predictions(actual, predicted)
