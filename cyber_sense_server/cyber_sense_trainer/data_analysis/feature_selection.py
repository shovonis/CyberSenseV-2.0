import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def correlation_heatmap(data):
    plt.close('all')
    corrmat = data.corr()
    f, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1, annot=True)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig('correlation_heatmap2.pdf', dpi=100)
    plt.show()


data = pd.read_csv("raw_data_feature_only.csv", delimiter=',')
print(data.shape)
data = data.sample(frac=0.1, replace=True, random_state=1)
# cross_validation_data = cross_validation_data[cross_validation_data['Feedback'] > 7]
print(data.shape)
correlation_heatmap(data)
