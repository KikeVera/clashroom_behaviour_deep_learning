import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import Adam

from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


def generate_scaled_data(dataframe: pd.DataFrame):
    dataframe = dataframe.drop("action", axis=1)

    random_values = pd.DataFrame(np.random.uniform(0.7, 1.3, size=dataframe.shape), columns=dataframe.columns,
                                 index=dataframe.index)

    scaled_dataframe = dataframe * random_values

    return scaled_dataframe


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    prepared_df = pd.read_csv("data/datasets/standard/standard_dataset.csv")
    for action in prepared_df["action"].unique():
        action_df = prepared_df[prepared_df["action"] == action]
        path = "data/datasets/scaled_data/"
        if not os.path.exists(path):
            os.makedirs(path)
        scaled_df = generate_scaled_data(action_df)
        scaled_df["action"] = action
        scaled_df.to_csv(path + action + ".csv", index=False)
