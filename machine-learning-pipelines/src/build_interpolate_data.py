import os

import numpy as np

import pandas as pd
from sklearn.metrics import accuracy_score

from ml_utils.machine_learning_utils import MachineLearningUtils


def test_interpolated_data(dataframe: pd.DataFrame, test_df: pd.DataFrame):
    ml_utils = MachineLearningUtils()
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=dataframe,
                                                                  predicted_var='action', test_size=0.0000001)
    x_external_test, y_external_test = ml_utils.get_external_test(train_df=test_df, predicted_var="action")
    knn = ml_utils.knn_train(x_train, x_external_test, y_train, y_external_test)
    rf = ml_utils.random_forest_train(x_train, x_external_test, y_train, y_external_test)
    xt = ml_utils.extra_tree_train(x_train, x_external_test, y_train, y_external_test)

    knn_y_pred = knn.predict(x_external_test)
    knn_original_accuracy = accuracy_score(y_external_test, knn_y_pred)

    rf_y_pred = rf.predict(x_external_test)
    rf_original_accuracy = accuracy_score(y_external_test, rf_y_pred)

    xt_y_pred = xt.predict(x_external_test)
    xt_original_accuracy = accuracy_score(y_external_test, xt_y_pred)
    train_df = dataframe
    for action in dataframe["action"].unique():

        interpolated_action = dataframe[dataframe["action"] == action].drop("action", axis=1)
        max_values = interpolated_action.max()
        min_values = interpolated_action.min()
        intermediate_values = pd.DataFrame()
        for column in interpolated_action.columns:
            intermediate = np.linspace(min_values[column], max_values[column], num=1000)
            intermediate_values[column] = intermediate
        intermediate_values["action"] = action
        train_df = pd.concat([train_df, intermediate_values])

    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action',
                                                                  test_size=0.0000001)
    knn = ml_utils.knn_train(x_train, x_external_test, y_train, y_external_test)
    rf = ml_utils.random_forest_train(x_train, x_external_test, y_train, y_external_test)
    xt = ml_utils.extra_tree_train(x_train, x_external_test, y_train, y_external_test)



    # path = "data/datasets/interpolated/"
    # if not os.path.exists(path):
    #     os.makedirs(path)




if __name__ == '__main__':
    prepared_df = pd.read_csv("data/datasets/standard/standard_dataset.csv")
    users_df = pd.read_csv("data/datasets/new_users/all_users.csv")
    test_interpolated_data(prepared_df, users_df)
