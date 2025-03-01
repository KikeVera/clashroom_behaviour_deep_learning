import os

import numpy as np

import pandas as pd
from sklearn.metrics import accuracy_score

from ml_utils.machine_learning_utils import MachineLearningUtils


def test_normalized_data(dataframe: pd.DataFrame, test_df: pd.DataFrame):
    ml_utils = MachineLearningUtils()

    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=dataframe,
                                                                  predicted_var='action', test_size=0.0000001)
    x_external_test, y_external_test = ml_utils.get_external_test(train_df=test_df, predicted_var="action")
    knn = ml_utils.knn_train(x_train, x_external_test, y_train, y_external_test)
    rf = ml_utils.random_forest_train(x_train, x_external_test, y_train, y_external_test)
    xt = ml_utils.extra_tree_train(x_train, x_external_test, y_train, y_external_test)

    train_actions = dataframe["action"]
    test_actions = test_df["action"]
    train_df = dataframe.drop("action", axis=1)
    test_n_df = test_df.drop("action", axis=1)

    max_df = pd.concat([train_df, test_n_df]).max()

    train_df = train_df.div(max_df)
    test_n_df = test_n_df.div(max_df)

    train_df["action"] = train_actions
    test_n_df["action"] = test_actions

    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action', test_size=0.0000001)
    x_external_test, y_external_test = ml_utils.get_external_test(train_df=test_n_df, predicted_var="action")
    knn = ml_utils.knn_train(x_train, x_external_test, y_train, y_external_test)
    rf = ml_utils.random_forest_train(x_train, x_external_test, y_train, y_external_test)
    xt = ml_utils.extra_tree_train(x_train, x_external_test, y_train, y_external_test)
    svm = ml_utils.svm_train(x_train, x_external_test, y_train, y_external_test)
    mlp = ml_utils.neuronal_network_train(x_train, x_external_test, y_train, y_external_test)

    train_df = dataframe.drop("action", axis=1)
    test_n_df = test_df.drop("action", axis=1)

    train_df = train_df * 10
    test_n_df = test_n_df * 10

    train_df["action"] = train_actions
    test_n_df["action"] = test_actions

    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action', test_size=0.0000001)
    x_external_test, y_external_test = ml_utils.get_external_test(train_df=test_n_df, predicted_var="action")
    knn = ml_utils.knn_train(x_train, x_external_test, y_train, y_external_test)
    rf = ml_utils.random_forest_train(x_train, x_external_test, y_train, y_external_test)
    xt = ml_utils.extra_tree_train(x_train, x_external_test, y_train, y_external_test)
    svm = ml_utils.svm_train(x_train, x_external_test, y_train, y_external_test)
    mlp = ml_utils.neuronal_network_train(x_train, x_external_test, y_train, y_external_test)


    # path = "data/datasets/interpolated/"
    # if not os.path.exists(path):
    #     os.makedirs(path)


if __name__ == '__main__':
    prepared_df = pd.read_csv("data/datasets/standard/standard_dataset.csv")
    users_df = pd.read_csv("data/datasets/new_users/all_users.csv")
    test_normalized_data(prepared_df, users_df)
