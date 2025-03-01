import os

import numpy as np

import pandas as pd
from sklearn.metrics import accuracy_score

from ml_utils.machine_learning_utils import MachineLearningUtils


def test_scaled_data(dataframe: pd.DataFrame, test_df: pd.DataFrame):
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

    transformations_df = pd.DataFrame(
        columns=["model", "action", "measure", "min_range", "max_range", "operation", "gain"])

    for action in dataframe["action"].unique():
        print("action")
        for measure in [measure for measure in dataframe.columns if measure != "action"]:
            for i in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
                for j in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
                    train_df = dataframe

                    scaling_test = dataframe[dataframe["action"] == action]
                    scaling_test[measure] = scaling_test[measure] * pd.Series(
                        np.random.uniform(1 - j, 1 + i, size=len(scaling_test)), index=scaling_test.index)
                    train_df = pd.concat([train_df, scaling_test])

                    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                                  predicted_var='action',
                                                                                  test_size=0.0000001)
                    knn = ml_utils.knn_train(x_train, x_external_test, y_train, y_external_test)
                    rf = ml_utils.random_forest_train(x_train, x_external_test, y_train, y_external_test)
                    xt = ml_utils.extra_tree_train(x_train, x_external_test, y_train, y_external_test)

                    knn_y_pred = knn.predict(x_external_test)
                    knn_accuracy = accuracy_score(y_external_test, knn_y_pred)

                    rf_y_pred = rf.predict(x_external_test)
                    rf_accuracy = accuracy_score(y_external_test, rf_y_pred)

                    xt_y_pred = xt.predict(x_external_test)
                    xt_accuracy = accuracy_score(y_external_test, xt_y_pred)

                    new_row = [{"model": "knn", "action": action, "measure": measure, "min_range": 1 - j,
                                "max_range": 1 + i,
                                "operation": "*", "gain": knn_accuracy - knn_original_accuracy}]
                    new_row_df = pd.DataFrame(new_row)
                    transformations_df = pd.concat([transformations_df, new_row_df])

                    new_row = [{"model": "random forest", "action": action, "measure": measure, "min_range": 1 - j,
                                "max_range": 1 + i,
                                "operation": "*", "gain": rf_accuracy - rf_original_accuracy}]
                    new_row_df = pd.DataFrame(new_row)
                    transformations_df = pd.concat([transformations_df, new_row_df])

                    new_row = [{"model": "xtree", "action": action, "measure": measure, "min_range": 1 - j,
                                "max_range": 1 + i,
                                "operation": "*", "gain": xt_accuracy - xt_original_accuracy}]
                    new_row_df = pd.DataFrame(new_row)
                    transformations_df = pd.concat([transformations_df, new_row_df])
    path = "data/transformations/"
    if not os.path.exists(path):
        os.makedirs(path)
    transformations_df.to_csv(path + "transformations.csv", index=False)
    best_transformations = transformations_df.groupby(["action", "measure", "min_range", "max_range", "operation"]).agg(
        {"gain": 'mean'}).reset_index()
    best_transformations.to_csv(path + "mean_transformations.csv", index=False)
    best_transformations_max = best_transformations.groupby(["action", "measure", "operation"]).agg(
        {"gain": ['idxmax', 'max']})
    best_transformations = best_transformations.loc[best_transformations_max['gain', 'idxmax']].reset_index()
    best_transformations.to_csv(path + "best_transformations.csv", index=False)


if __name__ == '__main__':
    prepared_df = pd.read_csv("data/datasets/standard/standard_dataset.csv")
    users_df = pd.read_csv("data/datasets/new_users/all_users.csv")
    test_scaled_data(prepared_df, users_df)
