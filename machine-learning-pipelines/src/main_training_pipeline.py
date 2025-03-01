# This is a sample Python script.
import os

import pandas as pd

from data_utils import ETL
from ml_utils.train_wisdm_dataframe import run_wisdm_train_stacking, run_wisdm_knn_train, run_wisdm_mlp_train, \
    run_wisdm_rf_train, run_wisdm_xtree_train, run_wisdm_decision_tree_train, run_wisdm_svm_train, \
    run_wisdm_bayes_train, run_wisdm_stochastic_gradient_train
from ml_utils.variables_importance import get_cross_variables_and, get_cross_variables_or, knn_variables, mlp_variables, \
    rf_variables, xtree_variables, variables_importance, variables_importance_single_models


def standard_train(train_df, test_df=None, sub_path=""):
    print("SINGLE MODELS ALL VARIABLES")

    path = sub_path + "data/models/wisdm_single/all_variables/single_models/"
    if not os.path.exists(path):
        os.makedirs(path)
    run_wisdm_decision_tree_train(wisdm_train_df=train_df, path=path, external_test=test_df)
    run_wisdm_svm_train(wisdm_train_df=train_df, path=path, external_test=test_df)
    run_wisdm_bayes_train(wisdm_train_df=train_df, path=path, external_test=test_df)
    run_wisdm_stochastic_gradient_train(wisdm_train_df=train_df, path=path, external_test=test_df)

    run_wisdm_knn_train(wisdm_train_df=train_df, path=path, external_test=test_df)
    run_wisdm_mlp_train(wisdm_train_df=train_df, path=path, external_test=test_df)
    run_wisdm_rf_train(wisdm_train_df=train_df, path=path, external_test=test_df)
    run_wisdm_xtree_train(wisdm_train_df=train_df, path=path, external_test=test_df)

    print("STACK MODELS ALL VARIABLES")
    path = sub_path + "data/models/wisdm_stack/all_variables/"
    if not os.path.exists(path):
        os.makedirs(path)
    run_wisdm_train_stacking(wisdm_train_df=train_df, path=path,
                             external_test=test_df)

    print("VARIABLES IMPORTANCE")

    variables_importance(
        train_df=train_df,
        test_df=test_df,
        sub_path=sub_path
    )


def get_outdated_dataset() -> pd.DataFrame:
    old_df = pd.read_csv("data/datasets/outdated/old_df.csv")
    return old_df


def get_wisdm_processed_dataset() -> pd.DataFrame:
    prepared_df = pd.read_csv("data/datasets/standard/standard_dataset.csv")
    # etl = ETL()
    # prepared_df = etl.process_wisdm()
    return prepared_df


def get_users_collected_data() -> pd.DataFrame:
    users_df = pd.read_csv("data/datasets/new_users/all_users.csv")
    return users_df


def get_synthetic_data() -> pd.DataFrame:
    synthetic_df = pd.DataFrame(columns=prepared_df.columns)
    for file in os.listdir("data/datasets/synthetic_data/"):
        synthetic_action = pd.read_csv("data/datasets/synthetic_data/" + file)
        synthetic_df = pd.concat([synthetic_df, synthetic_action])
    return synthetic_df


def get_scaled_data() -> pd.DataFrame:
    scaled_df = pd.DataFrame(columns=prepared_df.columns)
    for file in os.listdir("data/datasets/scaled_data/"):
        scaled_action = pd.read_csv("data/datasets/scaled_data/" + file)
        scaled_df = pd.concat([scaled_df, scaled_action])
    return scaled_df


if __name__ == '__main__':
    outdated_df = get_outdated_dataset()
    wisdm_prepared_df = get_wisdm_processed_dataset()
    users_collected_df = get_users_collected_data()
    synthetic_data_df = get_synthetic_data()
    scaled_data_df = get_scaled_data()

    print("RUN WISDM OLD MODELS")
    standard_train(outdated_df, sub_path="old_wisdm")
    print("WISDM DATAFRAME")
    standard_train(wisdm_prepared_df, sub_path="wisdm/")
    print("COMBINED DATAFRAME")
    standard_train(pd.concat([wisdm_prepared_df, users_collected_df]), sub_path="combined/")
    print("WISDM AGAINST NEW USERS")
    standard_train(wisdm_prepared_df, users_collected_df, sub_path="against/")
    print("WISDM SYNTHETIC COMBINED")
    standard_train(synthetic_data_df, sub_path="synthetic_combined/")
    print("WISDM ONLY SYNTHETIC AGAINST NEW USERS")
    standard_train(synthetic_data_df, synthetic_data_df, sub_path="only_synthetic_against/")
    print("WISDM SYNTHETIC AGAINST NEW USERS")
    standard_train(pd.concat([wisdm_prepared_df, synthetic_data_df]), users_collected_df, sub_path="synthetic_against/")
    print("WISDM ONLY SCALED AGAINST NEW USERS")
    standard_train(scaled_data_df, users_collected_df, sub_path="only_scaled_against/")
    print("WISDM SCALED AGAINST NEW USERS")
    standard_train(pd.concat([wisdm_prepared_df, scaled_data_df]), users_collected_df, sub_path="scaled_against/")
