import copy
import os

from sqlalchemy import create_engine
from matplotlib import pyplot as plt
import pandas as pd
from machine_learning_library.data_utils import DataUtils
import pickle
from os import listdir

from machine_learning_library.machine_learning_utils import MachineLearningUtils


class MLModule:

    def __init__(self):
        self.wrong = 0
        self.machine_learning_utils = MachineLearningUtils()
        self.data_utils = DataUtils()



    def run_wisdm(self):
        self.data_utils.process_wisdm()



    def generate_all_stacked_models(self):
        wisdm_train_df = pd.read_csv("data/processed_csv/actions/prepared_df_less_eat.csv")
        all_users_df = pd.read_csv("data/processed_csv/new_users/all_users.csv")
        x_train, x_test, y_train, y_test = self.machine_learning_utils.get_training_data(train_df=wisdm_train_df,
                                                                                         predicted_var='action')
        x_train2, x_test2, y_train2, y_test2 = self.machine_learning_utils.get_training_data(train_df=all_users_df,
                                                                                             predicted_var='action')
        x_train3 = pd.concat([x_train, x_train2], ignore_index=True)
        x_test3 = pd.concat([x_test, x_test2], ignore_index=True)
        y_train3 = pd.concat([y_train, y_train2], ignore_index=True)
        y_test3 = pd.concat([y_test, y_test2], ignore_index=True)

        stack = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3)
        pickle.dump(stack, open("./data/models/new_models/stacking_model.pkl", "wb"))

        mlp_knn_rf = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, xtree=False)
        pickle.dump(mlp_knn_rf, open("./data/models/new_models/mlp_knn_rf_model.pkl", "wb"))

        mlp_knn_xtree = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, rf=False)
        pickle.dump(mlp_knn_xtree, open("./data/models/new_models/mlp_knn_xtree_model.pkl", "wb"))

        xtree_knn_rf = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, mlp=False)
        pickle.dump(xtree_knn_rf, open("./data/models/new_models/xtree_knn_rf_model.pkl", "wb"))

        mlp_xtree_rf = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, knn=False)
        pickle.dump(mlp_xtree_rf, open("./data/models/new_models/mlp_xtree_rf_model.pkl", "wb"))

        mlp_knn = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, rf=False,
                                                             xtree=False)
        pickle.dump(mlp_knn, open("./data/models/new_models/mlp_knn_model.pkl", "wb"))

        rf_knn = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, mlp=False,
                                                            xtree=False)
        pickle.dump(rf_knn, open("./data/models/new_models/rf_knn_model.pkl", "wb"))

        rf_mlp = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, knn=False,
                                                            xtree=False)
        pickle.dump(rf_mlp, open("./data/models/new_models/rf_mlp_model.pkl", "wb"))

        xtree_mlp = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, knn=False,
                                                               rf=False)
        pickle.dump(xtree_mlp, open("./data/models/new_models/xtree_mlp_model.pkl", "wb"))

        xtree_knn = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, rf=False,
                                                               mlp=False)
        pickle.dump(xtree_knn, open("./data/models/new_models/xtree_knn_model.pkl", "wb"))

        xtree_rf = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, mlp=False,
                                                              knn=False)
        pickle.dump(xtree_rf, open("./data/models/new_models/xtree_rf_model.pkl", "wb"))

        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=stack,
                                                          model_name="mlp_knn_rf_xtree")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=mlp_knn_rf,
                                                          model_name="mlp_knn_rf")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=mlp_xtree_rf,
                                                          model_name="mlp_xtree_rf")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=mlp_knn_xtree,
                                                          model_name="mlp_knn_xtree")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=xtree_knn_rf,
                                                          model_name="xtree_knn_rf")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=mlp_knn,
                                                          model_name="mlp_knn")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=rf_mlp,
                                                          model_name="rf_mlp")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=rf_knn,
                                                          model_name="rf_knn")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=xtree_rf,
                                                          model_name="xtree_rf")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=xtree_mlp,
                                                          model_name="xtree_mlp")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=xtree_knn,
                                                          model_name="xtree_knn")
        plt.show()

    def generate_all_stacked_models_new_actions(self):
        wisdm_train_df = pd.read_csv("data/processed_csv/actions/prepared_df_less_eat.csv")
        all_users_df = pd.concat(
            [pd.read_csv("data/processed_csv/new_users/all_users_new_actions.csv"),
             pd.read_csv("data/processed_csv/new_users/all_users.csv")])
        x_train, x_test, y_train, y_test = self.machine_learning_utils.get_training_data(train_df=wisdm_train_df,
                                                                                         predicted_var='action')
        x_train2, x_test2, y_train2, y_test2 = self.machine_learning_utils.get_training_data(train_df=all_users_df,
                                                                                             predicted_var='action')
        x_train3 = pd.concat([x_train, x_train2], ignore_index=True)
        x_test3 = pd.concat([x_test, x_test2], ignore_index=True)
        y_train3 = pd.concat([y_train, y_train2], ignore_index=True)
        y_test3 = pd.concat([y_test, y_test2], ignore_index=True)

        stack = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3)
        pickle.dump(stack, open("./data/models/new_models/stacking_model_new_actions.pkl", "wb"))

        mlp_knn_rf = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, xtree=False)
        pickle.dump(mlp_knn_rf, open("./data/models/new_models/mlp_knn_rf_model_new_actions.pkl", "wb"))

        mlp_knn_xtree = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, rf=False)
        pickle.dump(mlp_knn_xtree, open("./data/models/new_models/mlp_knn_xtree_model_new_actions.pkl", "wb"))

        xtree_knn_rf = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, mlp=False)
        pickle.dump(xtree_knn_rf, open("./data/models/new_models/xtree_knn_rf_model_new_actions.pkl", "wb"))

        mlp_xtree_rf = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, knn=False)
        pickle.dump(mlp_xtree_rf, open("./data/models/new_models/mlp_xtree_rf_model_new_actions.pkl", "wb"))

        mlp_knn = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, rf=False,
                                                             xtree=False)
        pickle.dump(mlp_knn, open("./data/models/new_models/mlp_knn_model_new_actions.pkl", "wb"))

        rf_knn = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, mlp=False,
                                                            xtree=False)
        pickle.dump(rf_knn, open("./data/models/new_models/rf_knn_model_new_actions.pkl", "wb"))

        rf_mlp = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, knn=False,
                                                            xtree=False)
        pickle.dump(rf_mlp, open("./data/models/new_models/rf_mlp_model_new_actions.pkl", "wb"))

        xtree_mlp = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, knn=False,
                                                               rf=False)
        pickle.dump(xtree_mlp, open("./data/models/new_models/xtree_mlp_model_new_actions.pkl", "wb"))

        xtree_knn = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, rf=False,
                                                               mlp=False)
        pickle.dump(xtree_knn, open("./data/models/new_models/xtree_knn_model_new_actions.pkl", "wb"))

        xtree_rf = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, mlp=False,
                                                              knn=False)
        pickle.dump(xtree_rf, open("./data/models/new_models/xtree_rf_model_new_actions.pkl", "wb"))

        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=stack,
                                                          model_name="mlp_knn_rf_xtree")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=mlp_knn_rf,
                                                          model_name="mlp_knn_rf")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=mlp_xtree_rf,
                                                          model_name="mlp_xtree_rf")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=mlp_knn_xtree,
                                                          model_name="mlp_knn_xtree")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=xtree_knn_rf,
                                                          model_name="xtree_knn_rf")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=mlp_knn,
                                                          model_name="mlp_knn")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=rf_mlp,
                                                          model_name="rf_mlp")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=rf_knn,
                                                          model_name="rf_knn")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=xtree_rf,
                                                          model_name="xtree_rf")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=xtree_mlp,
                                                          model_name="xtree_mlp")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=xtree_knn,
                                                          model_name="xtree_knn")


    def generate_all_simple_models(self, testing_mode="full"):
        print("\nFULL DATA TRAIN: \n")
        wisdm_train_df = pd.read_csv("data/processed_csv/actions/prepared_df_less_eat.csv")
        all_users_df = pd.read_csv("data/processed_csv/new_users/all_users.csv")
        x_train, x_test, y_train, y_test = self.machine_learning_utils.get_training_data(train_df=wisdm_train_df,
                                                                                         predicted_var='action')
        x_train2, x_test2, y_train2, y_test2 = self.machine_learning_utils.get_training_data(train_df=all_users_df,
                                                                                             predicted_var='action')

        x_train3 = pd.concat([x_train, x_train2], ignore_index=True)
        x_test3 = pd.concat([x_test, x_test2], ignore_index=True)
        y_train3 = pd.concat([y_train, y_train2], ignore_index=True)
        y_test3 = pd.concat([y_test, y_test2], ignore_index=True)
        if testing_mode == "original":
            y_test3 = y_test
            x_test3 = x_test

        if testing_mode == "new":
            y_test3 = y_test2
            x_test3 = x_test2

        xtree = self.machine_learning_utils.extra_tree_train(x_train3, x_test3, y_train3, y_test3)
        pickle.dump(xtree, open("./data/models/new_models/xtree_model.pkl", "wb"))

        rf = self.machine_learning_utils.random_forest_train(x_train3, x_test3, y_train3, y_test3)
        pickle.dump(rf, open("./data/models/new_models/random_forest_model.pkl", "wb"))

        knn = self.machine_learning_utils.knn_train(x_train3, x_test3, y_train3, y_test3)
        pickle.dump(knn, open("./data/models/new_models/knn_model.pkl", "wb"))
        #
        clf = self.machine_learning_utils.neuronal_network_train(x_train3, x_test3, y_train3, y_test3)
        pickle.dump(clf, open("./data/models/new_models/neural_network_model.pkl", "wb"))

        print("\nOWN DATA TRAIN: \n")
        xtree = self.machine_learning_utils.extra_tree_train(x_train2, x_test3, y_train2, y_test3)
        pickle.dump(xtree, open("./data/models/new_models/xtree_model_own_data.pkl", "wb"))
        #
        rf = self.machine_learning_utils.random_forest_train(x_train2, x_test3, y_train2, y_test3)
        pickle.dump(rf, open("./data/models/new_models/random_forest_model_own_data.pkl", "wb"))
        #
        knn = self.machine_learning_utils.knn_train(x_train2, x_test3, y_train2, y_test3)
        pickle.dump(knn, open("./data/models/new_models/knn_model_own_data.pkl", "wb"))
        #
        clf = self.machine_learning_utils.neuronal_network_train(x_train2, x_test3, y_train2, y_test3)
        pickle.dump(clf, open("./data/models/new_models/neural_network_model_own_data.pkl", "wb"))

        print("\nEXTERNAL DATA TRAIN: \n")
        xtree = self.machine_learning_utils.extra_tree_train(x_train, x_test3, y_train, y_test3)
        pickle.dump(xtree, open("./data/models/new_models/xtree_model_external_data.pkl", "wb"))
        #
        rf = self.machine_learning_utils.random_forest_train(x_train, x_test3, y_train, y_test3)
        pickle.dump(rf, open("./data/models/new_models/random_forest_model_external_data.pkl", "wb"))
        #
        knn = self.machine_learning_utils.knn_train(x_train, x_test3, y_train, y_test3)
        pickle.dump(knn, open("./data/models/new_models/knn_model_external_data.pkl", "wb"))
        #
        clf = self.machine_learning_utils.neuronal_network_train(x_train, x_test3, y_train, y_test3)
        pickle.dump(clf, open("./data/models/new_models/neural_network_model_external_data.pkl", "wb"))

    def test_new_users_with_old_data(self):
        print("\nFULL DATA TRAIN: \n")
        wisdm_train_df = pd.read_csv("data/processed_csv/actions/prepared_df_less_eat.csv")
        all_users_df = pd.read_csv("data/processed_csv/new_users/all_users.csv")

        x_train, x_test, y_train, y_test = self.machine_learning_utils.get_training_data(train_df=wisdm_train_df,
                                                                                         predicted_var='action')
        x_train2, x_test2, y_train2, y_test2 = self.machine_learning_utils.get_training_data(train_df=all_users_df,
                                                                                             predicted_var='action')

        #
        xtree = self.machine_learning_utils.extra_tree_train(x_train, x_test2, y_train, y_test2)
        rf = self.machine_learning_utils.random_forest_train(x_train, x_test2, y_train, y_test2)
        knn = self.machine_learning_utils.knn_train(x_train, x_test2, y_train, y_test2)
        clf = self.machine_learning_utils.neuronal_network_train(x_train, x_test2, y_train, y_test2)

        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=knn, model_name="KNN")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=clf,
                                                          model_name="Neural Network")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=rf,
                                                          model_name="Random Forest")
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=xtree, model_name="Xtree")

        plt.show()

    def test_old_data_with_new_users(self):
        print("\nFULL DATA TRAIN: \n")
        wisdm_train_df = pd.read_csv("data/processed_csv/actions/prepared_df_less_eat.csv")
        all_users_df = pd.read_csv("data/processed_csv/new_users/all_users.csv")
        x_train, x_test, y_train, y_test = self.machine_learning_utils.get_training_data(train_df=wisdm_train_df,
                                                                                         predicted_var='action')
        x_train2, x_test2, y_train2, y_test2 = self.machine_learning_utils.get_training_data(train_df=all_users_df,
                                                                                             predicted_var='action')

        self.machine_learning_utils.extra_tree_train(x_train2, x_test, y_train2, y_test)
        self.machine_learning_utils.random_forest_train(x_train2, x_test, y_train2, y_test)
        self.machine_learning_utils.knn_train(x_train2, x_test, y_train2, y_test)
        self.machine_learning_utils.neuronal_network_train(x_train2, x_test, y_train2, y_test)

    def search_best_parameters_wisdm_knn(self):
        wisdm_train_df = pd.read_csv("data/processed_csv/actions/prepared_df_10s.csv")
        x_train, x_test, y_train, y_test = self.machine_learning_utils.get_training_data(train_df=wisdm_train_df,
                                                                                         predicted_var='action')
        self.machine_learning_utils.search_best_knn(x_train, y_train)

    def search_best_parameters_wisdm_neural_network(self):
        wisdm_train_df = pd.read_csv("data/processed_csv/actions/prepared_df_10s.csv")
        x_train, x_test, y_train, y_test = self.machine_learning_utils.get_training_data(train_df=wisdm_train_df,
                                                                                         predicted_var='action')
        self.machine_learning_utils.search_best_neuronal_network(x_train, y_train)

    def search_best_parameters_wisdm_random_forest(self):
        wisdm_train_df = pd.read_csv("data/processed_csv/actions/prepared_df_10s.csv")
        x_train, x_test, y_train, y_test = self.machine_learning_utils.get_training_data(train_df=wisdm_train_df,
                                                                                         predicted_var='action')
        self.machine_learning_utils.search_best_random_forest(x_train, y_train)

    def check_all_df_with_best_model(self):
        wisdm_train_df = pd.read_csv("data/processed_csv/actions/prepared_df_less_eat.csv")
        all_users_df = pd.read_csv("data/processed_csv/new_users/all_users.csv")

        new_actions_df = pd.concat(
            [all_users_df, pd.read_csv("data/processed_csv/new_users/all_users_new_actions.csv")],
            ignore_index=True)

        x_train, x_test, y_train, y_test = self.machine_learning_utils.get_training_data(train_df=wisdm_train_df,
                                                                                         predicted_var='action')
        x_train2, x_test2, y_train2, y_test2 = self.machine_learning_utils.get_training_data(train_df=all_users_df,
                                                                                             predicted_var='action')

        x_train3 = pd.concat([x_train, x_train2], ignore_index=True)
        x_test3 = pd.concat([x_test, x_test2], ignore_index=True)
        y_train3 = pd.concat([y_train, y_train2], ignore_index=True)
        y_test3 = pd.concat([y_test, y_test2], ignore_index=True)

        stck = self.machine_learning_utils.stacking_train(x_train3, x_test, y_train3, y_test, xtree=False)
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=stck,
                                                          model_name="Old data")

        stck = self.machine_learning_utils.stacking_train(x_train3, x_test2, y_train3, y_test2, xtree=False)
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test2, y_test=y_test2, model=stck,
                                                          model_name="New data")

        stck = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, xtree=False)
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=stck,
                                                          model_name="Full data")

        x_train2, x_test2, y_train2, y_test2 = self.machine_learning_utils.get_training_data(train_df=new_actions_df,
                                                                                             predicted_var='action')

        x_train3 = pd.concat([x_train, x_train2], ignore_index=True)
        x_test3 = pd.concat([x_test, x_test2], ignore_index=True)
        y_train3 = pd.concat([y_train, y_train2], ignore_index=True)
        y_test3 = pd.concat([y_test, y_test2], ignore_index=True)

        stck = self.machine_learning_utils.stacking_train(x_train3, x_test2, y_train3, y_test2, xtree=False)
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test2, y_test=y_test2, model=stck,
                                                          model_name="New actions")

        stck = self.machine_learning_utils.stacking_train(x_train3, x_test3, y_train3, y_test3, xtree=False)
        self.machine_learning_utils.plot_confusion_matrix(x_test=x_test3, y_test=y_test3, model=stck,
                                                          model_name="Full data new actions")

        plt.show()

    def check_all_users(self):
        real_data_df_list = []
        predicted_data_df_list = []
        for i in range(1, 25):
            if i != 10:
                user = "user" + str(i)
                for folder in listdir("data/users/" + user + "/old"):
                    for file in listdir("data/users/" + user + "/old/" + folder):
                        predicted_df = pd.read_csv("data/users/" + user + "/old/" + folder + "/" + file).dropna()
                        real_df = copy.deepcopy(predicted_df)
                        real_df["action"] = folder
                        real_data_df_list.append(real_df)
                        predicted_data_df_list.append(predicted_df)

        columns = ["mean_ax", "std_ax", "max_ax", "min_ax", "var_ax", "median_ax", "mean_peak_values_ax",
                   "std_peak_values_ax", "max_peak_values_ax", "min_peak_values_ax", "var_peak_values_ax",
                   "mean_distance_peak_values_ax", "p25_ax", "p50_ax", "p75_ax", "mean_ay", "std_ay", "max_ay",
                   "min_ay", "var_ay", "median_ay", "mean_peak_values_ay", "std_peak_values_ay", "max_peak_values_ay",
                   "min_peak_values_ay", "var_peak_values_ay", "mean_distance_peak_values_ay", "p25_ay", "p50_ay",
                   "p75_ay",
                   "mean_az", "std_az", "max_az", "min_az", "var_az", "median_az", "mean_peak_values_az",
                   "std_peak_values_az", "max_peak_values_az", "min_peak_values_az", "var_peak_values_az",
                   "mean_distance_peak_values_az", "p25_az", "p50_az", "p75_az", "mean_gx", "std_gx", "max_gx",
                   "min_gx",
                   "var_gx", "median_gx", "mean_peak_values_gx", "std_peak_values_gx", "max_peak_values_gx",
                   "min_peak_values_gx",
                   "var_peak_values_gx", "mean_distance_peak_values_gx", "p25_gx", "p50_gx", "p75_gx", "mean_gy",
                   "std_gy",
                   "max_gy", "min_gy",
                   "var_gy", "median_gy", "mean_peak_values_gy", "std_peak_values_gy", "max_peak_values_gy",
                   "min_peak_values_gy",
                   "var_peak_values_gy", "mean_distance_peak_values_gy", "p25_gy", "p50_gy", "p75_gy", "mean_gz",
                   "std_gz",
                   "max_gz", "min_gz",
                   "var_gz", "median_gz", "mean_peak_values_gz", "std_peak_values_gz", "max_peak_values_gz",
                   "min_peak_values_gz",
                   "var_peak_values_gz", "mean_distance_peak_values_gz", "p25_gz", "p50_gz", "p75_gz", "action"]
        combined_real_df = pd.DataFrame(columns=columns)
        combined_predicted_df = pd.DataFrame(columns=columns)


        for df in real_data_df_list:
            combined_real_df = pd.concat([combined_real_df, df], ignore_index=True)

        for df in predicted_data_df_list:
            combined_predicted_df = pd.concat([combined_predicted_df, df], ignore_index=True)

        combined_predicted_df = combined_predicted_df.drop([
            "mean_distance_peak_values_ax", "mean_distance_peak_values_ay", "mean_distance_peak_values_az",
            "mean_distance_peak_values_gx", "mean_distance_peak_values_gy", "mean_distance_peak_values_gz"], axis=1)
        combined_real_df = combined_real_df.drop([
            "mean_distance_peak_values_ax", "mean_distance_peak_values_ay", "mean_distance_peak_values_az",
            "mean_distance_peak_values_gx", "mean_distance_peak_values_gy", "mean_distance_peak_values_gz"], axis=1)

        combined_predicted_df=combined_predicted_df[['action','mean_ax', 'std_ax', 'max_ax', 'min_ax', 'var_ax', 'median_ax', 'mean_peak_values_ax', 'max_peak_values_ax',
         'min_peak_values_ax', 'p25_ax', 'p50_ax', 'p75_ax', 'std_ay', 'var_ay', 'p75_ay', 'mean_az', 'std_az',
         'max_az', 'min_az', 'var_az', 'median_az', 'mean_peak_values_az', 'max_peak_values_az', 'min_peak_values_az',
         'p25_az', 'p50_az', 'p75_az', 'var_gx', 'p25_gx', 'p75_gx', 'std_gy', 'var_gy', 'p25_gy', 'p75_gy', 'std_gz',
         'var_gz', 'p25_gz', 'p75_gz']]

        combined_real_df = combined_real_df[
            ['action', 'mean_ax', 'std_ax', 'max_ax', 'min_ax', 'var_ax', 'median_ax', 'mean_peak_values_ax',
             'max_peak_values_ax',
             'min_peak_values_ax', 'p25_ax', 'p50_ax', 'p75_ax', 'std_ay', 'var_ay', 'p75_ay', 'mean_az', 'std_az',
             'max_az', 'min_az', 'var_az', 'median_az', 'mean_peak_values_az', 'max_peak_values_az',
             'min_peak_values_az',
             'p25_az', 'p50_az', 'p75_az', 'var_gx', 'p25_gx', 'p75_gx', 'std_gy', 'var_gy', 'p25_gy', 'p75_gy',
             'std_gz',
             'var_gz', 'p25_gz', 'p75_gz']]

        self.machine_learning_utils.check_accuracy_of_real_users(
            model=pickle.load(open("./data/models/old_models/stacking_model.pkl", 'rb')),
            real_df=combined_real_df, predicted_df=combined_predicted_df, user="all users")

    @staticmethod
    def generate_combinated_file():
        real_data_df_list = []
        predicted_data_df_list = []
        for i in range(0, 25):
            if i != 10:
                user = "user" + str(i)
                for folder in listdir("data/users/" + user + "/old"):
                    for file in listdir("data/users/" + user + "/old/" + folder):
                        predicted_df = pd.read_csv("data/users/" + user + "/old/" + folder + "/" + file).dropna()
                        real_df = copy.deepcopy(predicted_df)
                        real_df["action"] = folder
                        real_data_df_list.append(real_df)
                        predicted_data_df_list.append(predicted_df)

        columns = ["mean_ax", "std_ax", "max_ax", "min_ax", "var_ax", "median_ax", "mean_peak_values_ax",
                   "std_peak_values_ax", "max_peak_values_ax", "min_peak_values_ax", "var_peak_values_ax",
                   "mean_distance_peak_values_ax", "p25_ax", "p50_ax", "p75_ax", "mean_ay", "std_ay", "max_ay",
                   "min_ay", "var_ay", "median_ay", "mean_peak_values_ay", "std_peak_values_ay", "max_peak_values_ay",
                   "min_peak_values_ay", "var_peak_values_ay", "mean_distance_peak_values_ay", "p25_ay", "p50_ay",
                   "p75_ay",
                   "mean_az", "std_az", "max_az", "min_az", "var_az", "median_az", "mean_peak_values_az",
                   "std_peak_values_az", "max_peak_values_az", "min_peak_values_az", "var_peak_values_az",
                   "mean_distance_peak_values_az", "p25_az", "p50_az", "p75_az", "mean_gx", "std_gx", "max_gx",
                   "min_gx",
                   "var_gx", "median_gx", "mean_peak_values_gx", "std_peak_values_gx", "max_peak_values_gx",
                   "min_peak_values_gx",
                   "var_peak_values_gx", "mean_distance_peak_values_gx", "p25_gx", "p50_gx", "p75_gx", "mean_gy",
                   "std_gy",
                   "max_gy", "min_gy",
                   "var_gy", "median_gy", "mean_peak_values_gy", "std_peak_values_gy", "max_peak_values_gy",
                   "min_peak_values_gy",
                   "var_peak_values_gy", "mean_distance_peak_values_gy", "p25_gy", "p50_gy", "p75_gy", "mean_gz",
                   "std_gz",
                   "max_gz", "min_gz",
                   "var_gz", "median_gz", "mean_peak_values_gz", "std_peak_values_gz", "max_peak_values_gz",
                   "min_peak_values_gz",
                   "var_peak_values_gz", "mean_distance_peak_values_gz", "p25_gz", "p50_gz", "p75_gz", "action"]
        combined_real_df = pd.DataFrame(columns=columns)
        combined_predicted_df = pd.DataFrame(columns=columns)

        for df in real_data_df_list:
            combined_real_df = pd.concat([combined_real_df, df], ignore_index=True)

        for df in predicted_data_df_list:
            combined_predicted_df = pd.concat([combined_predicted_df, df], ignore_index=True)

        # combined_real_df.rename(columns={"result": "action"}, inplace=True)
        # combined_predicted_df.rename(columns={"result": "action"}, inplace=True)
        if not os.path.exists("testingData"):
            os.makedirs("testingData")
        combined_real_df.to_csv("data/processed_csv/new_users/all_users.csv", index=False)

    @staticmethod
    def generate_combinated_file_new_actions():
        real_data_df_list = []
        predicted_data_df_list = []
        for i in range(0, 25):
            if i != 10:
                user = "user" + str(i)
                for folder in listdir("data/users/" + user + "/new"):
                    for file in listdir("data/users/" + user + "/new/" + folder):
                        predicted_df = pd.read_csv("data/users/" + user + "/new/" + folder + "/" + file).dropna()
                        real_df = copy.deepcopy(predicted_df)
                        real_df["action"] = folder
                        real_data_df_list.append(real_df)
                        predicted_data_df_list.append(predicted_df)

        columns = ["mean_ax", "std_ax", "max_ax", "min_ax", "var_ax", "median_ax", "mean_peak_values_ax",
                   "std_peak_values_ax", "max_peak_values_ax", "min_peak_values_ax", "var_peak_values_ax",
                   "mean_distance_peak_values_ax", "p25_ax", "p50_ax", "p75_ax", "mean_ay", "std_ay", "max_ay",
                   "min_ay", "var_ay", "median_ay", "mean_peak_values_ay", "std_peak_values_ay", "max_peak_values_ay",
                   "min_peak_values_ay", "var_peak_values_ay", "mean_distance_peak_values_ay", "p25_ay", "p50_ay",
                   "p75_ay",
                   "mean_az", "std_az", "max_az", "min_az", "var_az", "median_az", "mean_peak_values_az",
                   "std_peak_values_az", "max_peak_values_az", "min_peak_values_az", "var_peak_values_az",
                   "mean_distance_peak_values_az", "p25_az", "p50_az", "p75_az", "mean_gx", "std_gx", "max_gx",
                   "min_gx",
                   "var_gx", "median_gx", "mean_peak_values_gx", "std_peak_values_gx", "max_peak_values_gx",
                   "min_peak_values_gx",
                   "var_peak_values_gx", "mean_distance_peak_values_gx", "p25_gx", "p50_gx", "p75_gx", "mean_gy",
                   "std_gy",
                   "max_gy", "min_gy",
                   "var_gy", "median_gy", "mean_peak_values_gy", "std_peak_values_gy", "max_peak_values_gy",
                   "min_peak_values_gy",
                   "var_peak_values_gy", "mean_distance_peak_values_gy", "p25_gy", "p50_gy", "p75_gy", "mean_gz",
                   "std_gz",
                   "max_gz", "min_gz",
                   "var_gz", "median_gz", "mean_peak_values_gz", "std_peak_values_gz", "max_peak_values_gz",
                   "min_peak_values_gz",
                   "var_peak_values_gz", "mean_distance_peak_values_gz", "p25_gz", "p50_gz", "p75_gz", "action"]
        combined_real_df = pd.DataFrame(columns=columns)
        combined_predicted_df = pd.DataFrame(columns=columns)

        for df in real_data_df_list:
            combined_real_df = pd.concat([combined_real_df, df], ignore_index=True)

        for df in predicted_data_df_list:
            combined_predicted_df = pd.concat([combined_predicted_df, df], ignore_index=True)

        # combined_real_df.rename(columns={"result": "action"}, inplace=True)
        # combined_predicted_df.rename(columns={"result": "action"}, inplace=True)
        if not os.path.exists("testingData"):
            os.makedirs("testingData")
        combined_real_df.to_csv("data/processed_csv/new_users/all_users_new_actions.csv", index=False)

    def read_user_data(self, user):
        real_data_df_list = []
        predicted_data_df_list = []

        for folder in listdir("data/users/" + user + "/old"):
            for file in listdir("data/users/" + user + "/old/" + folder):
                predicted_df = pd.read_csv("data/users/" + user + "/old/" + folder + "/" + file).dropna()
                real_df = copy.deepcopy(predicted_df)
                real_df["action"] = folder
                real_data_df_list.append(real_df)
                predicted_data_df_list.append(predicted_df)

        columns = ["mean_ax", "std_ax", "max_ax", "min_ax", "var_ax", "median_ax", "mean_peak_values_ax",
                   "std_peak_values_ax", "max_peak_values_ax", "min_peak_values_ax", "var_peak_values_ax",
                   "mean_distance_peak_values_ax", "p25_ax", "p50_ax", "p75_ax", "mean_ay", "std_ay", "max_ay",
                   "min_ay", "var_ay", "median_ay", "mean_peak_values_ay", "std_peak_values_ay", "max_peak_values_ay",
                   "min_peak_values_ay", "var_peak_values_ay", "mean_distance_peak_values_ay", "p25_ay", "p50_ay",
                   "p75_ay",
                   "mean_az", "std_az", "max_az", "min_az", "var_az", "median_az", "mean_peak_values_az",
                   "std_peak_values_az", "max_peak_values_az", "min_peak_values_az", "var_peak_values_az",
                   "mean_distance_peak_values_az", "p25_az", "p50_az", "p75_az", "mean_gx", "std_gx", "max_gx",
                   "min_gx",
                   "var_gx", "median_gx", "mean_peak_values_gx", "std_peak_values_gx", "max_peak_values_gx",
                   "min_peak_values_gx",
                   "var_peak_values_gx", "mean_distance_peak_values_gx", "p25_gx", "p50_gx", "p75_gx", "mean_gy",
                   "std_gy",
                   "max_gy", "min_gy",
                   "var_gy", "median_gy", "mean_peak_values_gy", "std_peak_values_gy", "max_peak_values_gy",
                   "min_peak_values_gy",
                   "var_peak_values_gy", "mean_distance_peak_values_gy", "p25_gy", "p50_gy", "p75_gy", "mean_gz",
                   "std_gz",
                   "max_gz", "min_gz",
                   "var_gz", "median_gz", "mean_peak_values_gz", "std_peak_values_gz", "max_peak_values_gz",
                   "min_peak_values_gz",
                   "var_peak_values_gz", "mean_distance_peak_values_gz", "p25_gz", "p50_gz", "p75_gz", "action"]
        combined_real_df = pd.DataFrame(columns=columns)
        combined_predicted_df = pd.DataFrame(columns=columns)

        for df in real_data_df_list:
            combined_real_df = pd.concat([combined_real_df, df], ignore_index=True)

        for df in predicted_data_df_list:
            combined_predicted_df = pd.concat([combined_predicted_df, df], ignore_index=True)

        self.machine_learning_utils.check_accuracy_of_real_users(
            model=pickle.load(open("data/final_models/stacking_model.pkl", 'rb')),
            real_df=combined_real_df, predicted_df=combined_predicted_df, user=user)

    @staticmethod
    def extract_user_train_from_sql(user, folders: list):
        engine = create_engine(
            'mysql+pymysql://admin:12345678@mydatabase.cayqpljkaacf.eu-north-1.rds.amazonaws.com:3306/sessions')
        sessions = \
            pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema='sessions'", engine)[
                "TABLE_NAME"].tolist()
        if not os.path.exists("data/users/" + user):
            os.makedirs("data/users/" + user)
        i = 0
        for folder in folders:
            if not os.path.exists("data/users/" + user + "/" + folder):
                os.makedirs("data/users/" + user + "/" + folder)
            pd.read_sql(sessions[i], engine).to_csv(
                "data/users/" + user + "/" + folder + "/" + sessions[i].replace(":", "") + ".csv", index=False)
            i += 1
