import copy
import os

import pandas as pd


def generate_combinated_file():
    columns = ["mean_ax", "std_ax", "max_ax", "min_ax", "var_ax", "median_ax", "mean_peak_values_ax",
               "std_peak_values_ax", "max_peak_values_ax", "min_peak_values_ax", "var_peak_values_ax",
               "p25_ax", "p50_ax", "p75_ax", "mean_ay", "std_ay", "max_ay",
               "min_ay", "var_ay", "median_ay", "mean_peak_values_ay", "std_peak_values_ay", "max_peak_values_ay",
               "min_peak_values_ay", "var_peak_values_ay", "p25_ay", "p50_ay",
               "p75_ay",
               "mean_az", "std_az", "max_az", "min_az", "var_az", "median_az", "mean_peak_values_az",
               "std_peak_values_az", "max_peak_values_az", "min_peak_values_az", "var_peak_values_az",
               "p25_az", "p50_az", "p75_az", "mean_gx", "std_gx", "max_gx",
               "min_gx",
               "var_gx", "median_gx", "mean_peak_values_gx", "std_peak_values_gx", "max_peak_values_gx",
               "min_peak_values_gx",
               "var_peak_values_gx", "p25_gx", "p50_gx", "p75_gx", "mean_gy",
               "std_gy",
               "max_gy", "min_gy",
               "var_gy", "median_gy", "mean_peak_values_gy", "std_peak_values_gy", "max_peak_values_gy",
               "min_peak_values_gy",
               "var_peak_values_gy", "p25_gy", "p50_gy", "p75_gy", "mean_gz",
               "std_gz",
               "max_gz", "min_gz",
               "var_gz", "median_gz", "mean_peak_values_gz", "std_peak_values_gz", "max_peak_values_gz",
               "min_peak_values_gz",
               "var_peak_values_gz", "p25_gz", "p50_gz", "p75_gz", "action"]
    real_data_df_list = []
    predicted_data_df_list = []
    for i in range(1, 16):
        if i != 10:
            user = "user" + str(i)
            for folder in os.listdir("data/users/stage-1/" + user + "/old"):
                for file in os.listdir("data/users/stage-1/" + user + "/old/" + folder):
                    predicted_df = pd.read_csv("data/users/stage-1/" + user + "/old/" + folder + "/" + file).dropna()[columns]
                    real_df = copy.deepcopy(predicted_df)
                    real_df["action"] = folder
                    real_data_df_list.append(real_df)
                    predicted_data_df_list.append(predicted_df)


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
    combined_real_df.to_csv("data/datasets/new_users/all_users.csv", index=False)


def generate_combinated_file_new_actions():
    columns = ["mean_ax", "std_ax", "max_ax", "min_ax", "var_ax", "median_ax", "mean_peak_values_ax",
               "std_peak_values_ax", "max_peak_values_ax", "min_peak_values_ax", "var_peak_values_ax",
               "p25_ax", "p50_ax", "p75_ax", "mean_ay", "std_ay", "max_ay",
               "min_ay", "var_ay", "median_ay", "mean_peak_values_ay", "std_peak_values_ay", "max_peak_values_ay",
               "min_peak_values_ay", "var_peak_values_ay", "p25_ay", "p50_ay",
               "p75_ay",
               "mean_az", "std_az", "max_az", "min_az", "var_az", "median_az", "mean_peak_values_az",
               "std_peak_values_az", "max_peak_values_az", "min_peak_values_az", "var_peak_values_az",
               "p25_az", "p50_az", "p75_az", "mean_gx", "std_gx", "max_gx",
               "min_gx",
               "var_gx", "median_gx", "mean_peak_values_gx", "std_peak_values_gx", "max_peak_values_gx",
               "min_peak_values_gx",
               "var_peak_values_gx", "p25_gx", "p50_gx", "p75_gx", "mean_gy",
               "std_gy",
               "max_gy", "min_gy",
               "var_gy", "median_gy", "mean_peak_values_gy", "std_peak_values_gy", "max_peak_values_gy",
               "min_peak_values_gy",
               "var_peak_values_gy", "p25_gy", "p50_gy", "p75_gy", "mean_gz",
               "std_gz",
               "max_gz", "min_gz",
               "var_gz", "median_gz", "mean_peak_values_gz", "std_peak_values_gz", "max_peak_values_gz",
               "min_peak_values_gz",
               "var_peak_values_gz", "p25_gz", "p50_gz", "p75_gz", "action"]
    real_data_df_list = []
    predicted_data_df_list = []
    for i in range(1, 16):
        if i != 10:
            user = "user" + str(i)
            for folder in os.listdir("data/users/stage-1/" + user + "/new"):
                for file in os.listdir("data/users/stage-1/" + user + "/new/" + folder):
                    predicted_df = pd.read_csv("data/users/stage-1/" + user + "/new/" + folder + "/" + file).dropna()[columns]
                    real_df = copy.deepcopy(predicted_df)
                    real_df["action"] = folder
                    real_data_df_list.append(real_df)
                    predicted_data_df_list.append(predicted_df)


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
    combined_real_df.to_csv("data/datasets/new_users/all_users_new_actions.csv", index=False)
