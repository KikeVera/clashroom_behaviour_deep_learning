import copy
import os
import pandas as pd
import numpy as np
from statistics import mode


class ETL:

    def __init__(self):
        self.samples_size = 200
        self.num_peaks = 5
        self.freq = 20

    def __process_sensors(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.sort_values(by=['ts'], ignore_index=True)
        dataframe["az"] = dataframe["az"].apply(lambda value: value.rstrip(value[-1]))
        dataframe["az"] = pd.to_numeric(dataframe["az"])
        dataframe["gz"] = dataframe["gz"].apply(lambda value: value.rstrip(value[-1]))
        dataframe["gz"] = pd.to_numeric(dataframe["az"])
        dataframe['group'] = dataframe.reset_index().index / self.samples_size
        dataframe['group'] = dataframe['group'].astype("int32")
        dataframe.drop('user', inplace=True, axis=1)
        dataframe.drop('ts', inplace=True, axis=1)
        dataframe = dataframe.groupby(['action', 'group']).agg(lambda value: list(value)).reset_index()
        dataframe.drop('group', inplace=True, axis=1)
        dataframe = dataframe[
            (dataframe['action'] != 'B') & (dataframe['action'] != 'C') & (dataframe['action'] != 'G') &
            (dataframe['action'] != 'M') & (dataframe['action'] != 'O') & (dataframe['action'] != 'P') &
            (dataframe['action'] != 'S') & (dataframe['action'] != 'H') & (dataframe['action'] != 'J')]
        dataframe.loc[dataframe['action'] == 'A', 'action'] = 'Walking'
        dataframe.loc[dataframe['action'] == 'D', 'action'] = 'Sitting'
        dataframe.loc[dataframe['action'] == 'E', 'action'] = 'Standing'
        dataframe.loc[dataframe['action'] == 'F', 'action'] = 'Typing'
        dataframe.loc[dataframe['action'] == 'I', 'action'] = 'Eating'
        dataframe.loc[dataframe['action'] == 'K', 'action'] = 'Drinking'
        dataframe.loc[dataframe['action'] == 'L', 'action'] = 'Eating'
        dataframe.loc[dataframe['action'] == 'Q', 'action'] = 'Writing'
        dataframe.loc[dataframe['action'] == 'R', 'action'] = 'Clapping'

        dataframe = dataframe[dataframe['ax'].map(len) == self.samples_size]
        dataframe.reset_index()
        return dataframe

    def prepare_dataframe_to_train(self, df: pd.DataFrame) -> pd.DataFrame:
        df['mean_ax'] = df['ax'].map(np.mean)
        df['std_ax'] = df['ax'].map(np.std)
        df['max_ax'] = df['ax'].map(np.max)
        df['min_ax'] = df['ax'].map(np.min)
        df['var_ax'] = df['ax'].map(np.var)
        df['median_ax'] = df['ax'].map(np.median)
        df["peak_values_ax"] = df['ax']
        df["peak_values_ax"] = df["peak_values_ax"].map(self.__get_peak_n_values)
        df['mean_peak_values_ax'] = df['peak_values_ax'].map(np.mean)
        df['std_peak_values_ax'] = df['peak_values_ax'].map(np.std)
        df['max_peak_values_ax'] = df['peak_values_ax'].map(np.max)
        df['min_peak_values_ax'] = df['peak_values_ax'].map(np.min)
        df['var_peak_values_ax'] = df['peak_values_ax'].map(np.var)
        # df['mean_distance_peak_values_ax'] = df.apply(lambda x: self.get_peak_n_mean_distance(
        #     peak_values_list=x['peak_values_ax'], values_list=x['ax']), axis=1)
        df['p25_ax'] = df['ax'].apply(lambda x: np.percentile(a=x, q=25))
        df['p50_ax'] = df['ax'].apply(lambda x: np.percentile(a=x, q=50))
        df['p75_ax'] = df['ax'].apply(lambda x: np.percentile(a=x, q=75))

        df['mean_ay'] = df['ay'].map(np.mean)
        df['std_ay'] = df['ay'].map(np.std)
        df['max_ay'] = df['ay'].map(np.max)
        df['min_ay'] = df['ay'].map(np.min)
        df['var_ay'] = df['ay'].map(np.var)
        df['median_ay'] = df['ay'].map(np.median)
        df["peak_values_ay"] = df['ay']
        df["peak_values_ay"] = df["peak_values_ay"].map(self.__get_peak_n_values)
        df['mean_peak_values_ay'] = df['peak_values_ay'].map(np.mean)
        df['std_peak_values_ay'] = df['peak_values_ay'].map(np.std)
        df['max_peak_values_ay'] = df['peak_values_ay'].map(np.max)
        df['min_peak_values_ay'] = df['peak_values_ay'].map(np.min)
        df['var_peak_values_ay'] = df['peak_values_ay'].map(np.var)
        # df['mean_distance_peak_values_ay'] = df.apply(lambda x: self.get_peak_n_mean_distance(
        #     peak_values_list=x['peak_values_ay'], values_list=x['ay']), axis=1)
        df['p25_ay'] = df['ay'].apply(lambda x: np.percentile(a=x, q=25))
        df['p50_ay'] = df['ay'].apply(lambda x: np.percentile(a=x, q=50))
        df['p75_ay'] = df['ay'].apply(lambda x: np.percentile(a=x, q=75))

        df['mean_az'] = df['az'].map(np.mean)
        df['std_az'] = df['az'].map(np.std)
        df['max_az'] = df['az'].map(np.max)
        df['min_az'] = df['az'].map(np.min)
        df['var_az'] = df['az'].map(np.var)
        df['median_az'] = df['az'].map(np.median)
        df["peak_values_az"] = df['az']
        df["peak_values_az"] = df["peak_values_az"].map(self.__get_peak_n_values)
        df['mean_peak_values_az'] = df['peak_values_az'].map(np.mean)
        df['std_peak_values_az'] = df['peak_values_az'].map(np.std)
        df['max_peak_values_az'] = df['peak_values_az'].map(np.max)
        df['min_peak_values_az'] = df['peak_values_az'].map(np.min)
        df['var_peak_values_az'] = df['peak_values_az'].map(np.var)
        # df['mean_distance_peak_values_az'] = df.apply(lambda x: self.get_peak_n_mean_distance(
        #     peak_values_list=x['peak_values_az'], values_list=x['az']), axis=1)
        df['p25_az'] = df['az'].apply(lambda x: np.percentile(a=x, q=25))
        df['p50_az'] = df['az'].apply(lambda x: np.percentile(a=x, q=50))
        df['p75_az'] = df['az'].apply(lambda x: np.percentile(a=x, q=75))

        df['mean_gx'] = df['gx'].map(np.mean)
        df['std_gx'] = df['gx'].map(np.std)
        df['max_gx'] = df['gx'].map(np.max)
        df['min_gx'] = df['gx'].map(np.min)
        df['var_gx'] = df['gx'].map(np.var)
        df['median_gx'] = df['gx'].map(np.median)
        df["peak_values_gx"] = df['gx']
        df["peak_values_gx"] = df["peak_values_gx"].map(self.__get_peak_n_values)
        df['mean_peak_values_gx'] = df['peak_values_gx'].map(np.mean)
        df['std_peak_values_gx'] = df['peak_values_gx'].map(np.std)
        df['max_peak_values_gx'] = df['peak_values_gx'].map(np.max)
        df['min_peak_values_gx'] = df['peak_values_gx'].map(np.min)
        df['var_peak_values_gx'] = df['peak_values_gx'].map(np.var)
        # df['mean_distance_peak_values_gx'] = df.apply(lambda x: self.get_peak_n_mean_distance(
        #     peak_values_list=x['peak_values_gx'], values_list=x['gx']), axis=1)
        df['p25_gx'] = df['gx'].apply(lambda x: np.percentile(a=x, q=25))
        df['p50_gx'] = df['gx'].apply(lambda x: np.percentile(a=x, q=50))
        df['p75_gx'] = df['gx'].apply(lambda x: np.percentile(a=x, q=75))

        df['mean_gy'] = df['gy'].map(np.mean)
        df['std_gy'] = df['gy'].map(np.std)
        df['max_gy'] = df['gy'].map(np.max)
        df['min_gy'] = df['gy'].map(np.min)
        df['var_gy'] = df['gy'].map(np.var)
        df['median_gy'] = df['gy'].map(np.median)
        df["peak_values_gy"] = df['gy']
        df["peak_values_gy"] = df["peak_values_gy"].map(self.__get_peak_n_values)
        df['mean_peak_values_gy'] = df['peak_values_gy'].map(np.mean)
        df['std_peak_values_gy'] = df['peak_values_gy'].map(np.std)
        df['max_peak_values_gy'] = df['peak_values_gy'].map(np.max)
        df['min_peak_values_gy'] = df['peak_values_gy'].map(np.min)
        df['var_peak_values_gy'] = df['peak_values_gy'].map(np.var)
        # df['mean_distance_peak_values_gy'] = df.apply(lambda x: self.get_peak_n_mean_distance(
        #     peak_values_list=x['peak_values_gy'], values_list=x['gy']), axis=1)
        df['p25_gy'] = df['gy'].apply(lambda x: np.percentile(a=x, q=25))
        df['p50_gy'] = df['gy'].apply(lambda x: np.percentile(a=x, q=50))
        df['p75_gy'] = df['gy'].apply(lambda x: np.percentile(a=x, q=75))

        df['mean_gz'] = df['gz'].map(np.mean)
        df['std_gz'] = df['gz'].map(np.std)
        df['max_gz'] = df['gz'].map(np.max)
        df['min_gz'] = df['gz'].map(np.min)
        df['var_gz'] = df['gz'].map(np.var)
        df['median_gz'] = df['gz'].map(np.median)
        df["peak_values_gz"] = df['gz']
        df["peak_values_gz"] = df["peak_values_gz"].map(self.__get_peak_n_values)
        df['mean_peak_values_gz'] = df['peak_values_gz'].map(np.mean)
        df['std_peak_values_gz'] = df['peak_values_gz'].map(np.std)
        df['max_peak_values_gz'] = df['peak_values_gz'].map(np.max)
        df['min_peak_values_gz'] = df['peak_values_gz'].map(np.min)
        df['var_peak_values_gz'] = df['peak_values_gz'].map(np.var)
        # df['mean_distance_peak_values_gz'] = df.apply(lambda x: self.get_peak_n_mean_distance(
        #     peak_values_list=x['peak_values_gz'], values_list=x['gz']), axis=1)
        df['p25_gz'] = df['gz'].apply(lambda x: np.percentile(a=x, q=25))
        df['p50_gz'] = df['gz'].apply(lambda x: np.percentile(a=x, q=50))
        df['p75_gz'] = df['gz'].apply(lambda x: np.percentile(a=x, q=75))

        df.drop('ax', inplace=True, axis=1)
        df.drop('ay', inplace=True, axis=1)
        df.drop('az', inplace=True, axis=1)
        df.drop('gx', inplace=True, axis=1)
        df.drop('gy', inplace=True, axis=1)
        df.drop('gz', inplace=True, axis=1)
        df.drop('peak_values_ax', inplace=True, axis=1)
        df.drop('peak_values_ay', inplace=True, axis=1)
        df.drop('peak_values_az', inplace=True, axis=1)
        df.drop('peak_values_gx', inplace=True, axis=1)
        df.drop('peak_values_gy', inplace=True, axis=1)
        df.drop('peak_values_gz', inplace=True, axis=1)

        return df

    def pre_process(self, accel_path: str, gyro_path: str):
        accel = pd.read_table(accel_path, delimiter=',',
                              names=["user", "action", "ts", "ax", "ay", "az"])
        gyro = pd.read_table(gyro_path, delimiter=',',
                             names=["user", "action", "ts", "gx", "gy", "gz"])
        dataframe = accel.merge(gyro, on=["user", "action", "ts"], how="inner")

        return dataframe

    def __get_peak_n_values(self, values_list: list) -> list:
        list_len = len(values_list)
        copy_list = copy.deepcopy(values_list)
        while list_len > self.samples_size:
            copy_list.remove(min(copy_list, key=abs))
            list_len -= 1
        return copy_list

    def __get_peak_n_mean_distance(self, peak_values_list: list, values_list: list, freq=None) -> np.ndarray:
        if freq is None:
            freq = self.freq
        copy_list = copy.deepcopy(values_list)
        index_list = []
        for value in peak_values_list:
            positions = [i for i, x in enumerate(copy_list) if x == value]
            cont = 0
            pos = positions[cont]
            while pos in index_list:
                cont += 1
                pos = positions[cont]
            index_list.append(pos)
        peak_distance = []
        for i in range(1, len(index_list)):
            peak_distance.append((index_list[i] - index_list[i - 1]) / freq)

        return np.mean(peak_distance)

    def process_wisdm(self) -> pd.DataFrame:
        combined_df = pd.DataFrame(data={"action": [], "ax": [], "ay": [], "az": [], "gx": [], "gy": [], "gz": []})

        for i in range(1600, 1651):
            accel_gyro = self.pre_process('data/raw_data/wisdm/accel/data_' + str(i) + '_accel_watch.txt',
                                          'data/raw_data/wisdm/gyro/data_' + str(i) + '_gyro_watch.txt')

            actions_df = self.__process_sensors(accel_gyro)

            combined_df = pd.concat([combined_df, actions_df], ignore_index=True)

        combined_df = combined_df.dropna().reset_index(drop=True)

        return combined_df
