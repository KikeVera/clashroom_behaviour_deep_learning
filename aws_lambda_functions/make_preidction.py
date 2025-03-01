import boto3
import pandas as pd
import numpy as np
import json
import copy
import pickle
import sklearn
import botocore
import logging
import pymysql
from sqlalchemy import create_engine
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from io import StringIO, BytesIO

NUM_PEAKS=5
FREQ = 20
MODEL = "mlp_knn_rf_model.pkl"


def lambda_handler(event, context):
    
    received_data=json.loads(event['body'])
    
    
   
    columns = ["mean_ax", "std_ax", "max_ax", "min_ax", "var_ax", "median_ax", "mean_peak_values_ax",
               "std_peak_values_ax", "max_peak_values_ax", "min_peak_values_ax", "var_peak_values_ax",
               "mean_distance_peak_values_ax", "p25_ax", "p50_ax", "p75_ax", "mean_ay", "std_ay", "max_ay", 
               "min_ay", "var_ay", "median_ay", "mean_peak_values_ay", "std_peak_values_ay", "max_peak_values_ay",
               "min_peak_values_ay", "var_peak_values_ay", "mean_distance_peak_values_ay", "p25_ay", "p50_ay", "p75_ay",
               "mean_az", "std_az", "max_az", "min_az", "var_az", "median_az", "mean_peak_values_az",
               "std_peak_values_az", "max_peak_values_az", "min_peak_values_az", "var_peak_values_az",
               "mean_distance_peak_values_az", "p25_az", "p50_az", "p75_az", "mean_gx", "std_gx", "max_gx", "min_gx", 
               "var_gx", "median_gx", "mean_peak_values_gx", "std_peak_values_gx", "max_peak_values_gx", "min_peak_values_gx", 
               "var_peak_values_gx", "mean_distance_peak_values_gx", "p25_gx", "p50_gx", "p75_gx", "mean_gy", "std_gy", "max_gy", "min_gy", 
               "var_gy", "median_gy", "mean_peak_values_gy", "std_peak_values_gy", "max_peak_values_gy", "min_peak_values_gy", 
               "var_peak_values_gy", "mean_distance_peak_values_gy", "p25_gy", "p50_gy", "p75_gy", "mean_gz", "std_gz", "max_gz", "min_gz",
               "var_gz", "median_gz", "mean_peak_values_gz", "std_peak_values_gz", "max_peak_values_gz", "min_peak_values_gz",
               "var_peak_values_gz", "mean_distance_peak_values_gz", "p25_gz", "p50_gz", "p75_gz"]
               
               
    predict_df = pd.DataFrame(np.nan, index=range(0, 1), columns=columns)
    
    values_ax=received_data["accel"]["x"]
    values_ay=received_data["accel"]["y"]
    values_az=received_data["accel"]["z"]
    values_gx=received_data["gyro"]["x"]
    values_gy=received_data["gyro"]["y"]
    values_gz=received_data["gyro"]["z"]
    
    peaks_ax=get_peak_n_values(received_data["accel"]["x"])
    peaks_ay=get_peak_n_values(received_data["accel"]["y"])
    peaks_az=get_peak_n_values(received_data["accel"]["z"])
    peaks_gx=get_peak_n_values(received_data["gyro"]["x"])
    peaks_gy=get_peak_n_values(received_data["gyro"]["y"])
    peaks_gz=get_peak_n_values(received_data["gyro"]["z"])
    
    predict_df["mean_ax"][0]=np.mean(values_ax)
    predict_df["std_ax"][0]=np.std(values_ax)
    predict_df["max_ax"][0]=np.max(values_ax)
    predict_df["min_ax"][0]=np.min(values_ax)
    predict_df["var_ax"][0]=np.var(values_ax)
    predict_df["median_ax"][0]=np.median(values_ax)
    predict_df["mean_peak_values_ax"][0]=np.mean(peaks_ax)
    predict_df["std_peak_values_ax"][0]=np.std(peaks_ax)
    predict_df["max_peak_values_ax"][0]=np.max(peaks_ax)
    predict_df["min_peak_values_ax"][0]=np.min(peaks_ax)
    predict_df["var_peak_values_ax"][0]=np.var(peaks_ax)
    predict_df["mean_distance_peak_values_ax"][0]=get_peak_n_mean_distance(peaks_ax, values_ax)
    predict_df["p25_ax"][0]=np.percentile(a=values_ax, q=25)
    predict_df["p50_ax"][0]=np.percentile(a=values_ax, q=50)
    predict_df["p75_ax"][0]=np.percentile(a=values_ax, q=75)
    
    predict_df["mean_ay"][0]=np.mean(values_ay)
    predict_df["std_ay"][0]=np.std(values_ay)
    predict_df["max_ay"][0]=np.max(values_ay)
    predict_df["min_ay"][0]=np.min(values_ay)
    predict_df["var_ay"][0]=np.var(values_ay)
    predict_df["median_ay"][0]=np.median(values_ay)
    predict_df["mean_peak_values_ay"][0]=np.mean(peaks_ay)
    predict_df["std_peak_values_ay"][0]=np.std(peaks_ay)
    predict_df["max_peak_values_ay"][0]=np.max(peaks_ay)
    predict_df["min_peak_values_ay"][0]=np.min(peaks_ay)
    predict_df["var_peak_values_ay"][0]=np.var(peaks_ay)
    predict_df["mean_distance_peak_values_ay"][0]=get_peak_n_mean_distance(peaks_ay, values_ay)
    predict_df["p25_ay"][0]=np.percentile(a=values_ay, q=25)
    predict_df["p50_ay"][0]=np.percentile(a=values_ay, q=50)
    predict_df["p75_ay"][0]=np.percentile(a=values_ay, q=75)
    
    predict_df["mean_az"][0]=np.mean(values_az)
    predict_df["std_az"][0]=np.std(values_az)
    predict_df["max_az"][0]=np.max(values_az)
    predict_df["min_az"][0]=np.min(values_az)
    predict_df["var_az"][0]=np.var(values_az)
    predict_df["median_az"][0]=np.median(values_az)
    predict_df["mean_peak_values_az"][0]=np.mean(peaks_az)
    predict_df["std_peak_values_az"][0]=np.std(peaks_az)
    predict_df["max_peak_values_az"][0]=np.max(peaks_az)
    predict_df["min_peak_values_az"][0]=np.min(peaks_az)
    predict_df["var_peak_values_az"][0]=np.var(peaks_az)
    predict_df["mean_distance_peak_values_az"][0]=get_peak_n_mean_distance(peaks_az, values_az)
    predict_df["p25_az"][0]=np.percentile(a=values_az, q=25)
    predict_df["p50_az"][0]=np.percentile(a=values_az, q=50)
    predict_df["p75_az"][0]=np.percentile(a=values_az, q=75)
    
    predict_df["mean_gx"][0]=np.mean(values_gx)
    predict_df["std_gx"][0]=np.std(values_gx)
    predict_df["max_gx"][0]=np.max(values_gx)
    predict_df["min_gx"][0]=np.min(values_gx)
    predict_df["var_gx"][0]=np.var(values_gx)
    predict_df["median_gx"][0]=np.median(values_gx)
    predict_df["mean_peak_values_gx"][0]=np.mean(peaks_gx)
    predict_df["std_peak_values_gx"][0]=np.std(peaks_gx)
    predict_df["max_peak_values_gx"][0]=np.max(peaks_gx)
    predict_df["min_peak_values_gx"][0]=np.min(peaks_gx)
    predict_df["var_peak_values_gx"][0]=np.var(peaks_gx)
    predict_df["mean_distance_peak_values_gx"][0]=get_peak_n_mean_distance(peaks_gx, values_gx)
    predict_df["p25_gx"][0]=np.percentile(a=values_gx, q=25)
    predict_df["p50_gx"][0]=np.percentile(a=values_gx, q=50)
    predict_df["p75_gx"][0]=np.percentile(a=values_gx, q=75)
    
    predict_df["mean_gy"][0]=np.mean(values_gy)
    predict_df["std_gy"][0]=np.std(values_gy)
    predict_df["max_gy"][0]=np.max(values_gy)
    predict_df["min_gy"][0]=np.min(values_gy)
    predict_df["var_gy"][0]=np.var(values_gy)
    predict_df["median_gy"][0]=np.median(values_gy)
    predict_df["mean_peak_values_gy"][0]=np.mean(peaks_gy)
    predict_df["std_peak_values_gy"][0]=np.std(peaks_gy)
    predict_df["max_peak_values_gy"][0]=np.max(peaks_gy)
    predict_df["min_peak_values_gy"][0]=np.min(peaks_gy)
    predict_df["var_peak_values_gy"][0]=np.var(peaks_gy)
    predict_df["mean_distance_peak_values_gy"][0]=get_peak_n_mean_distance(peaks_gy, values_gy)
    predict_df["p25_gy"][0]=np.percentile(a=values_gy, q=25)
    predict_df["p50_gy"][0]=np.percentile(a=values_gy, q=50)
    predict_df["p75_gy"][0]=np.percentile(a=values_gy, q=75)
    
    predict_df["mean_gz"][0]=np.mean(values_gz)
    predict_df["std_gz"][0]=np.std(values_gz)
    predict_df["max_gz"][0]=np.max(values_gz)
    predict_df["min_gz"][0]=np.min(values_gz)
    predict_df["var_gz"][0]=np.var(values_gz)
    predict_df["median_gz"][0]=np.median(values_gz)
    predict_df["mean_peak_values_gz"][0]=np.mean(peaks_gz)
    predict_df["std_peak_values_gz"][0]=np.std(peaks_gz)
    predict_df["max_peak_values_gz"][0]=np.max(peaks_gz)
    predict_df["min_peak_values_gz"][0]=np.min(peaks_gz)
    predict_df["var_peak_values_gz"][0]=np.var(peaks_gz)
    predict_df["mean_distance_peak_values_gz"][0]=get_peak_n_mean_distance(peaks_gz, values_gz)
    predict_df["p25_gz"][0]=np.percentile(a=values_gz, q=25)
    predict_df["p50_gz"][0]=np.percentile(a=values_gz, q=50)
    predict_df["p75_gz"][0]=np.percentile(a=values_gz, q=75)
    
    s3_resource = boto3.resource('s3')
    bucket = 'clashroombehaviourml' 
   
    model = pickle.loads(s3_resource.Bucket(bucket).Object(MODEL).get()['Body'].read())
   
        
   
    result=model.predict(predict_df)
    
    predict_df["action"]=result
    
    engine = create_engine('mysql+pymysql://admin:12345678@mydatabase.cayqpljkaacf.eu-north-1.rds.amazonaws.com:3306/sessions')
    
    
    try:
        previous_df = pd.read_sql(received_data["date"], engine)
        combined_df = pd.concat([previous_df, predict_df])
    except Exception as e:
        combined_df=predict_df
        
    
    
  
    combined_df=combined_df.reset_index(drop=True)
    combined_df.to_sql(name=received_data["date"], con=engine, if_exists='replace', index=False)
    
    
    print(result)

   

    
    
 
    

    
    # Devolver el contenido del archivo como respuesta
    return {
        'statusCode': 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "action": result[0]
        })
    }


def get_peak_n_values(values_list):
    list_len = len(values_list)
    copy_list = copy.deepcopy(values_list)
    while list_len > NUM_PEAKS:
        copy_list.remove(min(copy_list, key=abs))
        list_len -= 1
    return copy_list
    

def get_peak_n_mean_distance(peak_values_list, values_list, freq = FREQ):
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