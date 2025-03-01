import json
import boto3
import pandas as pd
import pymysql
from sqlalchemy import create_engine

def lambda_handler(event, context):
    
    received_data=json.loads(event['body'])
    file_number=received_data["number"]
   
    engine = create_engine('mysql+pymysql://admin:12345678@mydatabase.cayqpljkaacf.eu-north-1.rds.amazonaws.com:3306/sessions')
    sessions=pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema='sessions'", engine)["TABLE_NAME"].tolist()
    
 
    
    print(sessions)
   
    if file_number<0:
        file_number=len(sessions)-1
    elif file_number>len(sessions)-1:
        file_number=0
    
    n=len(sessions) - file_number - 1
    
   

    
    
    try:
        session=sessions[n]
        predictions = pd.read_sql(session, engine)["action"].tolist()
    except:
        predictions=["Not sessions recorded"]
        session=""
        file_number=0
        
    
    print(predictions)
    
    
    return {
        'statusCode': 200,
        "body": json.dumps({
            "predictions": predictions,
            "session":session ,
            "number": file_number
        })
    }
