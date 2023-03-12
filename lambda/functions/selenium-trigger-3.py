from io import StringIO
import pandas as pd
import json
import time
import boto3

s3_resource = boto3.resource('s3')
lambda_client = boto3.client('lambda')

def write_s3(file_path, myfile, bucket='hwm-nba'):
    if type(myfile) == pd.core.frame.DataFrame:
        output_buffer = StringIO()
        myfile.to_csv(output_buffer, index=False)
        myfile = output_buffer
    s3_resource.Object(bucket, file_path).put(Body=myfile.getvalue())

def main(*args):
    lambda_payload_3 = args[0]
    try:
        print(lambda_payload_3['num_pages'])
    except:
        lambda_payload_3 = eval(lambda_payload_3)
    response = lambda_client.invoke(
        FunctionName='selenium-test',
        InvocationType='RequestResponse',
        Payload=json.dumps(lambda_payload_3)
    )
    if response['StatusCode'] != 200:
        write_s3('data/plays_success.csv',pd.DataFrame(['test_val']))
        # add some error reporting code here later