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
    lambda_payload_2 = args[0]
    try:
        print(lambda_payload_2['num_pages'])
    except:
        lambda_payload_2 = eval(lambda_payload_2)
    response = lambda_client.invoke(
        FunctionName='selenium-test',
        InvocationType='RequestResponse',
        Payload=json.dumps(lambda_payload_2)
    )
    if response['StatusCode'] == 200:
        time.sleep(6)
        lambda_payload_3 = {'table':'plays', 'num_pages':lambda_payload_2['num_pages']}
        lambda_client.invoke(
            FunctionName='selenium-trigger-3',
            InvocationType='Event',
            Payload=json.dumps(lambda_payload_3)
        )
    else:
        write_s3('data/boxes_fail.csv',pd.DataFrame(['test_val']))
        pass
        # add some error reporting code here later