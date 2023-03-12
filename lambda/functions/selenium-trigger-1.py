from io import StringIO
import pandas as pd
import json
import time
import boto3

s3_resource = boto3.resource('s3')
lambda_client = boto3.client('lambda')

def main(*args):
    lambda_payload_1 = args[0]
    try:
        print(lambda_payload_1['num_pages'])
    except:
        lambda_payload_1 = eval(lambda_payload_1)
    response = lambda_client.invoke(
        FunctionName='selenium-test',
        InvocationType='RequestResponse',
        Payload=json.dumps(lambda_payload_1)
    )
    if response['StatusCode'] == 200:
        time.sleep(6)
        lambda_payload_2 = {'table':'boxes', 'num_pages':lambda_payload_1['num_pages']}
        lambda_client.invoke(
            FunctionName='selenium-trigger-2',
            InvocationType='Event',
            Payload=json.dumps(lambda_payload_2)
        )
    else:
        pass
        # add some error reporting code here later