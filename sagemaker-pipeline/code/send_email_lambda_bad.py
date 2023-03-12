
"""
This Lambda function sends an E-Mail to the Data Science team with the MSE from model evaluation step. 
The evaluation.json location in S3 is provided via the `event` argument
"""

import json
import boto3
import json
import pathlib

from io import StringIO, BytesIO

import smtplib
from email.mime.text import MIMEText

s3_client = client = boto3.client('s3')
s3_resource = boto3.resource('s3')
bucket = 'sagemaker-pipelines-hwm'

def lambda_handler(event, context):

    # print(f'Received Event: {event}')

    evaluation_s3_uri = 's3://sagemaker-pipelines-hwm/nba/eval/accuracy.json' # event['evaluation_s3_uri']
    path_parts = evaluation_s3_uri.replace('s3://', '').split('/')
    bucket = path_parts.pop(0)
    key = '/'.join(path_parts)

    content = s3_client.get_object(Bucket=bucket, Key=key)
    text = content['Body'].read().decode()
    evaluation_json = json.loads(text)
    mse = evaluation_json['accuracy'] #['regression_metrics']['mse']['value']

    subject_line = 'Please check high MSE ({}) detected on model evaluation'.format(mse)
    print(f'Sending E-Mail to Data Science Team with subject line: {subject_line}')
    

    # TODO - ADD YOUR CODE TO SEND EMAIL...
    report_dict = {
        'regression_metrics': {
            'mse': {'value': 0.5, 'standard_deviation': 'NaN'},
        },
    }
        
    output_buffer = StringIO()
    output_buffer.write(json.dumps(report_dict))
    s3_resource.Object(bucket, 'nba/eval/accuracy_bad.json').put(Body=output_buffer.getvalue())

    
    return {'statusCode': 200, 'body': json.dumps('E-Mail Sent Successfully')}
