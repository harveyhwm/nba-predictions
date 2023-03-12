import json
import boto3
import pathlib

from io import StringIO, BytesIO

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
    
s3_client = client = boto3.client('s3')
s3_resource = boto3.resource('s3')
bucket = 'sagemaker-pipelines-hwm'

def lambda_handler(event, context):

    evaluation_s3_uri = 's3://sagemaker-pipelines-hwm/nba/eval/accuracy.json' # event['evaluation_s3_uri']
    path_parts = evaluation_s3_uri.replace('s3://', '').split('/')
    bucket = path_parts.pop(0)
    key = '/'.join(path_parts)

    content = s3_client.get_object(Bucket=bucket, Key=key)
    text = content['Body'].read().decode()
    evaluation_json = json.loads(text)
    accuracy = evaluation_json['accuracy'] #['regression_metrics']['mse']['value']
    
    # email dispatch code
    msg = MIMEMultipart('alternative')
    msg['From'] = formataddr((str(Header('MyWebsite', 'utf-8')), params['my_email']))
    msg['To'] = params['my_email']
    msg['Subject'] = 'Great Accuracy! {}'.format(accuracy)
    html = 'be proud of this great accuracy wooop!'
    msg.attach(MIMEText(html, 'html'))
    s = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    s.ehlo()
    s.login(params['my_email'], params['my_app_password'])
    s.sendmail(params['my_email'], params['my_email'], msg.as_string())
    s.quit()
    
    report_dict = {
        'regression_metrics': {
            'mse': {'value': 0.5, 'standard_deviation': 'NaN'},
        },
    }
        
    output_buffer = StringIO()
    output_buffer.write(json.dumps(report_dict))
    s3_resource.Object(bucket, 'nba/eval/accuracy_good.json').put(Body=output_buffer.getvalue())
    
    return {'statusCode': 200, 'body': json.dumps('E-Mail Sent Successfully')}
