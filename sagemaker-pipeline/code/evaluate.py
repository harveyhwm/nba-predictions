import os
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pathlib
import tarfile
import boto3

from io import StringIO, BytesIO

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket = params['bucket_pipeline']

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

if __name__ == '__main__':

    install('tensorflow==2.9.2') # we can add any package similarly
    
    target = 'home_win'
    
    model_path = f'/opt/ml/processing/model/model.tar.gz'
    with tarfile.open(model_path, 'r:gz') as tar:
        tar.extractall('./model')
    import tensorflow as tf

    model = tf.keras.models.load_model('./model/1')
    test_path = '/opt/ml/processing/test/'
    x_test = pd.read_csv(os.path.join(test_path, 'test.csv'))
    y_test = x_test.pop(target)
    x_test, y_test = np.array(x_test), np.array(y_test)
    #y_test = np.load(os.path.join(test_path, 'y_test.npy'))
    scores = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest Loss & Accuracy:', scores)

    report_dict = {'accuracy': scores[1]}

    output_dir = '/opt/ml/processing/evaluation'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f'{output_dir}/accuracy_eval.json'
    with open(evaluation_path, 'w') as f:
        f.write(json.dumps(report_dict))
        
    output_buffer = StringIO()
    output_buffer.write(json.dumps(report_dict))
    s3_resource.Object(bucket, 'nba/eval/accuracy_eval.json').put(Body=output_buffer.getvalue())
