import argparse
import boto3
import numpy as np
import pandas as pd
import os
import json
import pathlib
import subprocess
import sys
import tarfile
import glob
import shutil

import datetime
from datetime import timedelta as td
from datetime import datetime as dt

from io import StringIO, BytesIO

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket = params['bucket_pipeline']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--train', type=str, default='/opt/ml/processing/train/') 
    parser.add_argument('--test', type=str, default='/opt/ml/processing/test/')
    parser.add_argument('--sm-model-dir', type=str, default='/opt/ml/processing/model/')

    return parser.parse_known_args()


#class PrintDot(tf.keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs):
#        #if epoch == 0:
#        print(str(epoch)+' ', end='')


def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

        
if __name__ == '__main__':
    
    # if we use a processing step to train, install tensorflow first
    install('tensorflow==2.9.2')
    import tensorflow as tf
    
    args, _ = parse_args()
    
    print('args are:', args)
    
    target = 'home_win'
    
    x_train = pd.read_csv(os.path.join(args.train, 'train.csv'))
    x_test = pd.read_csv(os.path.join(args.test, 'test.csv'))
    
    y_train = x_train.pop(target)
    y_test = x_test.pop(target)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # y_train = np.load(os.path.join(args.train, 'y_train.npy'))
    # y_val = np.load(os.path.join(args.val, 'y_val.npy'))
    # y_test = np.load(os.path.join(args.test, 'y_test.npy'))
    
    # set random seed
    tf.random.set_seed(16)

    # create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        #tf.keras.layers.Dropout(0.1), 
        tf.keras.layers.Dense(32, activation='relu'),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(4, activation='relu'),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid') # output shape is 1
    ])

    # compile the model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )

    # fit the model
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        validation_data=(x_test, y_test)
    ) # see how the model performs on the test set during training

    model_assets = {
        'model': model,
        'history': pd.DataFrame.from_dict(history.history),
        'name': 'model_'+str(int(datetime.datetime.now().timestamp()*100000))+'_'+str(history.history['val_accuracy'][-1])
    }
    
    name = 'model'
    tar_name = 'model.tar.gz'
    local_path = args.sm_model_dir
    model.save(os.path.join(local_path, name, '1'))
    # tf.keras.models.save_model(model_assets['model'], os.path.join(path, name, '00001'))
    tar = tarfile.open(os.path.join(local_path, tar_name), 'w:gz')
    for file_name in glob.glob(os.path.join(local_path, '*')):
        print('Adding %s...' % file_name)
        tar.add(file_name, os.path.basename(file_name))
    tar.close()

    try:
        shutil.rmtree('/opt/ml/processing/model/model') 
    except:
        pass
    
    # store model accuracy in a dict (JSON)
    my_result = {'accuracy': history.history['val_accuracy'][-1]}
    
    # option 1) write JSON to S3 via buffer
    output_buffer = StringIO()
    output_buffer.write(json.dumps(my_result))
    s3_resource.Object(bucket, 'nba/eval/accuracy.json').put(Body=output_buffer.getvalue())

    # option 2) write JSON inside sagemaker session for use in next steps
    output_dir = '/opt/ml/processing/evaluation'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = f'{output_dir}/evaluation.json'
    with open(evaluation_path, 'w') as f:
       f.write(json.dumps(my_result))
