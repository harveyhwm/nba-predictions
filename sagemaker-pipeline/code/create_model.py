import argparse
import boto3
import json
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint_instance_type', type=str)
    parser.add_argument('--endpoint_config_name', type=str)
    parser.add_argument('--endpoint_name', type=str)
    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()
    
    sm = boto3.client('sagemaker', region_name='us-west-2')

    role = params['sm_role']
    container_def = {
        'Image': '763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.9.2-cpu',
        'Environment': {},
        'ModelDataUrl': 's3:.....model.tar.gz' # (check and make dynamic)
    }

    sm.create_model(ModelName=args.model_name, ExecutionRoleArn=role, PrimaryContainer=container_def)

    current_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    endpoint_instance_type = args.endpoint_instance_type
    endpoint_config_name = args.endpoint_config_name
    endpoint_name = args.endpoint_name
    model_name = args.model_name

    create_endpoint_config_response = sm.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ProductionVariants = [
            {
                'InstanceType': endpoint_instance_type,
                'InitialVariantWeight': 1,
                'InitialInstanceCount': 1,
                'ModelName': model_name,
                'VariantName': 'AllTraffic',
            }
        ],
    )
    print(f'create_endpoint_config_response: {create_endpoint_config_response}')

    list_endpoints_response = sm.list_endpoints(
        SortBy='CreationTime',
        SortOrder='Descending',
        NameContains=endpoint_name,
    )
    print(f'list_endpoints_response: {list_endpoints_response}')

    if len(list_endpoints_response['Endpoints']) > 0:
        print('Updating Endpoint with new Endpoint Configuration')
        update_endpoint_response = sm.update_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        print(f'update_endpoint_response: {update_endpoint_response}')
    else:
        print('Creating Endpoint')
        create_endpoint_response = sm.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        print(f'create_endpoint_response: {create_endpoint_response}')
