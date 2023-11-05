import datetime
import json
import os

import boto3

#USER_STORAGE_URL = "https://docapi.serverless.yandexcloud.net/ru-central1/b1glupep1uovle4uth3q/etnovnnefhk3ghissequ" # face swap
import requests
from env import USER_STORAGE_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

#USER_STORAGE_URL = "https://docapi.serverless.yandexcloud.net/ru-central1/b1glupep1uovle4uth3q/etn5dv48lab0thk62hhs" # PoSt
#AWS_ACCESS_KEY_ID = "YCAJE_7Izbb_v7SWsaczoAeyx"
#AWS_SECRET_ACCESS_KEY = "YCNoehRBIKHS3akms-kz9jxxgtgllTSV66VWbOmz"

def create_user_table():
    dynamodb = boto3.resource(
        'dynamodb',
        endpoint_url=USER_STORAGE_URL,
        region_name = 'ru-central1',
        aws_access_key_id = AWS_ACCESS_KEY_ID,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        )

    table = dynamodb.create_table(
        TableName = 'Users',
        #TableName='Language',
        KeySchema=[
            {
                'AttributeName': 'user_id',
                'KeyType': 'HASH'  # Partition key
            }
        ],
        AttributeDefinitions=[
            {'AttributeName': 'user_id', 'AttributeType': 'S'}
        ]
    )


    table = dynamodb.create_table(
        #TableName = 'Users',
        TableName='Language',
        KeySchema=[
            {
                'AttributeName': 'user_id',
                'KeyType': 'HASH' # Partition key
            }
        ],
        AttributeDefinitions=[
            {'AttributeName': 'user_id', 'AttributeType': 'S'}
        ]
    )
    return table



def create_advertisement_table():
    dynamodb = boto3.resource(
        'dynamodb',
        endpoint_url=USER_STORAGE_URL,
        region_name = 'ru-central1',
        aws_access_key_id = AWS_ACCESS_KEY_ID,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        )
    table = dynamodb.create_table(
        #TableName = 'Users',
        TableName='Advertisement',
        KeySchema=[
            {
                'AttributeName': 'ad_id',
                'KeyType': 'HASH' # Partition key
            }
        ],
        AttributeDefinitions=[
            {'AttributeName': 'ad_id', 'AttributeType': 'S'}
        ]
    )



    return table


def add_advertisement_table(ad_id, on_off = True, language = 'eng', type = 'url'):
    dynamodb = boto3.resource(
        'dynamodb',
        endpoint_url=USER_STORAGE_URL,
        region_name='ru-central1',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    table = dynamodb.Table('Advertisement')
    on_off = int(on_off)

    response = table.put_item(
        Item={
            'ad_id': str(ad_id),
            'active': str(on_off),
            'language' : str(language),
            'type' : str(type)
        }
    )
    return response


def on_off_advertisement_table(ad_id, on_off = True):
    dynamodb = boto3.resource(
        'dynamodb',
        endpoint_url=USER_STORAGE_URL,
        region_name='ru-central1',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    table = dynamodb.Table('Advertisement')
    on_off = int(on_off)

    response = table.update_item(
        Key={'ad_id': str(ad_id)},
        UpdateExpression="set active=:r",
        ExpressionAttributeValues={
            ':r': str(on_off)}
    )
    return response

def create_tokens_table():
    dynamodb = boto3.resource(
        'dynamodb',
        endpoint_url=USER_STORAGE_URL,
        region_name = 'ru-central1',
        aws_access_key_id = AWS_ACCESS_KEY_ID,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        )
    table = dynamodb.create_table(
        #TableName = 'Users',
        TableName='Tokens',
        KeySchema=[
            {
                'AttributeName': 'token_name',
                'KeyType': 'HASH' # Partition key
            }
        ],
        AttributeDefinitions=[
            {'AttributeName': 'token_name', 'AttributeType': 'S'}
        ]
    )

    response = table.put_item(
        Item={
            'token_name': 'oath_token',
            'value': 'y0_AgAAAAAJsguRAATuwQAAAADdJXSo3oLNM7_TThK04dqPw2L_64BvI8A',
            'date': str(datetime.datetime.now())
        }
    )

    response = table.put_item(
        Item={
            'token_name': 'iam_token',
            'value': '',
            'date': str(datetime.datetime.now())
        }
    )
    return response



def get_iam_token_from_oath(oauth_token = 'y0_AgAAAAAJsguRAATuwQAAAADdJXSo3oLNM7_TThK04dqPw2L_64BvI8A', iam_url = 'https://iam.api.cloud.yandex.net/iam/v1/tokens'):
    response = requests.post(iam_url, json={"yandexPassportOauthToken": oauth_token})
    json_data = json.loads(response.text)
    if json_data is not None and 'iamToken' in json_data:
        return json_data['iamToken']
    return None

def get_update_iam_token():
    dynamodb = boto3.resource(
        'dynamodb',
        endpoint_url=USER_STORAGE_URL,
        region_name='ru-central1',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    table = dynamodb.Table('Tokens')
    response = table.get_item(Key={'token_name': str('iam_token')})

    date_reg = response['Item']['date']
    iam_token = response['Item']['value']
    date_now = str(datetime.datetime.now())

    if date_now.split()[0] == date_reg.split()[0] and len(iam_token) != 0 and \
            abs(int(date_now.split()[1].split(':')[0]) - int(date_reg.split()[1].split(':')[0])) <= 3:
                return iam_token






    oath_token = table.get_item(Key={'token_name': str('oath_token')})['Item']['value']
    iam_token = get_iam_token_from_oath(oath_token)

    response = table.update_item(
        Key={'token_name': 'iam_token'},
        UpdateExpression="set value=:v, date=:d",
        ExpressionAttributeValues={
            ':v': iam_token,
            ':d': date_now
        }
    )

    return iam_token





#create_tokens_table()
#get_update_iam_token()





#create_advertisement_table()
#add_advertisement_table('228', False, 'rus', 'banner')
#add_advertisement_table('1337', False, 'eng', 'url')
#on_off_advertisement_table('228', False)
#on_off_advertisement_table('1337', False)




#create_user_table()
